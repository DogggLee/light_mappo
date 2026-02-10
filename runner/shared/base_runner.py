import csv
import json
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter

from utils.shared_buffer import SharedReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]

        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        self.model_dir = self.all_args.model_dir
        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / "models")
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.json"
        self.eval_metrics_csv = self.run_dir / "metrics_eval.csv"
        self.metrics = self._load_metrics()

        self.policy_share = getattr(self.all_args, "policy_share", True)
        self.target_policy_source = getattr(self.all_args, "target_policy_source", "train")

        from algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
        from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy

        self.role_by_agent = ["hunter"] * self.all_args.num_hunters + ["blocker"] * self.all_args.num_blockers + ["target"]
        self.train_agent_ids = [i for i in range(self.num_agents) if not (self.role_by_agent[i] == "target" and self.target_policy_source == "patrol")]

        if self.policy_share:
            groups = {}
            for aid in self.train_agent_ids:
                role = self.role_by_agent[aid]
                groups.setdefault(role, []).append(aid)
            self.policy_groups = groups
        else:
            self.policy_groups = {f"agent_{aid}": [aid] for aid in self.train_agent_ids}

        self.group_order = list(self.policy_groups.keys())

        self.policies = {}
        self.trainers = {}
        self.buffers = {}
        for group_name, agent_ids in self.policy_groups.items():
            share_observation_space = self.envs.share_observation_space[agent_ids[0]] if self.use_centralized_V else self.envs.observation_space[agent_ids[0]]
            policy = Policy(
                self.all_args,
                self.envs.observation_space[agent_ids[0]],
                share_observation_space,
                self.envs.action_space[agent_ids[0]],
                device=self.device,
            )
            trainer = TrainAlgo(self.all_args, policy, device=self.device)
            buffer = SharedReplayBuffer(
                self.all_args,
                len(agent_ids),
                self.envs.observation_space[agent_ids[0]],
                share_observation_space,
                self.envs.action_space[agent_ids[0]],
            )
            self.policies[group_name] = policy
            self.trainers[group_name] = trainer
            self.buffers[group_name] = buffer

        if self.model_dir is not None:
            self.restore()

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for group_name in self.group_order:
            trainer = self.trainers[group_name]
            buffer = self.buffers[group_name]
            trainer.prep_rollout()
            next_values = trainer.policy.get_values(
                np.concatenate(buffer.share_obs[-1]),
                np.concatenate(buffer.rnn_states_critic[-1]),
                np.concatenate(buffer.masks[-1]),
            )
            next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
            buffer.compute_returns(next_values, trainer.value_normalizer)

    def train(self):
        train_infos = {}
        for group_name in self.group_order:
            trainer = self.trainers[group_name]
            buffer = self.buffers[group_name]
            trainer.prep_training()
            train_infos[group_name] = trainer.train(buffer)
            buffer.after_update()
        return train_infos

    def save(self):
        for group_name in self.group_order:
            trainer = self.trainers[group_name]
            torch.save(trainer.policy.actor.state_dict(), f"{self.save_dir}/actor_{group_name}.pt")
            torch.save(trainer.policy.critic.state_dict(), f"{self.save_dir}/critic_{group_name}.pt")

    def restore(self):
        for group_name in self.group_order:
            actor_state_dict = torch.load(f"{self.model_dir}/actor_{group_name}.pt", map_location=self.device)
            self.policies[group_name].actor.load_state_dict(actor_state_dict)
            if not self.all_args.use_render:
                critic_state_dict = torch.load(f"{self.model_dir}/critic_{group_name}.pt", map_location=self.device)
                self.policies[group_name].critic.load_state_dict(critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for group_name, info in train_infos.items():
            for k, v in info.items():
                key = f"{group_name}/{k}"
                self.writter.add_scalars(key, {key: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if np.isscalar(v):
                self.writter.add_scalars(k, {k: float(v)}, total_num_steps)
            else:
                if len(v) > 0:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def _load_metrics(self):
        if self.metrics_path.exists():
            try:
                return json.loads(self.metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"train": [], "eval": []}

    def _save_metrics(self):
        self.metrics_path.write_text(json.dumps(self.metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    def record_train_metrics(self, total_num_steps, average_episode_rewards):
        self.metrics.setdefault("train", []).append({
            "total_num_steps": int(total_num_steps),
            "average_episode_rewards": float(average_episode_rewards),
        })
        self._save_metrics()

    def record_eval_metrics(self, total_num_steps, eval_average_episode_rewards, capture_success_rate, avg_capture_steps):
        self.metrics.setdefault("eval", []).append({
            "total_num_steps": int(total_num_steps),
            "eval_average_episode_rewards": float(eval_average_episode_rewards),
            "capture_success_rate": float(capture_success_rate),
            "avg_capture_steps": float(avg_capture_steps) if avg_capture_steps is not None else None,
        })
        self._save_metrics()
        self._append_eval_metrics_csv(total_num_steps, eval_average_episode_rewards, capture_success_rate, avg_capture_steps)

    def _append_eval_metrics_csv(self, total_num_steps, eval_average_episode_rewards, capture_success_rate, avg_capture_steps):
        write_header = not self.eval_metrics_csv.exists()
        with self.eval_metrics_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["total_num_steps", "eval_average_episode_rewards", "capture_success_rate", "avg_capture_steps"])
            writer.writerow([
                int(total_num_steps),
                float(eval_average_episode_rewards),
                float(capture_success_rate),
                "" if avg_capture_steps is None else float(avg_capture_steps),
            ])
