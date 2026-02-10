import time
import numpy as np
import torch

from runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super().__init__(config)

    def _build_share_obs(self, obs, group_size):
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            return np.expand_dims(share_obs, 1).repeat(group_size, axis=1)
        return obs

    def run(self):
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for group_name in self.group_order:
                    self.trainers[group_name].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                obs, rewards, dones, infos = self.envs.step(actions_env)
                self.insert((obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic))

            self.compute()
            train_infos = self.train()
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )
                avg_rewards = []
                for group_name in self.group_order:
                    avg_rewards.append(np.mean(self.buffers[group_name].rewards) * self.episode_length)
                train_infos["system"] = {"average_episode_rewards": float(np.mean(avg_rewards))}
                self.log_train(train_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        obs = self.envs.reset()
        for group_name, agent_ids in self.policy_groups.items():
            buffer = self.buffers[group_name]
            group_obs = obs[:, agent_ids]
            buffer.share_obs[0] = self._build_share_obs(obs, len(agent_ids)).copy()
            buffer.obs[0] = group_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        values, actions, action_log_probs, rnn_states, rnn_states_critic = {}, {}, {}, {}, {}
        actions_env = np.zeros((self.n_rollout_threads, self.num_agents, self.envs.action_space[0].shape[0]), dtype=np.float32)

        for group_name, agent_ids in self.policy_groups.items():
            trainer = self.trainers[group_name]
            buffer = self.buffers[group_name]
            trainer.prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = trainer.policy.get_actions(
                np.concatenate(buffer.share_obs[step]),
                np.concatenate(buffer.obs[step]),
                np.concatenate(buffer.rnn_states[step]),
                np.concatenate(buffer.rnn_states_critic[step]),
                np.concatenate(buffer.masks[step]),
            )
            values[group_name] = np.array(np.split(_t2n(value), self.n_rollout_threads))
            actions[group_name] = np.array(np.split(_t2n(action), self.n_rollout_threads))
            action_log_probs[group_name] = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
            rnn_states[group_name] = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
            rnn_states_critic[group_name] = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
            actions_env[:, agent_ids, :] = actions[group_name]

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        for group_name, agent_ids in self.policy_groups.items():
            buffer = self.buffers[group_name]
            group_dones = dones[:, agent_ids]
            group_rnn_states = rnn_states[group_name]
            group_rnn_states[group_dones] = np.zeros(((group_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            group_rnn_states_critic = rnn_states_critic[group_name]
            group_rnn_states_critic[group_dones] = np.zeros(((group_dones == True).sum(), *buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

            masks = np.ones((self.n_rollout_threads, len(agent_ids), 1), dtype=np.float32)
            masks[group_dones] = 0.0

            share_obs = self._build_share_obs(obs, len(agent_ids))
            group_obs = obs[:, agent_ids]
            group_rewards = rewards[:, agent_ids]
            if group_rewards.ndim == 2:
                group_rewards = group_rewards[..., None]

            buffer.insert(
                share_obs,
                group_obs,
                group_rnn_states,
                group_rnn_states_critic,
                actions[group_name],
                action_log_probs[group_name],
                values[group_name],
                group_rewards,
                masks,
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()
        eval_rnn_states = {
            group_name: np.zeros((self.n_eval_rollout_threads, len(agent_ids), self.recurrent_N, self.hidden_size), dtype=np.float32)
            for group_name, agent_ids in self.policy_groups.items()
        }
        eval_masks = {
            group_name: np.ones((self.n_eval_rollout_threads, len(agent_ids), 1), dtype=np.float32)
            for group_name, agent_ids in self.policy_groups.items()
        }

        for _ in range(self.episode_length):
            eval_actions_env = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.eval_envs.action_space[0].shape[0]), dtype=np.float32)
            for group_name, agent_ids in self.policy_groups.items():
                trainer = self.trainers[group_name]
                trainer.prep_rollout()
                eval_action, group_rnn = trainer.policy.act(
                    np.concatenate(eval_obs[:, agent_ids]),
                    np.concatenate(eval_rnn_states[group_name]),
                    np.concatenate(eval_masks[group_name]),
                    deterministic=True,
                )
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
                eval_rnn_states[group_name] = np.array(np.split(_t2n(group_rnn), self.n_eval_rollout_threads))
                eval_actions_env[:, agent_ids, :] = eval_actions

            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            for group_name, agent_ids in self.policy_groups.items():
                group_dones = eval_dones[:, agent_ids]
                eval_rnn_states[group_name][group_dones] = 0.0
                eval_masks[group_name] = np.ones((self.n_eval_rollout_threads, len(agent_ids), 1), dtype=np.float32)
                eval_masks[group_name][group_dones] = 0.0

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {"eval_average_episode_rewards": np.sum(eval_episode_rewards, axis=0)}
        self.log_env(eval_env_infos, total_num_steps)
