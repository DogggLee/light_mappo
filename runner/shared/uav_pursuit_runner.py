import json
import time
from pathlib import Path

import yaml
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.collections import LineCollection

from envs.uav_pursuit_env import MultiUavPursuitEnv
from runner.shared.env_runner import EnvRunner


def _t2n(x):
    return x.detach().cpu().numpy()


class UavPursuitRunner(EnvRunner):
    def __init__(self, config):
        super().__init__(config)
        self.gif_interval = getattr(self.all_args, "gif_interval", 10)
        self.gif_frame_duration = getattr(self.all_args, "gif_frame_duration", 0.1)
        self.gif_dir = Path(self.run_dir) / "gifs"
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        # 评估 GIF 输出目录：每个场景单独落在 val_{idx} 子目录。
        self.eval_gif_dir = Path(self.run_dir) / "eval_gifs"
        self.eval_gif_dir.mkdir(parents=True, exist_ok=True)
        self.target_patrol_names = self._normalize_name_list(getattr(self.all_args, "target_patrol_names", None))
        self.target_patrol_switch_interval = int(getattr(self.all_args, "target_patrol_switch_interval", 0))
        self.eval_random_patrol_routes = int(getattr(self.all_args, "eval_random_patrol_routes", 0))
        self.eval_random_patrol_points = int(getattr(self.all_args, "eval_random_patrol_points", 4))
        self._patrol_rng = np.random.RandomState(self.all_args.seed + 2027)
        self._current_patrol_name = None
        self._best_train_avg_reward = None
        self._best_eval_avg_reward = None
        self._best_capture_success_rate = None
        self._best_avg_capture_steps = None
        # 评估场景列表：为空时退化为默认单场景评估。
        self.scenario_suite = getattr(self.all_args, "scenario_suite_data", None) or []
        self._target_eval_policy_cache = {}

    def _maybe_report_best_metrics(
        self,
        total_num_steps,
        train_avg_reward=None,
        eval_avg_reward=None,
        capture_success_rate=None,
        avg_capture_steps=None,
    ):
        lines = []

        if train_avg_reward is not None:
            if self._best_train_avg_reward is None or train_avg_reward > self._best_train_avg_reward:
                self._best_train_avg_reward = float(train_avg_reward)
                lines.append(f"best_train_avg_reward: {self._best_train_avg_reward:.4f}")

        if eval_avg_reward is not None:
            if self._best_eval_avg_reward is None or eval_avg_reward > self._best_eval_avg_reward:
                self._best_eval_avg_reward = float(eval_avg_reward)
                lines.append(f"best_eval_avg_reward: {self._best_eval_avg_reward:.4f}")

        if capture_success_rate is not None:
            if self._best_capture_success_rate is None or capture_success_rate > self._best_capture_success_rate:
                self._best_capture_success_rate = float(capture_success_rate)
                lines.append(f"best_capture_success_rate: {self._best_capture_success_rate:.4f}")

        if avg_capture_steps is not None:
            if self._best_avg_capture_steps is None or avg_capture_steps < self._best_avg_capture_steps:
                self._best_avg_capture_steps = float(avg_capture_steps)
                lines.append(f"best_avg_capture_steps: {self._best_avg_capture_steps:.2f}")

        if lines:
            print(f"[step {int(total_num_steps)}] " + " | ".join(lines))

    @staticmethod
    def _normalize_name_list(names):
        if names is None:
            return None
        if isinstance(names, (list, tuple)):
            cleaned = [str(n).strip() for n in names if str(n).strip()]
            return cleaned or None
        text = str(names).strip()
        if not text:
            return None
        return [n.strip() for n in text.replace(";", ",").split(",") if n.strip()]

    def _sample_patrol_name(self):
        if not self.target_patrol_names:
            return None
        return self._patrol_rng.choice(self.target_patrol_names)

    def _maybe_switch_patrol_route(self, episode_idx, force=False):
        if self.target_policy_source != "patrol":
            return
        if not self.target_patrol_names:
            return
        if force or (self.target_patrol_switch_interval > 0 and episode_idx % self.target_patrol_switch_interval == 0):
            name = self._sample_patrol_name()
            if name is not None:
                self.envs.set_target_patrol_route(name)
                if self.eval_envs is not None:
                    self.eval_envs.set_target_patrol_route(name)
                self._current_patrol_name = name

    def _generate_random_patrol_routes(self, count, points):
        routes = []
        count = max(0, int(count))
        points = max(2, int(points))
        for _ in range(count):
            route = self._patrol_rng.uniform(0.0, 1.0, size=(points, 2)).astype(np.float32)
            routes.append(route)
        return routes

    def run(self):
        self._maybe_switch_patrol_route(0, force=True)
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            self._maybe_switch_patrol_route(episode)
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
                avg_rewards = []
                for group_name in self.group_order:
                    group_avg = np.mean(self.buffers[group_name].rewards) * self.episode_length
                    avg_rewards.append(group_avg)
                    train_infos.setdefault(group_name, {})
                    train_infos[group_name]["average_episode_rewards"] = float(group_avg)
                train_infos["system"] = {"average_episode_rewards": float(np.mean(avg_rewards))}
                self.log_train(train_infos, total_num_steps)
                self.record_train_metrics(total_num_steps, train_infos["system"]["average_episode_rewards"])
                self._maybe_report_best_metrics(
                    total_num_steps,
                    train_avg_reward=train_infos["system"]["average_episode_rewards"],
                )

            if (episode + 1) % self.gif_interval == 0:
                self._save_training_gif(episode + 1)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    @torch.no_grad()
    def _save_training_gif(self, episode_idx):
        frames = self._collect_episode_frames(episode_idx, self._current_patrol_name)
        gif_path = self.gif_dir / f"episode_{episode_idx:04d}.gif"
        imageio.mimsave(str(gif_path), frames, duration=self.gif_frame_duration)

    @torch.no_grad()
    def _collect_episode_frames(self, episode_idx, patrol_name=None):
        env = MultiUavPursuitEnv(
            num_hunters=self.all_args.num_hunters,
            num_blockers=self.all_args.num_blockers,
            world_size=self.all_args.world_size,
            dt=self.all_args.dt,
            capture_radius=self.all_args.capture_radius,
            capture_steps=self.all_args.capture_steps,
            max_steps=self.episode_length,
            seed=self.all_args.seed,
            target_policy_source=self.all_args.target_policy_source,
            target_patrol_path=self.all_args.target_patrol_path,
            target_patrol_names=self.all_args.target_patrol_names,
            max_speed_hunter=self.all_args.max_speed_hunter,
            max_speed_blocker=self.all_args.max_speed_blocker,
            max_speed_target=self.all_args.max_speed_target,
            perception_hunter=self.all_args.perception_hunter,
            perception_blocker=self.all_args.perception_blocker,
            perception_target=self.all_args.perception_target,
        )
        if patrol_name:
            env.set_target_patrol_route(patrol_name)
        obs = env.reset()

        role_groups = env.role_groups
        hunter_indices = role_groups.get("hunter", [])
        blocker_indices = role_groups.get("blocker", [])
        target_index = role_groups.get("target", [None])[0]
        positions = {idx: [env.positions[idx].copy()] for idx in range(env.agent_num)}
        capture = False

        eval_rnn_states = {
            group_name: np.zeros((1, len(agent_ids), self.recurrent_N, self.hidden_size), dtype=np.float32)
            for group_name, agent_ids in self.policy_groups.items()
        }
        eval_masks = {
            group_name: np.ones((1, len(agent_ids), 1), dtype=np.float32)
            for group_name, agent_ids in self.policy_groups.items()
        }

        frames = [self._draw_frame(positions, env.world_size, env.perception_ranges, hunter_indices, blocker_indices, target_index, capture, episode_idx, step=0)]

        for step in range(1, self.episode_length + 1):
            actions_env = np.zeros((env.agent_num, env.action_dim), dtype=np.float32)
            for group_name, agent_ids in self.policy_groups.items():
                trainer = self.trainers[group_name]
                trainer.prep_rollout()
                action, next_rnn = trainer.policy.act(
                    obs[agent_ids],
                    eval_rnn_states[group_name][0],
                    eval_masks[group_name][0],
                    deterministic=True,
                )
                actions_env[agent_ids] = _t2n(action)
                eval_rnn_states[group_name][0] = _t2n(next_rnn)

            obs, rewards, dones, infos = env.step(actions_env)
            capture = capture or any(info.get("capture", False) for info in infos)
            for idx in range(env.agent_num):
                positions[idx].append(env.positions[idx].copy())

            frames.append(self._draw_frame(positions, env.world_size, env.perception_ranges, hunter_indices, blocker_indices, target_index, capture, episode_idx, step=step))

            for group_name, agent_ids in self.policy_groups.items():
                group_dones = dones[agent_ids]
                eval_rnn_states[group_name][0][group_dones] = 0.0
                eval_masks[group_name][0] = np.ones((len(agent_ids), 1), dtype=np.float32)
                eval_masks[group_name][0][group_dones] = 0.0

            if np.all(dones):
                break

        env.close()
        return frames

    def _build_target_eval_policy(self, model_path):
        # 懒加载外部 Target policy（仅评估使用），避免重复加载模型文件。
        if not model_path:
            return None
        model_path = str(Path(model_path).resolve())
        if model_path in self._target_eval_policy_cache:
            return self._target_eval_policy_cache[model_path]

        from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy

        target_idx = self.num_agents - 1
        share_observation_space = self.eval_envs.share_observation_space[target_idx] if self.use_centralized_V else self.eval_envs.observation_space[target_idx]
        target_policy = Policy(
            self.all_args,
            self.eval_envs.observation_space[target_idx],
            share_observation_space,
            self.eval_envs.action_space[target_idx],
            device=self.device,
        )
        actor_state_dict = torch.load(model_path, map_location=self.device)
        target_policy.actor.load_state_dict(actor_state_dict)
        target_policy.actor.eval()
        self._target_eval_policy_cache[model_path] = target_policy
        return target_policy

    def _resolve_target_model_path(self, scenario):
        # 支持绝对路径、相对场景文件路径、以及目录（自动查找 actor_target.pt）。
        raw_path = scenario.get("target_policy_model_path")
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_absolute() and scenario.get("scenario_file"):
            path = Path(scenario["scenario_file"]).resolve().parent / path
        path = path.resolve()
        if path.is_dir():
            candidate = path / "actor_target.pt"
            if candidate.exists():
                return str(candidate)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Target policy model not found: {path}")

    def _resolve_patrol_route_path(self, scenario):
        # 巡逻路径按序号组织在 datasets/<split>/patrol_routes/<idx>.yaml。
        route_id = scenario.get("target_patrol_route_id")
        if route_id in (None, "", "null"):
            return None
        route_stem = str(route_id).strip()
        # 兼容 route_id 被 YAML 解析为整数的情况（如 001 -> 1）。
        route_stem_candidates = [route_stem]
        if route_stem.isdigit():
            route_stem_candidates.append(route_stem.zfill(3))
        if scenario.get("scenario_file"):
            base_dir = Path(scenario["scenario_file"]).resolve().parent / "patrol_routes"
        else:
            base_dir = Path("datasets")
        for stem in route_stem_candidates:
            for ext in (".yaml", ".yml", ".json"):
                candidate = base_dir / f"{stem}{ext}"
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(f"Patrol route file not found for route_id={route_stem} in {base_dir}")

    def _load_patrol_waypoints(self, route_path):
        # 支持 yaml/json 两种巡逻路径文件，并统一提取 waypoints 列表。
        if route_path is None:
            return None
        text = route_path.read_text(encoding="utf-8")
        if route_path.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            payload = yaml.safe_load(text) or {}
        if isinstance(payload, dict):
            waypoints = payload.get("waypoints", payload)
        else:
            waypoints = payload
        return waypoints

    @torch.no_grad()
    def _save_eval_scenario_gif(self, total_num_steps, scenario, scenario_index, target_mode, target_eval_policy=None):
        # 每个场景/模式保存独立 GIF，目录形如 eval_gifs/val_{idx}/。
        scenario_env_cfg = dict(scenario)
        scenario_env_cfg["target_policy_source"] = target_mode

        env = MultiUavPursuitEnv(
            num_hunters=self.all_args.num_hunters,
            num_blockers=self.all_args.num_blockers,
            world_size=self.all_args.world_size,
            dt=self.all_args.dt,
            capture_radius=self.all_args.capture_radius,
            capture_steps=self.all_args.capture_steps,
            max_steps=self.episode_length,
            seed=self.all_args.seed,
            target_policy_source=target_mode,
            target_patrol_path=self.all_args.target_patrol_path,
            target_patrol_names=self.all_args.target_patrol_names,
            max_speed_hunter=self.all_args.max_speed_hunter,
            max_speed_blocker=self.all_args.max_speed_blocker,
            max_speed_target=self.all_args.max_speed_target,
            perception_hunter=self.all_args.perception_hunter,
            perception_blocker=self.all_args.perception_blocker,
            perception_target=self.all_args.perception_target,
        )
        env.apply_scenario_config(scenario_env_cfg)
        # patrol 模式按场景 route_id 覆盖巡逻路径，避免依赖全局 patrol 文件。
        if target_mode == "patrol":
            route_path = self._resolve_patrol_route_path(scenario)
            route_waypoints = self._load_patrol_waypoints(route_path)
            if route_waypoints is not None:
                env.set_target_patrol_waypoints(route_waypoints, route_name=f"route_{scenario.get('target_patrol_route_id')}")
        obs = env.reset()

        role_groups = env.role_groups
        hunter_indices = role_groups.get("hunter", [])
        blocker_indices = role_groups.get("blocker", [])
        target_index = role_groups.get("target", [None])[0]
        positions = {idx: [env.positions[idx].copy()] for idx in range(env.agent_num)}
        capture = False

        eval_rnn_states = {
            group_name: np.zeros((1, len(agent_ids), self.recurrent_N, self.hidden_size), dtype=np.float32)
            for group_name, agent_ids in self.policy_groups.items()
        }
        eval_masks = {
            group_name: np.ones((1, len(agent_ids), 1), dtype=np.float32)
            for group_name, agent_ids in self.policy_groups.items()
        }

        target_needs_external_policy = target_mode == "train" and all(target_index not in ids for ids in self.policy_groups.values())
        target_rnn_states = np.zeros((1, 1, self.recurrent_N, self.hidden_size), dtype=np.float32)
        target_masks = np.ones((1, 1, 1), dtype=np.float32)

        scenario_episode_length = int(scenario.get("episode_length", self.episode_length))
        frames = [self._draw_frame(positions, env.world_size, env.perception_ranges, hunter_indices, blocker_indices, target_index, capture, scenario_index, step=0)]

        for step in range(1, scenario_episode_length + 1):
            actions_env = np.zeros((env.agent_num, env.action_dim), dtype=np.float32)
            for group_name, agent_ids in self.policy_groups.items():
                trainer = self.trainers[group_name]
                trainer.prep_rollout()
                action, next_rnn = trainer.policy.act(
                    obs[agent_ids],
                    eval_rnn_states[group_name][0],
                    eval_masks[group_name][0],
                    deterministic=True,
                )
                actions_env[agent_ids] = _t2n(action)
                eval_rnn_states[group_name][0] = _t2n(next_rnn)

            if target_needs_external_policy:
                if target_eval_policy is None:
                    raise ValueError(f"Scenario {scenario.get('scenario_id', scenario_index)} requires target_policy_model_path when evaluating train mode.")
                target_action, next_target_rnn = target_eval_policy.act(
                    obs[[target_index]],
                    target_rnn_states[0],
                    target_masks[0],
                    deterministic=True,
                )
                actions_env[target_index] = _t2n(target_action)[0]
                target_rnn_states[0] = _t2n(next_target_rnn)

            obs, rewards, dones, infos = env.step(actions_env)
            capture = capture or any(info.get("capture", False) for info in infos)
            for idx in range(env.agent_num):
                positions[idx].append(env.positions[idx].copy())

            frames.append(self._draw_frame(positions, env.world_size, env.perception_ranges, hunter_indices, blocker_indices, target_index, capture, scenario_index, step=step))

            for group_name, agent_ids in self.policy_groups.items():
                group_dones = dones[agent_ids]
                eval_rnn_states[group_name][0][group_dones] = 0.0
                eval_masks[group_name][0] = np.ones((len(agent_ids), 1), dtype=np.float32)
                eval_masks[group_name][0][group_dones] = 0.0

            if target_needs_external_policy:
                target_done = bool(dones[target_index])
                if target_done:
                    target_rnn_states[0][:] = 0.0
                    target_masks[0][:] = 0.0
                else:
                    target_masks[0][:] = 1.0

            if np.all(dones):
                break

        scenario_folder = self.eval_gif_dir / f"val_{scenario_index}"
        scenario_folder.mkdir(parents=True, exist_ok=True)
        scenario_id = scenario.get("scenario_id", f"scenario_{scenario_index}")
        gif_path = scenario_folder / f"{scenario_id}_{target_mode}_step_{int(total_num_steps)}.gif"
        imageio.mimsave(str(gif_path), frames, duration=self.gif_frame_duration)
        env.close()

    @torch.no_grad()
    def eval(self, total_num_steps):
        # 评估阶段按场景逐个执行，输出分场景指标并落盘。
        eval_episodes = int(self.all_args.eval_episodes)
        scenarios = self.scenario_suite if self.scenario_suite else [{"scenario_id": "default", "eval_target_modes": [self.target_policy_source]}]

        for scenario_index, scenario in enumerate(scenarios):
            scenario_id = str(scenario.get("scenario_id", f"scenario_{scenario_index}"))
            scenario_hunters = int(scenario.get("num_hunters", self.all_args.num_hunters))
            scenario_blockers = int(scenario.get("num_blockers", self.all_args.num_blockers))
            if scenario_hunters != self.all_args.num_hunters or scenario_blockers != self.all_args.num_blockers:
                raise ValueError(
                    f"Scenario {scenario_id} agent layout mismatch: "
                    f"num_hunters={scenario_hunters}, num_blockers={scenario_blockers}, "
                    f"expected {self.all_args.num_hunters}/{self.all_args.num_blockers}"
                )

            eval_target_modes = scenario.get("eval_target_modes", [scenario.get("target_policy_source", self.target_policy_source)])
            target_model_path = self._resolve_target_model_path(scenario)
            target_eval_policy = self._build_target_eval_policy(target_model_path) if target_model_path else None

            for target_mode in eval_target_modes:
                if target_mode not in {"patrol", "train"}:
                    raise ValueError(f"Unsupported target mode: {target_mode}")

                scenario_env_cfg = dict(scenario)
                scenario_env_cfg["target_policy_source"] = target_mode
                # 每个场景开始前同步覆盖评估环境参数。
                self.eval_envs.apply_scenario_config(scenario_env_cfg)
                scenario_episode_length = int(scenario.get("episode_length", self.episode_length))

                eval_episode_rewards = []
                capture_flags = []
                capture_steps = []
                random_routes = None

                if target_mode == "patrol" and self.eval_random_patrol_routes > 0:
                    random_routes = self._generate_random_patrol_routes(self.eval_random_patrol_routes, self.eval_random_patrol_points)

                target_idx = self.num_agents - 1
                target_needs_external_policy = target_mode == "train" and all(target_idx not in ids for ids in self.policy_groups.values())
                if target_needs_external_policy and target_eval_policy is None:
                    raise ValueError(f"Scenario {scenario_id} requires target_policy_model_path for train mode evaluation.")

                for ep in range(eval_episodes):
                    if target_mode == "patrol":
                        if random_routes:
                            route = random_routes[self._patrol_rng.randint(0, len(random_routes))]
                            self.eval_envs.set_target_patrol_waypoints(route, route_name=f"{scenario_id}_{target_mode}_eval_random_{ep}")
                        else:
                            # 优先使用场景 route_id 指定的巡逻路径文件。
                            route_path = self._resolve_patrol_route_path(scenario)
                            route_waypoints = self._load_patrol_waypoints(route_path)
                            if route_waypoints is not None:
                                self.eval_envs.set_target_patrol_waypoints(route_waypoints, route_name=f"route_{scenario.get('target_patrol_route_id')}")
                            elif self.target_patrol_names:
                                name = self._sample_patrol_name()
                                if name is not None:
                                    self.eval_envs.set_target_patrol_route(name)

                    eval_obs = self.eval_envs.reset()
                    eval_rnn_states = {
                        group_name: np.zeros((self.n_eval_rollout_threads, len(agent_ids), self.recurrent_N, self.hidden_size), dtype=np.float32)
                        for group_name, agent_ids in self.policy_groups.items()
                    }
                    eval_masks = {
                        group_name: np.ones((self.n_eval_rollout_threads, len(agent_ids), 1), dtype=np.float32)
                        for group_name, agent_ids in self.policy_groups.items()
                    }

                    target_rnn_states = np.zeros((self.n_eval_rollout_threads, 1, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    target_masks = np.ones((self.n_eval_rollout_threads, 1, 1), dtype=np.float32)

                    episode_done = np.zeros(self.n_eval_rollout_threads, dtype=bool)
                    episode_steps = np.zeros(self.n_eval_rollout_threads, dtype=np.int32)
                    episode_capture = np.zeros(self.n_eval_rollout_threads, dtype=bool)
                    episode_capture_step = np.full(self.n_eval_rollout_threads, -1, dtype=np.int32)
                    ep_rewards = np.zeros((self.n_eval_rollout_threads, self.num_agents), dtype=np.float32)

                    for _ in range(scenario_episode_length):
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

                        if target_needs_external_policy:
                            target_action, next_target_rnn = target_eval_policy.act(
                                np.concatenate(eval_obs[:, [target_idx]]),
                                np.concatenate(target_rnn_states),
                                np.concatenate(target_masks),
                                deterministic=True,
                            )
                            target_actions = np.array(np.split(_t2n(target_action), self.n_eval_rollout_threads))
                            target_rnn_states = np.array(np.split(_t2n(next_target_rnn), self.n_eval_rollout_threads))
                            eval_actions_env[:, target_idx, :] = target_actions[:, 0, :]

                        eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
                        ep_rewards += eval_rewards

                        for env_i in range(self.n_eval_rollout_threads):
                            if not episode_done[env_i]:
                                episode_steps[env_i] += 1
                                capture = any(info.get("capture", False) for info in eval_infos[env_i])
                                if capture and not episode_capture[env_i]:
                                    episode_capture[env_i] = True
                                    episode_capture_step[env_i] = episode_steps[env_i]
                                if np.all(eval_dones[env_i]):
                                    episode_done[env_i] = True

                        for group_name, agent_ids in self.policy_groups.items():
                            group_dones = eval_dones[:, agent_ids]
                            eval_rnn_states[group_name][group_dones] = 0.0
                            eval_masks[group_name] = np.ones((self.n_eval_rollout_threads, len(agent_ids), 1), dtype=np.float32)
                            eval_masks[group_name][group_dones] = 0.0

                        if target_needs_external_policy:
                            target_dones = eval_dones[:, [target_idx]]
                            target_rnn_states[target_dones] = 0.0
                            target_masks = np.ones((self.n_eval_rollout_threads, 1, 1), dtype=np.float32)
                            target_masks[target_dones] = 0.0

                        if np.all(episode_done):
                            break

                    eval_episode_rewards.append(ep_rewards)
                    capture_flags.extend(episode_capture.tolist())
                    capture_steps.extend([int(cs) for cs in episode_capture_step if cs > 0])

                eval_avg_rewards = float(np.mean([np.mean(ep) for ep in eval_episode_rewards])) if eval_episode_rewards else 0.0
                total_episodes = max(1, eval_episodes * self.n_eval_rollout_threads)
                capture_success_rate = float(np.sum(capture_flags) / total_episodes) if capture_flags else 0.0
                avg_capture_steps = float(np.mean(capture_steps)) if capture_steps else None

                metric_id = f"{scenario_id}/{target_mode}"
                eval_env_infos = {
                    f"eval/{metric_id}/average_episode_rewards": eval_avg_rewards,
                    f"eval/{metric_id}/capture_success_rate": capture_success_rate,
                }
                if avg_capture_steps is not None:
                    eval_env_infos[f"eval/{metric_id}/avg_capture_steps"] = avg_capture_steps

                if eval_episode_rewards:
                    all_ep_rewards = np.concatenate(eval_episode_rewards, axis=0)
                    for group_name, agent_ids in self.policy_groups.items():
                        if not agent_ids:
                            continue
                        role_avg = float(np.mean(all_ep_rewards[:, agent_ids]))
                        eval_env_infos[f"eval/{metric_id}/average_episode_rewards/{group_name}"] = role_avg

                # 控制台输出简要分场景评估摘要，便于快速查看回归结果。
                print(
                    f"[eval][{metric_id}] success_rate={capture_success_rate:.3f}, "
                    + (f"avg_capture_steps={avg_capture_steps:.2f}, " if avg_capture_steps is not None else "avg_capture_steps=NA, ")
                    + f"avg_reward={eval_avg_rewards:.3f}"
                )
                self.log_env(eval_env_infos, total_num_steps)
                self.record_eval_metrics(total_num_steps, eval_avg_rewards, capture_success_rate, avg_capture_steps, scenario_id=metric_id)
                self._save_eval_scenario_gif(total_num_steps, scenario, scenario_index, target_mode, target_eval_policy=target_eval_policy)
                self._maybe_report_best_metrics(
                    total_num_steps,
                    eval_avg_reward=eval_avg_rewards,
                    capture_success_rate=capture_success_rate,
                    avg_capture_steps=avg_capture_steps,
                )

    def _draw_fade_traj(self, ax, traj, color):
        if len(traj) < 2:
            return
        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        alpha = np.linspace(0.15, 0.9, len(segments))
        rgba = np.tile(plt.matplotlib.colors.to_rgba(color), (len(segments), 1))
        rgba[:, 3] = alpha
        lc = LineCollection(segments, colors=rgba, linewidths=2.0)
        ax.add_collection(lc)

    def _draw_frame(self, positions, world_size, perception_ranges, hunter_indices, blocker_indices, target_index, capture, episode_idx, step):
        fig, ax = plt.subplots(figsize=(6.8, 6.8), dpi=140)
        ax.set_xlim(-world_size, world_size)
        ax.set_ylim(-world_size, world_size)
        ax.set_aspect("equal")
        ax.set_title(f"Episode {episode_idx} - Step {step}")
        ax.grid(True, linestyle="--", alpha=0.25)

        palette = {"hunter": "#1f77b4", "blocker": "#2ca02c", "target": "#d62728"}

        for i, idx in enumerate(hunter_indices):
            traj = np.array(positions[idx])
            self._draw_fade_traj(ax, traj, palette["hunter"])
            ax.scatter(traj[-1, 0], traj[-1, 1], color=palette["hunter"], s=45)
            pr = plt.Circle((traj[-1, 0], traj[-1, 1]), perception_ranges["hunter"], color=palette["hunter"], alpha=0.08)
            ax.add_patch(pr)
            ax.text(traj[-1, 0], traj[-1, 1], f"H{i+1}", fontsize=8)

        for i, idx in enumerate(blocker_indices):
            traj = np.array(positions[idx])
            self._draw_fade_traj(ax, traj, palette["blocker"])
            ax.scatter(traj[-1, 0], traj[-1, 1], color=palette["blocker"], marker="s", s=42)
            pr = plt.Circle((traj[-1, 0], traj[-1, 1]), perception_ranges["blocker"], color=palette["blocker"], alpha=0.07)
            ax.add_patch(pr)
            ax.text(traj[-1, 0], traj[-1, 1], f"B{i+1}", fontsize=8)

        if target_index is not None:
            traj = np.array(positions[target_index])
            self._draw_fade_traj(ax, traj, palette["target"])
            ax.scatter(traj[-1, 0], traj[-1, 1], color=palette["target"], marker="*", s=95)
            pr = plt.Circle((traj[-1, 0], traj[-1, 1]), perception_ranges["target"], color=palette["target"], alpha=0.08)
            ax.add_patch(pr)
            ax.text(traj[-1, 0], traj[-1, 1], "Target", fontsize=8)

        status = "Captured" if capture else "Not captured"
        ax.text(0.02, 0.98, f"Status: {status}", transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.canvas.get_width_height()
        try:
            buf = canvas.tostring_rgb()
            image = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        except AttributeError:
            buf = canvas.tostring_argb()
            image = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
            image = image[:, :, [1, 2, 3]]
        plt.close(fig)
        return image
