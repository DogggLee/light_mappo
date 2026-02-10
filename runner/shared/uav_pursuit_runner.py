import time
from pathlib import Path

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
        self.target_patrol_names = self._normalize_name_list(getattr(self.all_args, "target_patrol_names", None))
        self.target_patrol_switch_interval = int(getattr(self.all_args, "target_patrol_switch_interval", 0))
        self.eval_random_patrol_routes = int(getattr(self.all_args, "eval_random_patrol_routes", 0))
        self.eval_random_patrol_points = int(getattr(self.all_args, "eval_random_patrol_points", 4))
        self._patrol_rng = np.random.RandomState(self.all_args.seed + 2027)
        self._current_patrol_name = None

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
                elapsed = time.time() - start
                fps = int(total_num_steps / elapsed) if elapsed > 0 else 0
                print(
                    f"\n Scenario {self.all_args.scenario_name} Algo {self.algorithm_name} Exp {self.experiment_name} updates {episode}/{episodes} episodes, total num timesteps {total_num_steps}/{self.num_env_steps}, FPS {fps}.\n"
                )
                avg_rewards = []
                for group_name in self.group_order:
                    avg_rewards.append(np.mean(self.buffers[group_name].rewards) * self.episode_length)
                train_infos["system"] = {"average_episode_rewards": float(np.mean(avg_rewards))}
                self.log_train(train_infos, total_num_steps)
                self.record_train_metrics(total_num_steps, train_infos["system"]["average_episode_rewards"])

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

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episodes = int(self.all_args.eval_episodes)
        eval_episode_rewards = []
        capture_flags = []
        capture_steps = []
        random_routes = None

        if self.target_policy_source == "patrol" and self.eval_random_patrol_routes > 0:
            random_routes = self._generate_random_patrol_routes(self.eval_random_patrol_routes, self.eval_random_patrol_points)

        for ep in range(eval_episodes):
            if self.target_policy_source == "patrol":
                if random_routes:
                    route = random_routes[self._patrol_rng.randint(0, len(random_routes))]
                    self.eval_envs.set_target_patrol_waypoints(route, route_name=f"eval_random_{ep}")
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

            episode_done = np.zeros(self.n_eval_rollout_threads, dtype=bool)
            episode_steps = np.zeros(self.n_eval_rollout_threads, dtype=np.int32)
            episode_capture = np.zeros(self.n_eval_rollout_threads, dtype=bool)
            episode_capture_step = np.full(self.n_eval_rollout_threads, -1, dtype=np.int32)
            ep_rewards = np.zeros((self.n_eval_rollout_threads, self.num_agents), dtype=np.float32)

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

                if np.all(episode_done):
                    break

            eval_episode_rewards.append(ep_rewards)
            capture_flags.extend(episode_capture.tolist())
            capture_steps.extend([int(cs) for cs in episode_capture_step if cs > 0])

        eval_avg_rewards = float(np.mean([np.mean(ep) for ep in eval_episode_rewards])) if eval_episode_rewards else 0.0
        total_episodes = max(1, eval_episodes * self.n_eval_rollout_threads)
        capture_success_rate = float(np.sum(capture_flags) / total_episodes) if capture_flags else 0.0
        avg_capture_steps = float(np.mean(capture_steps)) if capture_steps else None

        eval_env_infos = {"eval_average_episode_rewards": eval_avg_rewards}
        self.log_env(eval_env_infos, total_num_steps)
        self.record_eval_metrics(total_num_steps, eval_avg_rewards, capture_success_rate, avg_capture_steps)

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
