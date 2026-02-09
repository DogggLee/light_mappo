import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from envs.uav_pursuit_env import MultiUavPursuitEnv
from runner.shared.env_runner import EnvRunner


def _t2n(x):
    return x.detach().cpu().numpy()


class UavPursuitRunner(EnvRunner):
    """Runner with GIF logging for the UAV pursuit MPE scenario."""

    def __init__(self, config):
        super().__init__(config)
        self.gif_interval = getattr(self.all_args, "gif_interval", 10)
        self.gif_frame_duration = getattr(self.all_args, "gif_frame_duration", 0.1)
        self.gif_dir = Path(self.run_dir) / "gifs"
        self.gif_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                self.insert(data)

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                elapsed = time.time() - start
                fps = int(total_num_steps / elapsed) if elapsed > 0 else 0
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        fps,
                    )
                )

                train_infos["average_episode_rewards"] = (
                    np.mean(self.buffer.rewards) * self.episode_length
                )
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)

            if (episode + 1) % self.gif_interval == 0:
                self._save_training_gif(episode + 1)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    @torch.no_grad()
    def _save_training_gif(self, episode_idx):
        frames = self._collect_episode_frames(episode_idx)
        gif_path = self.gif_dir / f"episode_{episode_idx:04d}.gif"
        imageio.mimsave(str(gif_path), frames, duration=self.gif_frame_duration)

    @torch.no_grad()
    def _collect_episode_frames(self, episode_idx):
        env = MultiUavPursuitEnv(
            num_hunters=self.all_args.num_hunters,
            num_blockers=self.all_args.num_blockers,
            world_size=self.all_args.world_size,
            dt=self.all_args.dt,
            capture_radius=self.all_args.capture_radius,
            capture_steps=self.all_args.capture_steps,
            max_steps=self.episode_length,
            seed=self.all_args.seed,
        )
        obs = env.reset()

        role_groups = env.role_groups
        hunter_indices = role_groups.get("hunter", [])
        target_index = role_groups.get("target", [None])[0]

        positions = {idx: [env.positions[idx].copy()] for idx in range(env.agent_num)}
        capture = False

        rnn_states = np.zeros((env.agent_num, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((env.agent_num, 1), dtype=np.float32)

        frames = [
            self._draw_frame(
                positions,
                env.world_size,
                hunter_indices,
                target_index,
                capture,
                episode_idx,
                step=0,
            )
        ]

        for step in range(1, self.episode_length + 1):
            self.trainer.prep_rollout()
            actions, rnn_states = self.trainer.policy.act(
                obs, rnn_states, masks, deterministic=True
            )
            actions = _t2n(actions)
            rnn_states = _t2n(rnn_states)

            if env.action_space[0].__class__.__name__ == "MultiDiscrete":
                actions_env = []
                for i in range(env.action_space[0].shape):
                    actions_env.append(np.eye(env.action_space[0].high[i] + 1)[actions[:, i]])
                actions_env = np.concatenate(actions_env, axis=1)
            elif env.action_space[0].__class__.__name__ == "Discrete":
                actions_env = np.squeeze(np.eye(env.action_space[0].n)[actions], 1)
            else:
                actions_env = actions

            obs, rewards, dones, infos = env.step(actions_env)
            capture = capture or any(info.get("capture", False) for info in infos)

            for idx in range(env.agent_num):
                positions[idx].append(env.positions[idx].copy())

            frames.append(
                self._draw_frame(
                    positions,
                    env.world_size,
                    hunter_indices,
                    target_index,
                    capture,
                    episode_idx,
                    step=step,
                )
            )

            if np.any(dones):
                rnn_states[dones] = 0.0
                masks[dones] = 0.0
            if np.all(dones):
                break

        env.close()
        return frames

    def _draw_frame(
        self,
        positions,
        world_size,
        hunter_indices,
        target_index,
        capture,
        episode_idx,
        step,
    ):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
        ax.set_xlim(-world_size, world_size)
        ax.set_ylim(-world_size, world_size)
        ax.set_aspect("equal")
        ax.set_title(f"Episode {episode_idx} - Step {step}")
        ax.grid(True, linestyle="--", alpha=0.3)

        hunter_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
        for i, idx in enumerate(hunter_indices):
            traj = np.array(positions[idx])
            color = hunter_colors[i % len(hunter_colors)]
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker="o", s=35)
            ax.text(traj[-1, 0], traj[-1, 1], f"H{i+1}", fontsize=8)

        if target_index is not None:
            target_traj = np.array(positions[target_index])
            ax.plot(target_traj[:, 0], target_traj[:, 1], color="#d62728", linewidth=2.0)
            ax.scatter(target_traj[-1, 0], target_traj[-1, 1], color="#d62728", marker="*", s=80)
            ax.text(target_traj[-1, 0], target_traj[-1, 1], "Target", fontsize=8)

        status = "Captured" if capture else "Not captured"
        ax.text(
            0.02,
            0.98,
            f"Status: {status}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.canvas.get_width_height()
        try:
            buf = canvas.tostring_rgb()
            image = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        except AttributeError:
            # Matplotlib versions without tostring_rgb; use ARGB and convert.
            buf = canvas.tostring_argb()
            image = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
            image = image[:, :, [1, 2, 3]]  # ARGB -> RGB
        plt.close(fig)
        return image
