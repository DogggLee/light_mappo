import torch

from algorithms.algorithm.offpolicy_policy import _to_torch


class OffPolicyDeterministicTrainer:
    """
    功能:
        MADDPG/MATD3通用训练器（基于确定性策略梯度）。
    输入:
        args (argparse.Namespace): 扁平化参数。
        policy (OffPolicyDeterministicPolicy): 策略对象。
        device (torch.device): 训练设备。
    输出:
        无。
    """

    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.policy = policy
        self.algorithm_name = str(args.algorithm_name)
        self.gamma = float(args.gamma)
        self.tau = float(args.tau)
        self.batch_size = int(args.batch_size)
        self.warmup_steps = int(args.warmup_steps)
        self.policy_delay = int(args.policy_delay)
        self.train_updates_per_episode = int(args.train_updates_per_episode)
        self.target_policy_noise_std = float(args.target_policy_noise_std)
        self.target_noise_clip = float(args.target_noise_clip)
        self._update_step = 0
        self.value_normalizer = None

    def prep_training(self):
        """
        功能:
            切换网络到训练模式。
        输入:
            无。
        输出:
            无。
        """
        self.policy.actor.train()
        self.policy.critic1.train()
        if self.policy.use_twin_critic:
            self.policy.critic2.train()

    def prep_rollout(self):
        """
        功能:
            切换网络到推理模式。
        输入:
            无。
        输出:
            无。
        """
        self.policy.actor.eval()
        self.policy.critic1.eval()
        if self.policy.use_twin_critic:
            self.policy.critic2.eval()

    def _soft_update(self, src, dst):
        for p_src, p_dst in zip(src.parameters(), dst.parameters()):
            p_dst.data.copy_(self.tau * p_src.data + (1.0 - self.tau) * p_dst.data)

    def _train_one_step(self, replay_buffer):
        batch = replay_buffer.sample(self.batch_size)
        obs = _to_torch(batch["obs"], self.device)
        share_obs = _to_torch(batch["share_obs"], self.device)
        actions = _to_torch(batch["actions"], self.device)
        rewards = _to_torch(batch["rewards"], self.device)
        next_obs = _to_torch(batch["next_obs"], self.device)
        next_share_obs = _to_torch(batch["next_share_obs"], self.device)
        dones = _to_torch(batch["dones"], self.device)
        active_masks = _to_torch(batch["active_masks"], self.device)

        with torch.no_grad():
            target_action = torch.tanh(self.policy.target_actor(next_obs))
            if self.policy.use_twin_critic:
                noise = torch.randn_like(target_action) * self.target_policy_noise_std
                noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
                target_action = torch.clamp(target_action + noise, -1.0, 1.0)
                q1_t = self.policy.target_critic1(torch.cat([next_share_obs, target_action], dim=-1))
                q2_t = self.policy.target_critic2(torch.cat([next_share_obs, target_action], dim=-1))
                q_target = torch.min(q1_t, q2_t)
            else:
                q_target = self.policy.target_critic1(torch.cat([next_share_obs, target_action], dim=-1))
            y = rewards + self.gamma * (1.0 - dones) * q_target

        q1 = self.policy.critic1(torch.cat([share_obs, actions], dim=-1))
        critic1_loss = ((q1 - y) ** 2 * active_masks).sum() / (active_masks.sum() + 1e-6)
        self.policy.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.policy.critic1_optimizer.step()

        critic2_loss_val = 0.0
        if self.policy.use_twin_critic:
            q2 = self.policy.critic2(torch.cat([share_obs, actions], dim=-1))
            critic2_loss = ((q2 - y) ** 2 * active_masks).sum() / (active_masks.sum() + 1e-6)
            self.policy.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.policy.critic2_optimizer.step()
            critic2_loss_val = float(critic2_loss.item())

        actor_loss_val = 0.0
        if (self._update_step % max(1, self.policy_delay)) == 0:
            act = torch.tanh(self.policy.actor(obs))
            actor_loss = -self.policy.critic1(torch.cat([share_obs, act], dim=-1))
            actor_loss = (actor_loss * active_masks).sum() / (active_masks.sum() + 1e-6)
            self.policy.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.policy.actor_optimizer.step()
            actor_loss_val = float(actor_loss.item())

            self._soft_update(self.policy.actor, self.policy.target_actor)
            self._soft_update(self.policy.critic1, self.policy.target_critic1)
            if self.policy.use_twin_critic:
                self._soft_update(self.policy.critic2, self.policy.target_critic2)

        self._update_step += 1
        return {
            "critic_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss_val),
            "actor_loss": float(actor_loss_val),
            "q_mean": float(q1.mean().item()),
            "reward_mean": float(rewards.mean().item()),
        }

    def train(self, replay_buffer):
        """
        功能:
            使用回放池执行多步梯度更新。
        输入:
            replay_buffer (OffPolicyReplayBuffer): 回放池。
        输出:
            dict: 训练统计。
        """
        if int(replay_buffer.size) < int(self.warmup_steps):
            return {
                "replay_size": int(replay_buffer.size),
                "critic_loss": 0.0,
                "critic2_loss": 0.0,
                "actor_loss": 0.0,
                "q_mean": 0.0,
                "reward_mean": 0.0,
                "average_episode_rewards": 0.0,
            }
        if not replay_buffer.can_sample(self.batch_size):
            return {
                "replay_size": int(replay_buffer.size),
                "critic_loss": 0.0,
                "critic2_loss": 0.0,
                "actor_loss": 0.0,
                "q_mean": 0.0,
                "reward_mean": 0.0,
                "average_episode_rewards": 0.0,
            }

        logs = []
        for _ in range(max(1, int(self.train_updates_per_episode))):
            logs.append(self._train_one_step(replay_buffer))

        out = {"replay_size": int(replay_buffer.size)}
        for key in ["critic_loss", "critic2_loss", "actor_loss", "q_mean", "reward_mean"]:
            out[key] = float(sum(x[key] for x in logs) / max(1, len(logs)))
        out["average_episode_rewards"] = out["reward_mean"]
        return out
