import numpy as np
import torch
import torch.nn as nn


def _to_torch(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class _MLP(nn.Module):
    """
    功能:
        两层MLP骨干网络。
    输入:
        in_dim (int): 输入维度。
        hidden_dim (int): 隐藏层维度。
        out_dim (int): 输出维度。
    输出:
        torch.Tensor: 前向输出。
    """

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x):
        return self.net(x)


class OffPolicyDeterministicPolicy:
    """
    功能:
        MADDPG/MATD3通用策略容器（actor + critic，带target网络）。
    输入:
        args (argparse.Namespace): 扁平化参数。
        obs_space (gym.Space): actor观测空间。
        cent_obs_space (gym.Space): critic观测空间。
        act_space (gym.Space): 动作空间。
        device (torch.device): 设备。
    输出:
        无。
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.algorithm_name = str(args.algorithm_name)
        self.act_dim = int(act_space.shape[0])
        self.obs_dim = int(obs_space.shape[0])
        self.share_obs_dim = int(cent_obs_space.shape[0])

        self.actor_hidden_size = int(args.actor_hidden_size)
        self.critic_hidden_size = int(args.critic_hidden_size)
        self.lr = float(args.lr)
        self.critic_lr = float(args.critic_lr)
        self.opti_eps = float(args.opti_eps)
        self.weight_decay = float(args.weight_decay)

        self.action_noise_std = float(args.action_noise_std)
        self.target_policy_noise_std = float(args.target_policy_noise_std)
        self.target_noise_clip = float(args.target_noise_clip)

        self.actor = _MLP(self.obs_dim, self.actor_hidden_size, self.act_dim).to(self.device)
        self.target_actor = _MLP(self.obs_dim, self.actor_hidden_size, self.act_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic1 = _MLP(self.share_obs_dim + self.act_dim, self.critic_hidden_size, 1).to(self.device)
        self.target_critic1 = _MLP(self.share_obs_dim + self.act_dim, self.critic_hidden_size, 1).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())

        # 为了兼容现有runner的save/load接口，保留critic别名。
        self.critic = self.critic1

        self.use_twin_critic = self.algorithm_name == "matd3"
        if self.use_twin_critic:
            self.critic2 = _MLP(self.share_obs_dim + self.act_dim, self.critic_hidden_size, 1).to(self.device)
            self.target_critic2 = _MLP(self.share_obs_dim + self.act_dim, self.critic_hidden_size, 1).to(self.device)
            self.target_critic2.load_state_dict(self.critic2.state_dict())
        else:
            self.critic2 = None
            self.target_critic2 = None

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        if self.use_twin_critic:
            self.critic2_optimizer = torch.optim.Adam(
                self.critic2.parameters(),
                lr=self.critic_lr,
                eps=self.opti_eps,
                weight_decay=self.weight_decay,
            )
        else:
            self.critic2_optimizer = None

    def lr_decay(self, episode, episodes):
        """
        功能:
            off-policy后端暂不启用线性学习率衰减（占位兼容接口）。
        输入:
            episode (int): 当前轮次。
            episodes (int): 总轮次。
        输出:
            无。
        """
        return

    def _actor_action(self, obs, deterministic=False):
        obs_t = _to_torch(obs, self.device)
        act = torch.tanh(self.actor(obs_t))
        if deterministic:
            return act
        noise = torch.randn_like(act) * self.action_noise_std
        return torch.clamp(act + noise, -1.0, 1.0)

    def _critic1_forward(self, share_obs, action):
        x = torch.cat([share_obs, action], dim=-1)
        return self.critic1(x)

    def _critic2_forward(self, share_obs, action):
        x = torch.cat([share_obs, action], dim=-1)
        return self.critic2(x)

    @torch.no_grad()
    def get_actions(
        self,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False,
    ):
        """
        功能:
            采样动作并兼容MAPPO接口返回。
        输入:
            与RMAPPOPolicy.get_actions保持一致。
        输出:
            tuple: values/actions/action_log_probs/rnn_actor/rnn_critic。
        """
        action = self._actor_action(obs, deterministic=deterministic)
        share_obs_t = _to_torch(cent_obs, self.device)
        value = self._critic1_forward(share_obs_t, action)
        batch = int(action.shape[0])
        action_log_probs = torch.zeros((batch, 1), dtype=torch.float32, device=self.device)
        next_rnn_actor = _to_torch(rnn_states_actor, self.device)
        next_rnn_critic = _to_torch(rnn_states_critic, self.device)
        return value, action, action_log_probs, next_rnn_actor, next_rnn_critic

    @torch.no_grad()
    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        功能:
            返回critic对当前状态的占位值（用于兼容调用）。
        输入:
            cent_obs (np.ndarray): critic输入。
            rnn_states_critic: 未使用。
            masks: 未使用。
        输出:
            torch.Tensor: value张量。
        """
        share_obs_t = _to_torch(cent_obs, self.device)
        batch = int(share_obs_t.shape[0])
        zero_action = torch.zeros((batch, self.act_dim), dtype=torch.float32, device=self.device)
        return self._critic1_forward(share_obs_t, zero_action)

    @torch.no_grad()
    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        功能:
            评估/推理接口，返回动作与RNN状态占位。
        输入:
            obs (np.ndarray): actor输入观测。
            rnn_states_actor (np.ndarray): RNN状态占位。
            masks (np.ndarray): 掩码占位。
            deterministic (bool): 是否确定性动作。
        输出:
            tuple: actions, next_rnn_states_actor。
        """
        action = self._actor_action(obs, deterministic=deterministic)
        next_rnn_actor = _to_torch(rnn_states_actor, self.device)
        return action, next_rnn_actor

