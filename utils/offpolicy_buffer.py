import numpy as np


class OffPolicyReplayBuffer:
    """
    功能:
        简单环形回放池，存储连续控制off-policy训练所需转移样本。
    输入:
        capacity (int): 回放池容量。
        obs_dim (int): actor输入观测维度。
        share_obs_dim (int): critic输入观测维度。
        act_dim (int): 动作维度。
    输出:
        无。
    """

    def __init__(self, capacity, obs_dim, share_obs_dim, act_dim):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, int(obs_dim)), dtype=np.float32)
        self.share_obs = np.zeros((self.capacity, int(share_obs_dim)), dtype=np.float32)
        self.actions = np.zeros((self.capacity, int(act_dim)), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, int(obs_dim)), dtype=np.float32)
        self.next_share_obs = np.zeros((self.capacity, int(share_obs_dim)), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.active_masks = np.ones((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs,
        share_obs,
        actions,
        rewards,
        next_obs,
        next_share_obs,
        dones,
        active_masks,
    ):
        """
        功能:
            批量写入样本（通常对应并行环境的一步）。
        输入:
            各数组shape均为(batch, dim)。
        输出:
            无。
        """
        batch = int(obs.shape[0])
        for i in range(batch):
            idx = int(self.ptr)
            self.obs[idx] = obs[i]
            self.share_obs[idx] = share_obs[i]
            self.actions[idx] = actions[i]
            self.rewards[idx, 0] = float(rewards[i, 0])
            self.next_obs[idx] = next_obs[i]
            self.next_share_obs[idx] = next_share_obs[i]
            self.dones[idx, 0] = float(dones[i, 0])
            self.active_masks[idx, 0] = float(active_masks[i, 0])
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size):
        """
        功能:
            判断回放池是否满足采样条件。
        输入:
            batch_size (int): 采样批大小。
        输出:
            bool: 是否可采样。
        """
        return int(self.size) >= int(batch_size)

    def sample(self, batch_size):
        """
        功能:
            随机采样一批转移样本。
        输入:
            batch_size (int): 采样批大小。
        输出:
            dict: 采样字段字典。
        """
        idx = np.random.randint(0, int(self.size), size=int(batch_size))
        return {
            "obs": self.obs[idx],
            "share_obs": self.share_obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "next_share_obs": self.next_share_obs[idx],
            "dones": self.dones[idx],
            "active_masks": self.active_masks[idx],
        }

