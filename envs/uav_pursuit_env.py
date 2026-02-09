import numpy as np
import gym
from gym import spaces


class MultiUavPursuitEnv:
    """
    Multi-UAV pursuit environment with three roles:
    - Hunter: fast pursuer that can capture the target.
    - Blocker: medium-speed pursuer with larger perception range.
    - Target: single evasive UAV.
    """

    def __init__(
        self,
        num_hunters=1,
        num_blockers=0,
        world_size=1.0,
        dt=0.1,
        capture_radius=0.12,
        capture_steps=5,
        max_steps=300,
        seed=None,
    ):
        if num_hunters < 1:
            raise ValueError("num_hunters must be >= 1")
        if num_blockers < 0:
            raise ValueError("num_blockers must be >= 0")

        self.num_hunters = num_hunters
        self.num_blockers = num_blockers
        self.num_targets = 1
        self.agent_num = self.num_hunters + self.num_blockers + self.num_targets

        self.world_size = world_size
        self.dt = dt
        self.capture_radius = capture_radius
        self.capture_steps = capture_steps
        self.max_steps = max_steps
        self.np_random = np.random.RandomState(seed)

        self.role_names = (
            ["hunter"] * self.num_hunters
            + ["blocker"] * self.num_blockers
            + ["target"]
        )
        self.role_groups = {
            "hunter": list(range(self.num_hunters)),
            "blocker": list(range(self.num_hunters, self.num_hunters + self.num_blockers)),
            "target": [self.agent_num - 1],
        }

        # role-specific parameters
        self.max_speeds = {
            "hunter": 1.2,
            "blocker": 0.9,
            "target": 1.0,
        }
        self.perception_ranges = {
            "hunter": 0.8,
            "blocker": 1.2,
            "target": 0.8,
        }

        # state: positions and velocities for each agent
        self.positions = np.zeros((self.agent_num, 2), dtype=np.float32)
        self.velocities = np.zeros((self.agent_num, 2), dtype=np.float32)
        self._capture_counter = 0
        self._step_count = 0

        # action/observation spaces
        self.action_dim = 2
        self.action_space = [
            spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
            for _ in range(self.agent_num)
        ]

        self.obs_dim = self._calc_obs_dim()
        self.observation_space = [
            spaces.Box(
                low=-np.inf,
                high=+np.inf,
                shape=(self.obs_dim,),
                dtype=np.float32,
            )
            for _ in range(self.agent_num)
        ]
        self.share_observation_space = [
            spaces.Box(
                low=-np.inf,
                high=+np.inf,
                shape=(self.obs_dim * self.agent_num,),
                dtype=np.float32,
            )
            for _ in range(self.agent_num)
        ]

    def _calc_obs_dim(self):
        # own pos/vel (4) + per other agent: relative position (2) + distance (1)
        return 4 + (self.agent_num - 1) * 3

    def reset(self):
        self.positions = self.np_random.uniform(
            low=-self.world_size, high=self.world_size, size=(self.agent_num, 2)
        ).astype(np.float32)
        self.velocities = np.zeros((self.agent_num, 2), dtype=np.float32)
        self._capture_counter = 0
        self._step_count = 0
        return self._get_obs()

    def step(self, actions):
        self._step_count += 1
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.agent_num, self.action_dim):
            raise ValueError(
                f"Expected actions shape ({self.agent_num}, {self.action_dim}), got {actions.shape}"
            )

        # update velocities/positions
        for idx in range(self.agent_num):
            role = self.role_names[idx]
            max_speed = self.max_speeds[role]
            action = np.clip(actions[idx], -1.0, 1.0)
            velocity = action * max_speed
            self.velocities[idx] = velocity
            self.positions[idx] += velocity * self.dt
            self.positions[idx] = np.clip(
                self.positions[idx], -self.world_size, self.world_size
            )

        obs = self._get_obs()
        rewards, capture = self._get_rewards()
        dones = np.array([capture or self._step_count >= self.max_steps] * self.agent_num)
        infos = self._get_infos(capture)
        return obs, rewards, dones, infos

    def _get_obs(self):
        obs = []
        for idx in range(self.agent_num):
            role = self.role_names[idx]
            perception = self.perception_ranges[role]

            own_pos = self.positions[idx]
            own_vel = self.velocities[idx]

            other_features = []
            for jdx in range(self.agent_num):
                if jdx == idx:
                    continue
                rel = self.positions[jdx] - own_pos
                dist = np.linalg.norm(rel)
                if dist <= perception:
                    other_features.extend([rel[0], rel[1], dist])
                else:
                    other_features.extend([0.0, 0.0, 0.0])

            obs_vec = np.concatenate([own_pos, own_vel, np.array(other_features, dtype=np.float32)])
            obs.append(obs_vec.astype(np.float32))
        return np.stack(obs, axis=0)

    def _get_rewards(self):
        target_idx = self.agent_num - 1
        target_pos = self.positions[target_idx]

        hunter_indices = self.role_groups["hunter"]
        distances_to_target = [
            np.linalg.norm(self.positions[idx] - target_pos) for idx in hunter_indices
        ]
        min_distance = float(np.min(distances_to_target))

        # capture logic
        if min_distance <= self.capture_radius:
            self._capture_counter += 1
        else:
            self._capture_counter = 0

        capture = self._capture_counter >= self.capture_steps

        rewards = np.zeros(self.agent_num, dtype=np.float32)

        # rewards for hunters and blockers
        for idx in range(self.agent_num - 1):
            role = self.role_names[idx]
            dist = np.linalg.norm(self.positions[idx] - target_pos)
            if role == "hunter":
                rewards[idx] = -dist
                if capture:
                    rewards[idx] += 10.0
            else:
                rewards[idx] = -0.7 * dist
                if capture:
                    rewards[idx] += 6.0

        # reward for target
        rewards[target_idx] = min_distance
        if capture:
            rewards[target_idx] -= 12.0

        return rewards, capture

    def _get_infos(self, capture):
        infos = []
        for idx in range(self.agent_num):
            infos.append(
                {
                    "role": self.role_names[idx],
                    "capture": capture,
                    "role_groups": self.role_groups,
                }
            )
        return infos

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return seed
