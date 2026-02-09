import numpy as np


class UAVEncirclementEnv:
    def __init__(self, config=None):
        self.config = config
        self.num_hunters = int(getattr(config, "num_hunters", 2))
        self.num_blockers = int(getattr(config, "num_blockers", 1))
        self.num_targets = int(getattr(config, "num_targets", 1))
        self.min_hunters = self._resolve_bound("min_hunters", self.num_hunters)
        self.min_blockers = self._resolve_bound("min_blockers", self.num_blockers)
        self.min_targets = self._resolve_bound("min_targets", self.num_targets)
        self.max_hunters = self._resolve_bound("max_hunters", self.num_hunters)
        self.max_blockers = self._resolve_bound("max_blockers", self.num_blockers)
        self.max_targets = self._resolve_bound("max_targets", self.num_targets)
        self.use_presence_mask = bool(getattr(config, "use_presence_mask", True))
        self.agent_num = self.max_hunters + self.max_blockers + self.max_targets
        self.action_dim = 2
        self.world_size = float(getattr(config, "world_size", 10.0))
        self.dt = float(getattr(config, "dt", 0.1))
        self.max_episode_steps = int(getattr(config, "max_episode_steps", 300))
        self.max_accel = float(getattr(config, "max_accel", 2.0))
        self.hunter_max_speed = float(getattr(config, "hunter_max_speed", 2.5))
        self.blocker_max_speed = float(getattr(config, "blocker_max_speed", 1.8))
        self.target_max_speed = float(getattr(config, "target_max_speed", 2.0))
        self.hunter_perception = float(getattr(config, "hunter_perception", 8.0))
        self.blocker_perception = float(getattr(config, "blocker_perception", 10.0))
        self.target_perception = float(getattr(config, "target_perception", 6.0))
        self.neighbor_mode = getattr(config, "neighbor_mode", "knn")
        self.neighbor_k = int(getattr(config, "neighbor_k", 3))
        self.capture_distance = float(getattr(config, "capture_distance", 0.6))
        self.capture_hold_steps = int(getattr(config, "capture_hold_steps", 5))
        self.obstacle_count = int(getattr(config, "obstacle_count", 0))
        self.obstacle_radius = float(getattr(config, "obstacle_radius", 0.8))
        self.agent_radius = 0.2
        self._set_role_indices(
            self.num_hunters, self.num_blockers, self.num_targets
        )
        self.obs_dim = self._compute_obs_dim()
        self.step_count = 0
        self.capture_count = 0
        self.positions = None
        self.velocities = None
        self.obstacles = []

    def _set_role_indices(self, num_hunters, num_blockers, num_targets):
        self.num_hunters = int(num_hunters)
        self.num_blockers = int(num_blockers)
        self.num_targets = int(num_targets)
        active_total = self.num_hunters + self.num_blockers + self.num_targets
        self.active_count = active_total
        self.active_mask = np.zeros(self.agent_num, dtype=np.float32)
        if active_total > 0:
            self.active_mask[:active_total] = 1.0
        self.hunter_indices = list(range(0, self.num_hunters))
        self.blocker_indices = list(
            range(self.num_hunters, self.num_hunters + self.num_blockers)
        )
        self.target_indices = list(
            range(self.num_hunters + self.num_blockers, active_total)
        )
        self.roles = ["inactive"] * self.agent_num
        for idx in self.hunter_indices:
            self.roles[idx] = "hunter"
        for idx in self.blocker_indices:
            self.roles[idx] = "blocker"
        for idx in self.target_indices:
            self.roles[idx] = "target"

    def _compute_obs_dim(self):
        self_state = 7
        target_info = 5
        neighbor_block = 8 + (1 if self.use_presence_mask else 0)
        if self.neighbor_mode == "knn":
            neighbor_dim = neighbor_block * self.neighbor_k
        else:
            neighbor_dim = neighbor_block
        target_dim = target_info + (1 if self.use_presence_mask else 0)
        return self_state + target_dim + neighbor_dim

    def reset(self):
        self.step_count = 0
        self.capture_count = 0
        self.positions = self._sample_positions()
        self.velocities = np.zeros((self.agent_num, 2), dtype=np.float32)
        self.obstacles = self._sample_obstacles()
        return [self._get_obs(i) for i in range(self.agent_num)]

    def regen(self, num_hunters=None, num_blockers=None, num_targets=None):
        if num_hunters is None:
            num_hunters = self._random_count(self.min_hunters, self.max_hunters)
        if num_blockers is None:
            num_blockers = self._random_count(self.min_blockers, self.max_blockers)
        if num_targets is None:
            num_targets = self._random_count(self.min_targets, self.max_targets)
        self._set_role_indices(num_hunters, num_blockers, num_targets)
        return self.reset()

    def step(self, actions):
        self.step_count += 1
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.agent_num, self.action_dim):
            actions = actions.reshape(self.agent_num, self.action_dim)
        actions = self._clip_actions(actions)
        actions = actions * self.active_mask[:, None]
        self._apply_dynamics(actions)
        self._apply_boundaries()
        self._apply_obstacles()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        obs = [self._get_obs(i) for i in range(self.agent_num)]
        infos = [{"capture": self.capture_count >= self.capture_hold_steps} for _ in range(self.agent_num)]
        reward_list = [[float(r)] for r in rewards]
        done_list = [bool(dones) for _ in range(self.agent_num)]
        return [obs, reward_list, done_list, infos]

    def _sample_positions(self):
        positions = np.random.uniform(
            low=-self.world_size * 0.6,
            high=self.world_size * 0.6,
            size=(self.agent_num, 2),
        ).astype(np.float32)
        if self.active_count < self.agent_num:
            positions[self.active_count :] = 0.0
        return positions

    def _sample_obstacles(self):
        obstacles = []
        for _ in range(self.obstacle_count):
            pos = np.random.uniform(
                low=-self.world_size * 0.5,
                high=self.world_size * 0.5,
                size=(2,),
            )
            obstacles.append((pos.astype(np.float32), self.obstacle_radius))
        return obstacles

    def _clip_actions(self, actions):
        norms = np.linalg.norm(actions, axis=1, keepdims=True) + 1e-6
        scale = np.minimum(1.0, self.max_accel / norms)
        return actions * scale

    def _apply_dynamics(self, actions):
        self.velocities += actions * self.dt
        for idx, role in enumerate(self.roles):
            if role == "inactive":
                self.velocities[idx] = 0.0
                continue
            max_speed = self._role_max_speed(role)
            speed = np.linalg.norm(self.velocities[idx])
            if speed > max_speed:
                self.velocities[idx] = self.velocities[idx] / speed * max_speed
        self.positions += self.velocities * self.dt
        if self.active_count < self.agent_num:
            self.positions[self.active_count :] = 0.0

    def _apply_boundaries(self):
        for idx in range(self.agent_num):
            if self.roles[idx] == "inactive":
                continue
            for dim in range(2):
                if self.positions[idx, dim] < -self.world_size:
                    self.positions[idx, dim] = -self.world_size
                    self.velocities[idx, dim] *= -0.5
                if self.positions[idx, dim] > self.world_size:
                    self.positions[idx, dim] = self.world_size
                    self.velocities[idx, dim] *= -0.5

    def _apply_obstacles(self):
        if not self.obstacles:
            return
        for idx in range(self.agent_num):
            if self.roles[idx] == "inactive":
                continue
            for center, radius in self.obstacles:
                offset = self.positions[idx] - center
                dist = np.linalg.norm(offset)
                min_dist = radius + self.agent_radius
                if dist < min_dist and dist > 1e-6:
                    push = offset / dist
                    self.positions[idx] = center + push * min_dist
                    self.velocities[idx] *= -0.3

    def _compute_rewards(self):
        rewards = np.zeros(self.agent_num, dtype=np.float32)
        target_pos = self.positions[self.target_indices] if self.target_indices else np.empty((0, 2))
        hunter_pos = self.positions[self.hunter_indices] if self.hunter_indices else np.empty((0, 2))
        blocker_pos = self.positions[self.blocker_indices] if self.blocker_indices else np.empty((0, 2))
        boundary_penalty = self._boundary_penalty()
        obstacle_penalty = self._obstacle_penalty()
        for idx, role in enumerate(self.roles):
            if role == "inactive":
                continue
            if role == "target":
                rewards[idx] += self._target_reward(idx, hunter_pos, blocker_pos)
            elif role == "hunter":
                rewards[idx] += self._hunter_reward(idx, target_pos)
            else:
                rewards[idx] += self._blocker_reward(idx, target_pos, hunter_pos)
            rewards[idx] += boundary_penalty[idx] + obstacle_penalty[idx]
        return rewards

    def _target_reward(self, idx, hunter_pos, blocker_pos):
        reward = 0.0
        if hunter_pos.size > 0:
            dist = np.linalg.norm(hunter_pos - self.positions[idx], axis=1)
            reward += np.mean(dist)
        if blocker_pos.size > 0:
            dist = np.linalg.norm(blocker_pos - self.positions[idx], axis=1)
            reward += 0.5 * np.mean(dist)
        return reward

    def _hunter_reward(self, idx, target_pos):
        reward = 0.0
        if target_pos.size > 0:
            dist = np.linalg.norm(target_pos - self.positions[idx], axis=1)
            reward -= np.min(dist)
            if np.min(dist) <= self.capture_distance:
                reward += 1.0
        return reward

    def _blocker_reward(self, idx, target_pos, hunter_pos):
        reward = 0.0
        if target_pos.size == 0 or hunter_pos.size == 0:
            return reward
        dist_to_target = np.linalg.norm(target_pos - self.positions[idx], axis=1)
        reward -= np.min(dist_to_target)
        nearest_target = target_pos[np.argmin(dist_to_target)]
        dist_to_hunter = np.linalg.norm(hunter_pos - nearest_target, axis=1)
        nearest_hunter = hunter_pos[np.argmin(dist_to_hunter)]
        line_vec = nearest_hunter - nearest_target
        line_norm = np.linalg.norm(line_vec)
        if line_norm > 1e-6:
            proj = np.dot(self.positions[idx] - nearest_target, line_vec) / line_norm
            closest = nearest_target + line_vec / line_norm * np.clip(proj, 0.0, line_norm)
            line_dist = np.linalg.norm(self.positions[idx] - closest)
            reward += max(0.0, 1.0 - line_dist / (self.world_size + 1e-6))
        return reward

    def _boundary_penalty(self):
        penalty = np.zeros(self.agent_num, dtype=np.float32)
        margin = self.world_size * 0.95
        for idx in range(self.agent_num):
            if self.roles[idx] == "inactive":
                continue
            excess = np.maximum(0.0, np.abs(self.positions[idx]) - margin)
            penalty[idx] -= np.sum(excess) * 0.5
        return penalty

    def _obstacle_penalty(self):
        penalty = np.zeros(self.agent_num, dtype=np.float32)
        for idx in range(self.agent_num):
            if self.roles[idx] == "inactive":
                continue
            for center, radius in self.obstacles:
                dist = np.linalg.norm(self.positions[idx] - center)
                if dist < radius + self.agent_radius:
                    penalty[idx] -= 1.0
        return penalty

    def _compute_dones(self):
        captured = False
        if self.target_indices and self.hunter_indices:
            target_pos = self.positions[self.target_indices]
            hunter_pos = self.positions[self.hunter_indices]
            dist = np.linalg.norm(
                hunter_pos[:, None, :] - target_pos[None, :, :], axis=-1
            )
            if np.min(dist) <= self.capture_distance:
                self.capture_count += 1
            else:
                self.capture_count = 0
            captured = self.capture_count >= self.capture_hold_steps
        timeout = self.step_count >= self.max_episode_steps
        return captured or timeout

    def _get_obs(self, idx):
        role = self.roles[idx]
        if role == "inactive":
            return np.zeros(self.obs_dim, dtype=np.float32)
        role_one_hot = self._role_one_hot(role)
        self_state = np.concatenate([self.positions[idx], self.velocities[idx], role_one_hot])
        target_block = self._get_target_block(idx, role)
        neighbor_block = self._get_neighbor_block(idx, role)
        return np.concatenate([self_state, target_block, neighbor_block]).astype(np.float32)

    def _get_target_block(self, idx, role):
        perception = self._role_perception(role)
        if role == "target":
            if not self.hunter_indices:
                return self._pad_target_block()
            hunter_pos = self.positions[self.hunter_indices]
            hunter_vel = self.velocities[self.hunter_indices]
            rel = hunter_pos - self.positions[idx]
            dist = np.linalg.norm(rel, axis=1)
            nearest = np.argmin(dist)
            if dist[nearest] > perception:
                return self._pad_target_block()
            return self._format_target_block(
                rel[nearest], hunter_vel[nearest] - self.velocities[idx], dist[nearest]
            )
        if not self.target_indices:
            return self._pad_target_block()
        target_pos = self.positions[self.target_indices]
        target_vel = self.velocities[self.target_indices]
        rel = target_pos - self.positions[idx]
        dist = np.linalg.norm(rel, axis=1)
        nearest = np.argmin(dist)
        if dist[nearest] > perception:
            return self._pad_target_block()
        return self._format_target_block(
            rel[nearest], target_vel[nearest] - self.velocities[idx], dist[nearest]
        )

    def _get_neighbor_block(self, idx, role):
        perception = self._role_perception(role)
        neighbor_features = []
        for j in range(self.agent_num):
            if j == idx:
                continue
            if self.roles[j] == "inactive":
                continue
            rel = self.positions[j] - self.positions[idx]
            dist = np.linalg.norm(rel)
            if dist > perception:
                continue
            feature = self._format_neighbor_block(
                rel, self.velocities[j] - self.velocities[idx], self.roles[j], dist
            )
            neighbor_features.append((dist, feature))
        neighbor_features.sort(key=lambda x: x[0])
        if self.neighbor_mode == "knn":
            features = [item[1] for item in neighbor_features[: self.neighbor_k]]
            while len(features) < self.neighbor_k:
                features.append(self._pad_neighbor_block())
            return (
                np.concatenate(features)
                if features
                else np.zeros(len(self._pad_neighbor_block()) * self.neighbor_k, dtype=np.float32)
            )
        if not neighbor_features:
            return self._pad_neighbor_block()
        stacked = np.stack([item[1] for item in neighbor_features], axis=0)
        mean_feat = np.mean(stacked, axis=0)
        if self.use_presence_mask:
            mean_feat[-1] = min(1.0, float(len(neighbor_features)) / float(self.neighbor_k))
        return mean_feat

    def _role_one_hot(self, role):
        if role == "hunter":
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if role == "blocker":
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if role == "inactive":
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _role_max_speed(self, role):
        if role == "hunter":
            return self.hunter_max_speed
        if role == "blocker":
            return self.blocker_max_speed
        return self.target_max_speed

    def _role_perception(self, role):
        if role == "hunter":
            return self.hunter_perception
        if role == "blocker":
            return self.blocker_perception
        if role == "inactive":
            return 0.0
        return self.target_perception

    def _resolve_bound(self, name, default):
        value = getattr(self.config, name, None)
        if value is None:
            return int(default)
        return int(value)

    def _random_count(self, min_value, max_value):
        min_value = int(min_value)
        max_value = int(max_value)
        if max_value < min_value:
            max_value = min_value
        return np.random.randint(min_value, max_value + 1)

    def _format_target_block(self, rel, vel_rel, dist):
        if self.use_presence_mask:
            return np.concatenate([rel, vel_rel, [dist, 1.0]])
        return np.concatenate([rel, vel_rel, [dist]])

    def _pad_target_block(self):
        if self.use_presence_mask:
            return np.zeros(6, dtype=np.float32)
        return np.zeros(5, dtype=np.float32)

    def _format_neighbor_block(self, rel, vel_rel, role, dist):
        base = np.concatenate([rel, vel_rel, self._role_one_hot(role), [dist]])
        if self.use_presence_mask:
            return np.concatenate([base, [1.0]])
        return base

    def _pad_neighbor_block(self):
        length = 9 if self.use_presence_mask else 8
        return np.zeros(length, dtype=np.float32)
