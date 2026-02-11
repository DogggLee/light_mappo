import json
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiUavPursuitEnv:
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
        target_policy_source="train",
        target_patrol_path=None,
        max_speed_hunter=1.2,
        max_speed_blocker=0.9,
        max_speed_target=1.0,
        perception_hunter=0.8,
        perception_blocker=1.2,
        perception_target=0.8,
        speed_penalty=0.00,
        target_patrol_names=None,
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

        self.target_policy_source = target_policy_source
        self.target_patrol_names = self._normalize_name_list(target_patrol_names)
        self.patrol_routes, self.target_patrol_name, self.target_patrol_waypoints = self._load_patrol_routes(
            target_patrol_path,
            self.target_patrol_names,
        )
        self._target_patrol_idx = 0

        self.role_names = ["hunter"] * self.num_hunters + ["blocker"] * self.num_blockers + ["target"]
        self.role_groups = {
            "hunter": list(range(self.num_hunters)),
            "blocker": list(range(self.num_hunters, self.num_hunters + self.num_blockers)),
            "target": [self.agent_num - 1],
        }

        self.max_speeds = {
            "hunter": float(max_speed_hunter),
            "blocker": float(max_speed_blocker),
            "target": float(max_speed_target),
        }
        self.perception_ranges = {
            "hunter": float(perception_hunter),
            "blocker": float(perception_blocker),
            "target": float(perception_target),
        }
        self.speed_penalty = float(speed_penalty)

        self.positions = np.zeros((self.agent_num, 2), dtype=np.float32)
        self.velocities = np.zeros((self.agent_num, 2), dtype=np.float32)
        self._capture_counter = 0
        self._step_count = 0

        self.action_dim = 2
        self.action_space = [spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32) for _ in range(self.agent_num)]

        self.obs_dim = self._calc_obs_dim()
        self.observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32) for _ in range(self.agent_num)]
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim * self.agent_num,), dtype=np.float32)
            for _ in range(self.agent_num)
        ]

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

    def _convert_waypoints(self, waypoints):
        arr = np.asarray(waypoints, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
            raise ValueError("巡逻路径至少需要2个二维路点")
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if 0.0 <= min_val and max_val <= 1.0:
            arr = (arr * 2.0 - 1.0) * self.world_size
        elif -1.0 <= min_val and max_val <= 1.0:
            arr = arr * self.world_size
        return arr

    def _load_patrol_routes(self, target_patrol_path, preferred_names):
        if self.target_policy_source != "patrol":
            return {}, None, None
        if target_patrol_path is None:
            raise ValueError("target_policy_source=patrol 时必须提供 target_patrol_path")
        route_file = Path(target_patrol_path)
        data = json.loads(route_file.read_text(encoding="utf-8"))
        routes = {}

        if isinstance(data, dict) and "routes" in data:
            routes_data = data["routes"]
            if isinstance(routes_data, dict):
                for name, payload in routes_data.items():
                    waypoints = payload.get("waypoints", payload)
                    routes[str(name)] = self._convert_waypoints(waypoints)
            elif isinstance(routes_data, list):
                for item in routes_data:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name") or item.get("alias")
                    waypoints = item.get("waypoints")
                    if name is None or waypoints is None:
                        continue
                    routes[str(name)] = self._convert_waypoints(waypoints)
        else:
            waypoints = data.get("waypoints", data) if isinstance(data, dict) else data
            routes["default"] = self._convert_waypoints(waypoints)

        if not routes:
            raise ValueError("未找到可用的巡逻路径")

        chosen_name = None
        if preferred_names:
            for name in preferred_names:
                if name in routes:
                    chosen_name = name
                    break
            if chosen_name is None:
                raise ValueError(f"指定的巡逻路线名称不存在: {preferred_names}")
        else:
            chosen_name = next(iter(routes.keys()))

        return routes, chosen_name, routes[chosen_name]

    def set_target_patrol_route(self, route_name):
        if self.target_policy_source != "patrol":
            return
        if route_name not in self.patrol_routes:
            raise ValueError(f"巡逻路线不存在: {route_name}")
        self.target_patrol_name = route_name
        self.target_patrol_waypoints = self.patrol_routes[route_name]
        self._target_patrol_idx = 0

    def set_target_patrol_waypoints(self, waypoints, route_name=None):
        if self.target_policy_source != "patrol":
            return
        converted = self._convert_waypoints(waypoints)
        name = route_name or self.target_patrol_name or "custom"
        self.patrol_routes[name] = converted
        self.target_patrol_name = name
        self.target_patrol_waypoints = converted
        self._target_patrol_idx = 0

    def get_patrol_route_names(self):
        return list(self.patrol_routes.keys())

    def _calc_obs_dim(self):
        # own pos (2) + own vel (2) + per-other: rel pos (2) + rel vel (2) + dist (1)
        return 4 + (self.agent_num - 1) * 5

    def reset(self):
        self.positions = self.np_random.uniform(low=-self.world_size, high=self.world_size, size=(self.agent_num, 2)).astype(np.float32)
        self.velocities = np.zeros((self.agent_num, 2), dtype=np.float32)
        self._capture_counter = 0
        self._step_count = 0
        self._target_patrol_idx = 0
        return self._get_obs()

    def _patrol_action(self, target_pos):
        waypoint = self.target_patrol_waypoints[self._target_patrol_idx]
        diff = waypoint - target_pos
        dist = np.linalg.norm(diff)
        if dist < self.max_speeds["target"] * self.dt:
            self._target_patrol_idx = (self._target_patrol_idx + 1) % len(self.target_patrol_waypoints)
            waypoint = self.target_patrol_waypoints[self._target_patrol_idx]
            diff = waypoint - target_pos
            dist = np.linalg.norm(diff)
        if dist < 1e-8:
            return np.zeros(2, dtype=np.float32)
        return (diff / dist).astype(np.float32)

    def step(self, actions):
        self._step_count += 1
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.agent_num, self.action_dim):
            raise ValueError(f"Expected actions shape ({self.agent_num}, {self.action_dim}), got {actions.shape}")

        target_idx = self.agent_num - 1
        if self.target_policy_source == "patrol":
            actions[target_idx] = self._patrol_action(self.positions[target_idx])

        for idx in range(self.agent_num):
            role = self.role_names[idx]
            max_speed = self.max_speeds[role]
            action = np.clip(actions[idx], -1.0, 1.0)
            velocity = action * max_speed
            self.velocities[idx] = velocity
            self.positions[idx] += velocity * self.dt
            self.positions[idx] = np.clip(self.positions[idx], -self.world_size, self.world_size)

        obs = self._get_obs()
        rewards, capture = self._get_rewards()
        dones = np.array([capture or self._step_count >= self.max_steps] * self.agent_num)
        infos = self._get_infos(capture)
        return obs, rewards, dones, infos

    def _get_obs(self):
        obs = []
        target_idx = self.agent_num - 1
        target_pos = self.positions[target_idx]
        target_vel = self.velocities[target_idx]

        pursuer_ids = self.role_groups["hunter"] + self.role_groups["blocker"]
        target_spotted = False
        for idx in pursuer_ids:
            perception = self.perception_ranges[self.role_names[idx]]
            if np.linalg.norm(target_pos - self.positions[idx]) <= perception:
                target_spotted = True
                break

        for idx in range(self.agent_num):
            role = self.role_names[idx]
            perception = self.perception_ranges[role]
            own_pos = self.positions[idx]
            own_vel = self.velocities[idx]
            other_features = []
            for jdx in range(self.agent_num):
                if jdx == idx:
                    continue
                rel_pos = self.positions[jdx] - own_pos
                rel_vel = self.velocities[jdx] - own_vel
                dist = np.linalg.norm(rel_pos)
                in_range = dist <= perception
                if jdx == target_idx and role in ("hunter", "blocker") and target_spotted:
                    in_range = True
                if in_range:
                    other_features.extend([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], dist])
                else:
                    other_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            obs_vec = np.concatenate([own_pos, own_vel, np.array(other_features, dtype=np.float32)])
            obs.append(obs_vec.astype(np.float32))
        return np.stack(obs, axis=0)

    def _get_rewards(self):
        target_idx = self.agent_num - 1
        target_pos = self.positions[target_idx]
        hunter_indices = self.role_groups["hunter"]
        distances_to_target = [np.linalg.norm(self.positions[idx] - target_pos) for idx in hunter_indices]
        min_distance = float(np.min(distances_to_target))

        if min_distance <= self.capture_radius:
            self._capture_counter += 1
        else:
            self._capture_counter = 0
        capture = self._capture_counter >= self.capture_steps

        rewards = np.zeros(self.agent_num, dtype=np.float32)
        for idx in range(self.agent_num - 1):
            role = self.role_names[idx]
            dist = np.linalg.norm(self.positions[idx] - target_pos)
            if role == "hunter":
                rewards[idx] = -dist + (10.0 if capture else 0.0)
            else:
                rewards[idx] = -0.7 * dist + (6.0 if capture else 0.0)

        rewards[target_idx] = min_distance - (12.0 if capture else 0.0)
        if self.speed_penalty > 0.0:
            speeds = np.linalg.norm(self.velocities, axis=1)
            rewards -= self.speed_penalty * (speeds ** 2)
        return rewards, capture

    def _get_infos(self, capture):
        infos = []
        for idx in range(self.agent_num):
            infos.append({
                "role": self.role_names[idx],
                "capture": capture,
                "role_groups": self.role_groups,
                "target_policy_source": self.target_policy_source,
                "target_patrol_name": self.target_patrol_name,
            })
        return infos

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return seed
