from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml


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
        initial_target_estimate_noise_std=0.0,
        initial_target_estimate_lag_steps=0,
        pursuer_obs_noise_ratio_min=0.001,
        pursuer_obs_noise_ratio_max=0.1,
        max_hunters_for_obs=4,
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
        self.initial_target_estimate_noise_std = max(0.0, float(initial_target_estimate_noise_std))
        self.initial_target_estimate_lag_steps = max(0, int(initial_target_estimate_lag_steps))
        self.pursuer_obs_noise_ratio_min = max(0.0, float(pursuer_obs_noise_ratio_min))
        self.pursuer_obs_noise_ratio_max = max(0.0, float(pursuer_obs_noise_ratio_max))
        if self.pursuer_obs_noise_ratio_min > self.pursuer_obs_noise_ratio_max:
            self.pursuer_obs_noise_ratio_min, self.pursuer_obs_noise_ratio_max = self.pursuer_obs_noise_ratio_max, self.pursuer_obs_noise_ratio_min
        self.max_hunters_for_obs = max(0, int(max_hunters_for_obs))

        self.positions = np.zeros((self.agent_num, 2), dtype=np.float32)
        self.velocities = np.zeros((self.agent_num, 2), dtype=np.float32)
        # 评估场景可指定固定初始位置，用于复现实验。
        self.fixed_initial_positions = None
        self._capture_counter = 0
        self._step_count = 0
        self._target_pos_history = []
        self._target_belief = {}

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
            return {}, None, None
        route_file = Path(target_patrol_path)
        data = yaml.safe_load(route_file.read_text(encoding="utf-8")) or {}
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
        # + target belief meta: [available, is_exact, age]
        return 4 + (self.agent_num - 1) * 5 + 3


    def _get_pursuer_obs_noise_ratio(self, pursuer_pos, target_pos):
        # 距离 Target 越近，噪声越小；范围由 [min, max] 比例控制。
        distance = float(np.linalg.norm(target_pos - pursuer_pos))
        normalized = np.clip(distance / max(self.world_size * 2.0, 1e-6), 0.0, 1.0)
        return self.pursuer_obs_noise_ratio_min + (self.pursuer_obs_noise_ratio_max - self.pursuer_obs_noise_ratio_min) * normalized

    def _apply_relative_noise(self, vector, ratio):
        # 相对噪声：标准差与向量量级成比例。
        arr = np.asarray(vector, dtype=np.float32)
        sigma = max(float(ratio), 0.0) * np.maximum(np.abs(arr), 1e-3)
        noise = self.np_random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        return arr + noise

    def _capture_target_measurement(self):
        target_idx = self.agent_num - 1
        lag = min(self.initial_target_estimate_lag_steps, len(self._target_pos_history) - 1)
        lagged_pos = self._target_pos_history[-(lag + 1)].copy()
        if len(self._target_pos_history) >= lag + 2:
            prev = self._target_pos_history[-(lag + 2)]
            lagged_vel = (lagged_pos - prev) / max(self.dt, 1e-6)
        else:
            lagged_vel = np.zeros(2, dtype=np.float32)

        if self.initial_target_estimate_noise_std > 0.0:
            noise = self.np_random.normal(0.0, self.initial_target_estimate_noise_std, size=2).astype(np.float32)
            lagged_pos = np.clip(lagged_pos + noise, -self.world_size, self.world_size)

        self._target_belief = {}
        for idx in self.role_groups["hunter"] + self.role_groups["blocker"]:
            self._target_belief[idx] = {
                "pos": lagged_pos.copy(),
                "vel": lagged_vel.copy(),
                "is_exact": False,
                "age": 0,
            }
        self._target_belief[target_idx] = {
            "pos": self.positions[target_idx].copy(),
            "vel": self.velocities[target_idx].copy(),
            "is_exact": True,
            "age": 0,
        }

    def _update_target_beliefs(self):
        target_idx = self.agent_num - 1
        target_pos = self.positions[target_idx]
        target_vel = self.velocities[target_idx]

        for idx in self.role_groups["hunter"] + self.role_groups["blocker"]:
            belief = self._target_belief.get(idx)
            if belief is None:
                continue
            perception = self.perception_ranges[self.role_names[idx]]
            can_observe = np.linalg.norm(target_pos - self.positions[idx]) <= perception
            if can_observe:
                belief["pos"] = target_pos.copy()
                belief["vel"] = target_vel.copy()
                belief["is_exact"] = True
                belief["age"] = 0
            else:
                belief["pos"] = np.clip(belief["pos"] + belief["vel"] * self.dt, -self.world_size, self.world_size)
                belief["age"] = int(belief["age"]) + 1

        self._target_belief[target_idx] = {
            "pos": target_pos.copy(),
            "vel": target_vel.copy(),
            "is_exact": True,
            "age": 0,
        }


    def set_initial_positions(self, initial_positions=None):
        # 设置固定初始位置；若为空则回退到随机初始化。
        if initial_positions is None:
            self.fixed_initial_positions = None
            return
        arr = np.asarray(initial_positions, dtype=np.float32)
        if arr.shape != (self.agent_num, 2):
            raise ValueError(f"initial_positions shape must be ({self.agent_num}, 2), got {arr.shape}")
        self.fixed_initial_positions = np.clip(arr, -self.world_size, self.world_size)

    def apply_scenario_config(self, scenario):
        # 评估时按场景覆盖环境参数。
        if scenario is None:
            return
        self.world_size = float(scenario.get("world_size", self.world_size))
        self.dt = float(scenario.get("dt", self.dt))
        self.capture_radius = float(scenario.get("capture_radius", self.capture_radius))
        self.capture_steps = int(scenario.get("capture_steps", self.capture_steps))
        self.max_steps = int(scenario.get("episode_length", self.max_steps))
        if "seed" in scenario and scenario["seed"] is not None:
            self.seed(int(scenario["seed"]))
        if "target_policy_source" in scenario and scenario["target_policy_source"] is not None:
            self.target_policy_source = str(scenario["target_policy_source"])
        if "initial_target_estimate_noise_std" in scenario and scenario["initial_target_estimate_noise_std"] is not None:
            self.initial_target_estimate_noise_std = max(0.0, float(scenario["initial_target_estimate_noise_std"]))
        if "initial_target_estimate_lag_steps" in scenario and scenario["initial_target_estimate_lag_steps"] is not None:
            self.initial_target_estimate_lag_steps = max(0, int(scenario["initial_target_estimate_lag_steps"]))
        if "pursuer_obs_noise_ratio_min" in scenario and scenario["pursuer_obs_noise_ratio_min"] is not None:
            self.pursuer_obs_noise_ratio_min = max(0.0, float(scenario["pursuer_obs_noise_ratio_min"]))
        if "pursuer_obs_noise_ratio_max" in scenario and scenario["pursuer_obs_noise_ratio_max"] is not None:
            self.pursuer_obs_noise_ratio_max = max(0.0, float(scenario["pursuer_obs_noise_ratio_max"]))
        if self.pursuer_obs_noise_ratio_min > self.pursuer_obs_noise_ratio_max:
            self.pursuer_obs_noise_ratio_min, self.pursuer_obs_noise_ratio_max = self.pursuer_obs_noise_ratio_max, self.pursuer_obs_noise_ratio_min
        if "max_hunters_for_obs" in scenario and scenario["max_hunters_for_obs"] is not None:
            self.max_hunters_for_obs = max(0, int(scenario["max_hunters_for_obs"]))
        self.set_initial_positions(scenario.get("initial_positions"))

    def reset(self):
        if self.fixed_initial_positions is None:
            self.positions = self.np_random.uniform(low=-self.world_size, high=self.world_size, size=(self.agent_num, 2)).astype(np.float32)
        else:
            self.positions = self.fixed_initial_positions.copy()
        self.velocities = np.zeros((self.agent_num, 2), dtype=np.float32)
        self._capture_counter = 0
        self._step_count = 0
        self._target_patrol_idx = 0
        target_idx = self.agent_num - 1
        self._target_pos_history = [self.positions[target_idx].copy()]
        self._capture_target_measurement()
        self._update_target_beliefs()
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

        target_idx = self.agent_num - 1
        self._target_pos_history.append(self.positions[target_idx].copy())
        max_hist = self.initial_target_estimate_lag_steps + 2
        if len(self._target_pos_history) > max_hist:
            self._target_pos_history = self._target_pos_history[-max_hist:]
        self._update_target_beliefs()

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

        for idx in range(self.agent_num):
            role = self.role_names[idx]
            perception = self.perception_ranges[role]
            own_pos = self.positions[idx]
            own_vel = self.velocities[idx]
            other_features = []

            visible_hunter_teammates = set()
            if role == "hunter":
                teammate_candidates = [
                    jdx
                    for jdx in self.role_groups["hunter"]
                    if jdx != idx
                ]
                teammate_candidates.sort(
                    key=lambda jdx: np.linalg.norm(self.positions[jdx] - own_pos)
                )
                visible_hunter_teammates = set(
                    teammate_candidates[: self.max_hunters_for_obs]
                )

            for jdx in range(self.agent_num):
                if jdx == idx:
                    continue

                if (
                    role == "hunter"
                    and self.role_names[jdx] == "hunter"
                    and jdx not in visible_hunter_teammates
                ):
                    other_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                    continue

                if jdx == target_idx and role in ("hunter", "blocker"):
                    belief = self._target_belief[idx]
                    rel_pos = belief["pos"] - own_pos
                    rel_vel = belief["vel"] - own_vel
                    dist = np.linalg.norm(rel_pos)
                    in_range = True
                else:
                    rel_pos = self.positions[jdx] - own_pos
                    rel_vel = self.velocities[jdx] - own_vel
                    dist = np.linalg.norm(rel_pos)
                    in_range = dist <= perception

                if in_range and role in ("hunter", "blocker"):
                    noise_ratio = self._get_pursuer_obs_noise_ratio(own_pos, target_pos)
                    rel_pos = self._apply_relative_noise(rel_pos, noise_ratio)
                    rel_vel = self._apply_relative_noise(rel_vel, noise_ratio)
                    dist = float(np.linalg.norm(rel_pos))

                if in_range:
                    other_features.extend([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], dist])
                else:
                    other_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            if role in ("hunter", "blocker"):
                belief = self._target_belief[idx]
                belief_meta = np.array(
                    [
                        1.0,
                        1.0 if belief["is_exact"] else 0.0,
                        min(float(belief["age"]) / max(float(self.max_steps), 1.0), 1.0),
                    ],
                    dtype=np.float32,
                )
            else:
                belief_meta = np.array([1.0, 1.0, 0.0], dtype=np.float32)

            obs_vec = np.concatenate([own_pos, own_vel, np.array(other_features, dtype=np.float32), belief_meta])
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
                "target_belief_is_exact": bool(self._target_belief.get(idx, {}).get("is_exact", True)),
                "target_belief_age": int(self._target_belief.get(idx, {}).get("age", 0)),
            })
        return infos

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return seed
