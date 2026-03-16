"""
Swarm simulation GUI for search + assignment + pursuit pipeline.

设计目标:
1) 提供单文件GUI入口，支持任务创建、预规划、下发执行、开始/暂停/重置。
2) 按 hybrid_decision_method 描述实现搜索-分配-执行主流程。
3) 复用 env_uav_pursuit 中 Target/Hunter 运动与策略 step/select_action 逻辑。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib import font_manager

project_root = Path(__file__).resolve().parents[1]
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from envs.env_uav_pursuit import ExplorerAgent, HunterAgent, TargetAgent, UAVPursuitEnv
from utils.util import load_config

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog, simpledialog
except Exception:
    tk = None
    ttk = None
    messagebox = None
    filedialog = None
    simpledialog = None

try:
    from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy
    import torch
except Exception:
    RMAPPOPolicy = None
    torch = None


@dataclass
class AssignmentWeights:
    """
    功能:
        定义目标分配匹配成本的可调权重。
    输入:
        无（字段由GUI/配置赋值）。
    输出:
        无。
    """

    distance_weight: float = 1.0
    value_weight: float = 2.0
    endurance_weight: float = 1.0
    switch_weight: float = 0.5
    max_assign_dist: float = 250.0


@dataclass
class ExplorerRuntime:
    """
    功能:
        维护Explorer运行时状态。
    输入:
        agent (ExplorerAgent): Explorer对象。
    输出:
        无。
    """

    agent: ExplorerAgent
    state: str = "SEARCH"
    assigned_target: int = -1
    path: List[np.ndarray] = field(default_factory=list)
    path_index: int = 0
    resume_path_index: int = 0
    total_endurance: float = 12000.0
    remaining_endurance: float = 12000.0


@dataclass
class HunterRuntime:
    """
    功能:
        维护Hunter运行时状态。
    输入:
        agent (HunterAgent): Hunter对象。
    输出:
        无。
    """

    agent: HunterAgent
    standby_mode: str = "split"
    assigned_target: int = -1
    standby_path: List[np.ndarray] = field(default_factory=list)
    standby_index: int = 0
    standby_direction: int = 1
    standby_speed: float = 0.0
    last_target: int = -1
    zone_explorer: int = -1
    zone_offset: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    state: str = "STANDBY"
    resume_pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    total_endurance: float = 12000.0
    remaining_endurance: float = 12000.0
    pursuit_traj_start: int = -1


@dataclass
class TargetRuntime:
    """
    功能:
        维护Target运行时状态与任务池信息。
    输入:
        agent (TargetAgent): Target对象。
    输出:
        无。
    """

    agent: TargetAgent
    value: float = 1.0
    required_hunters: int = 1
    alive: bool = True
    in_pool: bool = False
    discovered: bool = False
    assigned_explorer: int = -1
    assigned_hunters: List[int] = field(default_factory=list)
    pursuit_started: bool = False
    assign_step: int = -1
    last_seen_step: int = -1
    last_seen_pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    last_seen_vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    policy_type: str = "random"
    seen_streak: int = 0


@dataclass
class MissionConfig:
    """
    功能:
        维护任务级配置参数。
    输入:
        无（字段由GUI与配置更新）。
    输出:
        无。
    """

    world_size: float = 100.0
    dt: float = 0.1
    max_steps: int = 3000
    hunters: int = 6
    explorers: int = 3
    targets: int = 5
    explorer_max_speed: float = 1.0
    hunter_max_speed: float = 1.0
    target_max_speed: float = 1.0
    overlap_rate: float = 0.2
    hunters_wait_mode: str = "split"
    explorer_track_speed_scale: float = 1.2
    loss_timeout_steps: int = 40
    target_min_init_separation: float = 5.0
    explorer_total_endurance: float = 12000.0
    hunter_total_endurance: float = 12000.0
    endurance_idle_cost: float = 0.2


@dataclass
class PursuitRuntime:
    """
    功能:
        维护单个Target追捕子任务的子环境与索引映射。
    输入:
        env (UAVPursuitEnv): 子任务环境。
        hunter_ids (List[int]): 全局Hunter ID按子环境局部顺序映射列表。
    输出:
        无。
    """

    env: UAVPursuitEnv
    hunter_ids: List[int] = field(default_factory=list)
    started: bool = False


class MinCostMatcher:
    """
    功能:
        提供最小成本匹配（匈牙利思想的最小费用增广实现）。
    输入:
        无。
    输出:
        无。
    """

    @staticmethod
    def solve(cost: np.ndarray) -> List[Tuple[int, int]]:
        """
        功能:
            计算最小成本匹配结果（支持矩形代价矩阵）。
        输入:
            cost (np.ndarray): shape=(n,m) 的代价矩阵。
        输出:
            List[Tuple[int,int]]: 匹配对(row_idx, col_idx)。
        """
        if cost.size == 0:
            return []
        arr = np.asarray(cost, dtype=np.float64)
        n, m = arr.shape
        transposed = False
        if n > m:
            arr = arr.T
            n, m = arr.shape
            transposed = True

        u = np.zeros(n + 1, dtype=np.float64)
        v = np.zeros(m + 1, dtype=np.float64)
        p = np.zeros(m + 1, dtype=np.int32)
        way = np.zeros(m + 1, dtype=np.int32)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = np.full(m + 1, np.inf, dtype=np.float64)
            used = np.zeros(m + 1, dtype=bool)

            while True:
                used[j0] = True
                i0 = p[j0]
                delta = np.inf
                j1 = 0
                for j in range(1, m + 1):
                    if used[j]:
                        continue
                    cur = arr[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                for j in range(0, m + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break

            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        pairs: List[Tuple[int, int]] = []
        for j in range(1, m + 1):
            if p[j] == 0:
                continue
            row = int(p[j] - 1)
            col = int(j - 1)
            if row < n:
                if transposed:
                    pairs.append((col, row))
                else:
                    pairs.append((row, col))
        return pairs


class LearnTargetActor:
    """
    功能:
        可选learn-target actor封装；无模型时自动回退零动作。
    输入:
        cfg (EasyDict): 合并配置。
        actor_path (Optional[str]): actor权重路径。
        obs_dim (int): 观测维度。
    输出:
        无。
    """

    def __init__(self, cfg, actor_path: Optional[str], obs_dim: int):
        self.enabled = False
        self.policy = None
        self.recurrent_N = 1
        self.hidden_size = 1
        self.rnn_states: Dict[int, np.ndarray] = {}

        if actor_path is None:
            return
        if RMAPPOPolicy is None or torch is None:
            return

        try:
            flat_args = self._build_flat_args_from_cfg(cfg)
            from gymnasium import spaces

            obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(int(obs_dim),), dtype=np.float32)
            act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.policy = RMAPPOPolicy(flat_args, obs_space, obs_space, act_space, device=device)
            ckpt = torch.load(str(actor_path), map_location=device)
            self.policy.actor.load_state_dict(ckpt)
            self.policy.actor.eval()
            self.recurrent_N = int(flat_args.recurrent_N)
            self.hidden_size = int(flat_args.hidden_size)
            self.enabled = True
        except Exception:
            self.enabled = False

    def _build_flat_args_from_cfg(self, merged_cfg):
        """
        功能:
            将分层配置映射为策略初始化所需扁平参数。
        输入:
            merged_cfg (EasyDict): 合并配置。
        输出:
            argparse.Namespace: 算法参数。
        """
        from runner.uav.role_runner import RoleBasedRunner

        class _Dummy(object):
            pass

        dummy = _Dummy()
        dummy.cfg = merged_cfg
        return RoleBasedRunner._build_flat_args_for_algorithm(dummy)

    def reset_target(self, target_id: int):
        """
        功能:
            重置目标的RNN状态。
        输入:
            target_id (int): 目标ID。
        输出:
            无。
        """
        self.rnn_states[int(target_id)] = np.zeros(
            (1, self.recurrent_N, self.hidden_size), dtype=np.float32
        )

    def act(self, target_id: int, obs: np.ndarray) -> np.ndarray:
        """
        功能:
            对learn目标执行一次策略前向推理。
        输入:
            target_id (int): 目标ID。
            obs (np.ndarray): shape=(obs_dim,)。
        输出:
            np.ndarray: shape=(2,) 归一化动作。
        """
        if (not self.enabled) or self.policy is None:
            return np.zeros(2, dtype=np.float32)
        tid = int(target_id)
        if tid not in self.rnn_states:
            self.reset_target(tid)
        rnn_state = self.rnn_states[tid]
        obs_batch = np.asarray(obs, dtype=np.float32)[None, :]
        masks = np.ones((1, 1), dtype=np.float32)
        with torch.no_grad():
            action_t, next_rnn = self.policy.act(obs_batch, rnn_state, masks, deterministic=True)
        self.rnn_states[tid] = next_rnn.detach().cpu().numpy().astype(np.float32)
        action = action_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return np.clip(action, -1.0, 1.0)


class LearnHunterActor:
    """
    功能:
        可选learn-hunter actor封装；无模型时自动回退None。
    输入:
        cfg (EasyDict): 合并配置。
        actor_path (Optional[str]): actor权重路径。
    输出:
        无。
    """

    def __init__(self, cfg, actor_path: Optional[str]):
        self.cfg = cfg
        self.actor_path = actor_path
        self.enabled = False
        self.policy = None
        self.recurrent_N = 1
        self.hidden_size = 1
        self.rnn_states: Dict[int, np.ndarray] = {}

        if actor_path is None:
            return
        if RMAPPOPolicy is None or torch is None:
            return

    def _build_flat_args_from_cfg(self, merged_cfg):
        """
        功能:
            将分层配置映射为策略初始化所需扁平参数。
        输入:
            merged_cfg (EasyDict): 合并配置。
        输出:
            argparse.Namespace: 算法参数。
        """
        from runner.uav.role_runner import RoleBasedRunner

        class _Dummy(object):
            pass

        dummy = _Dummy()
        dummy.cfg = merged_cfg
        return RoleBasedRunner._build_flat_args_for_algorithm(dummy)

    def _ensure_policy(self, obs_dim: int):
        """
        功能:
            按obs_dim初始化策略并加载权重。
        输入:
            obs_dim (int): 观测维度。
        输出:
            无。
        """
        if self.enabled or self.policy is not None:
            return
        if self.actor_path is None or RMAPPOPolicy is None or torch is None:
            return
        try:
            flat_args = self._build_flat_args_from_cfg(self.cfg)
            from gymnasium import spaces

            obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(int(obs_dim),), dtype=np.float32)
            act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.policy = RMAPPOPolicy(flat_args, obs_space, obs_space, act_space, device=device)
            ckpt = torch.load(str(self.actor_path), map_location=device)
            self.policy.actor.load_state_dict(ckpt)
            self.policy.actor.eval()
            self.recurrent_N = int(flat_args.recurrent_N)
            self.hidden_size = int(flat_args.hidden_size)
            self.enabled = True
        except Exception:
            self.enabled = False
            self.policy = None

    def reset_hunter(self, hunter_id: int):
        """
        功能:
            重置指定Hunter的RNN状态。
        输入:
            hunter_id (int): Hunter ID。
        输出:
            无。
        """
        if not self.enabled:
            return
        self.rnn_states[int(hunter_id)] = np.zeros(
            (1, self.recurrent_N, self.hidden_size), dtype=np.float32
        )

    def act(self, hunter_id: int, obs: np.ndarray) -> Optional[np.ndarray]:
        """
        功能:
            对learn-hunter执行一次策略前向推理。
        输入:
            hunter_id (int): Hunter ID。
            obs (np.ndarray): shape=(obs_dim,)。
        输出:
            Optional[np.ndarray]: shape=(2,) 归一化动作。
        """
        if obs is None:
            return None
        if self.policy is None:
            self._ensure_policy(int(obs.shape[0]))
        if (not self.enabled) or self.policy is None:
            return None
        hid = int(hunter_id)
        if hid not in self.rnn_states:
            self.reset_hunter(hid)
        rnn_state = self.rnn_states[hid]
        obs_batch = np.asarray(obs, dtype=np.float32)[None, :]
        masks = np.ones((1, 1), dtype=np.float32)
        with torch.no_grad():
            action_t, next_rnn = self.policy.act(obs_batch, rnn_state, masks, deterministic=True)
        self.rnn_states[hid] = next_rnn.detach().cpu().numpy().astype(np.float32)
        action = action_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return np.clip(action, -1.0, 1.0)


class SwarmSimulationCore:
    """
    功能:
        实现搜索-分配-追捕全流程的仿真核心。
    输入:
        cfg (EasyDict): 合并配置。
        seed (Optional[int]): 随机种子。
        target_actor_path (Optional[str]): learn-target actor路径。
        hunter_actor_path (Optional[str]): learn-hunter actor路径。
    输出:
        无。
    """

    def __init__(
        self,
        cfg,
        seed: Optional[int] = None,
        target_actor_path: Optional[str] = None,
        hunter_actor_path: Optional[str] = None,
        sim_overrides: Optional[dict] = None,
    ):
        self.cfg = cfg
        self.base_seed = int(cfg.exp.seed if seed is None else seed)
        self.current_seed = int(self.base_seed)
        self.rng = np.random.RandomState(int(self.current_seed))

        env_cfg = cfg.env
        multi_env_cfg = None
        if hasattr(cfg, "multi_infer") and hasattr(cfg.multi_infer, "env"):
            multi_env_cfg = cfg.multi_infer.env
        self.mission = MissionConfig(
            world_size=float(env_cfg.world_size),
            dt=float(env_cfg.dt),
            max_steps=int(cfg.eval.eval_episode_length if hasattr(cfg, "eval") else env_cfg.episode_length),
            hunters=max(1, int(env_cfg.max_hunters_num)),
            explorers=max(1, int(getattr(multi_env_cfg, "num_explorers", 3)) if multi_env_cfg is not None else 3),
            targets=max(1, int(getattr(multi_env_cfg, "num_targets", 5)) if multi_env_cfg is not None else 5),
            explorer_max_speed=float(getattr(cfg.Explorer, "max_velo", 1.0)),
            hunter_max_speed=float(getattr(cfg.Hunter, "max_velo", 1.0)),
            target_max_speed=float(getattr(cfg.Target, "max_velo", 1.0)),
            overlap_rate=0.2,
            hunters_wait_mode="split",
            explorer_track_speed_scale=1.2,
            loss_timeout_steps=40,
            target_min_init_separation=5.0,
        )

        default_policy = str(env_cfg.target_policy_source).lower()
        self.target_policy_choices: List[str] = [default_policy]
        self.target_policy_probs: Dict[str, float] = {default_policy: 1.0}

        self.weights = AssignmentWeights()
        self._apply_sim_overrides(sim_overrides)
        self.time_step = 0
        self.executing = False
        self.planned = False
        self.pending_assignment: Optional[List[Tuple[int, int, List[int]]]] = None
        self.assign_mode: str = "on-loop"

        self.explorers: List[ExplorerRuntime] = []
        self.hunters: List[HunterRuntime] = []
        self.targets: List[TargetRuntime] = []
        self.pursuit_tasks: Dict[int, PursuitRuntime] = {}

        self.patrol_routes = self._load_patrol_routes(
            route_path=str(env_cfg.target_patrol_path),
            route_names=list(env_cfg.target_patrol_names),
        )
        self.target_actor = LearnTargetActor(cfg, target_actor_path, obs_dim=20)
        self.hunter_actor = LearnHunterActor(cfg, hunter_actor_path) if hunter_actor_path else None
        self.reset_world()

    def reset_world_with_seed(self, seed: int):
        """
        功能:
            使用指定随机种子重置世界。
        输入:
            seed (int): 随机种子。
        输出:
            无。
        """
        self.current_seed = int(seed)
        self.rng = np.random.RandomState(int(self.current_seed))
        self.reset_world()

    def _apply_sim_overrides(self, sim_overrides: Optional[dict]):
        """
        功能:
            将独立仿真配置覆盖到任务参数与分配权重。
        输入:
            sim_overrides (Optional[dict]): 独立仿真配置字典。
        输出:
            无。
        """
        if not isinstance(sim_overrides, dict):
            return

        mission_cfg = sim_overrides.get("mission", {})
        if isinstance(mission_cfg, dict):
            self.mission.world_size = float(mission_cfg.get("world_size", self.mission.world_size))
            self.mission.dt = float(mission_cfg.get("dt", self.mission.dt))
            self.mission.max_steps = int(mission_cfg.get("max_steps", self.mission.max_steps))
            self.mission.hunters = max(1, int(mission_cfg.get("hunters", self.mission.hunters)))
            self.mission.explorers = max(1, int(mission_cfg.get("explorers", self.mission.explorers)))
            self.mission.targets = max(1, int(mission_cfg.get("targets", self.mission.targets)))
            self.mission.explorer_max_speed = float(
                mission_cfg.get("explorer_max_speed", self.mission.explorer_max_speed)
            )
            self.mission.hunter_max_speed = float(
                mission_cfg.get("hunter_max_speed", self.mission.hunter_max_speed)
            )
            self.mission.target_max_speed = float(
                mission_cfg.get("target_max_speed", self.mission.target_max_speed)
            )
            self.mission.overlap_rate = float(mission_cfg.get("overlap_rate", self.mission.overlap_rate))
            self.mission.hunters_wait_mode = str(
                mission_cfg.get("hunters_wait_mode", self.mission.hunters_wait_mode)
            ).lower()
            if self.mission.hunters_wait_mode not in ("split", "zone"):
                self.mission.hunters_wait_mode = "split"
            self.mission.explorer_track_speed_scale = float(
                mission_cfg.get("explorer_track_speed_scale", self.mission.explorer_track_speed_scale)
            )
            self.mission.loss_timeout_steps = max(
                1,
                int(mission_cfg.get("loss_timeout_steps", self.mission.loss_timeout_steps)),
            )
            self.mission.target_min_init_separation = float(
                mission_cfg.get("target_min_init_separation", self.mission.target_min_init_separation)
            )
            self.mission.explorer_total_endurance = float(
                mission_cfg.get("explorer_total_endurance", self.mission.explorer_total_endurance)
            )
            self.mission.hunter_total_endurance = float(
                mission_cfg.get("hunter_total_endurance", self.mission.hunter_total_endurance)
            )
            self.mission.endurance_idle_cost = float(
                mission_cfg.get("endurance_idle_cost", self.mission.endurance_idle_cost)
            )

        assign_cfg = sim_overrides.get("assignment", {})
        if isinstance(assign_cfg, dict):
            self.weights.distance_weight = float(
                assign_cfg.get("distance_weight", self.weights.distance_weight)
            )
            self.weights.value_weight = float(assign_cfg.get("value_weight", self.weights.value_weight))
            self.weights.endurance_weight = float(
                assign_cfg.get("endurance_weight", self.weights.endurance_weight)
            )
            self.weights.switch_weight = float(assign_cfg.get("switch_weight", self.weights.switch_weight))
            self.weights.max_assign_dist = float(
                assign_cfg.get("max_assign_dist", self.weights.max_assign_dist)
            )

        target_cfg = sim_overrides.get("target", {})
        if isinstance(target_cfg, dict):
            choices_raw = target_cfg.get("policy_choices", self.target_policy_choices)
            if isinstance(choices_raw, list) and len(choices_raw) > 0:
                self.target_policy_choices = [str(x).lower() for x in choices_raw]
            probs_raw = target_cfg.get("policy_probs", self.target_policy_probs)
            if isinstance(probs_raw, dict) and len(probs_raw) > 0:
                tmp_probs: Dict[str, float] = {}
                for key, val in probs_raw.items():
                    tmp_probs[str(key).lower()] = float(max(0.0, float(val)))
                if len(tmp_probs) > 0:
                    self.target_policy_probs = tmp_probs

    def reset_world(self):
        """
        功能:
            重置仿真世界并初始化全部实体状态。
        输入:
            无。
        输出:
            无。
        """
        self.time_step = 0
        self.executing = False
        self.planned = False
        self.pending_assignment = None

        self.explorers = []
        self.hunters = []
        self.targets = []
        for runtime in self.pursuit_tasks.values():
            runtime.env.close()
        self.pursuit_tasks = {}

        explorer_cfg = getattr(self.cfg, "Explorer")
        for idx in range(int(self.mission.explorers)):
            agent = ExplorerAgent(
                agent_id=idx,
                max_speed=float(self.mission.explorer_max_speed),
                safe_dis=float(explorer_cfg.safe_dis),
                control_mode="velocity",
                max_acc=0.0,
                max_turn_angle=180.0,
                min_turn_limit_velo=0.0,
                policy_type="search",
            )
            init_pos = self._sample_position()
            agent.reset(init_pos)
            self.explorers.append(
                ExplorerRuntime(
                    agent=agent,
                    total_endurance=float(self.mission.explorer_total_endurance),
                    remaining_endurance=float(self.mission.explorer_total_endurance),
                )
            )

        hunter_cfg = getattr(self.cfg, "Hunter")
        for idx in range(int(self.mission.hunters)):
            agent = HunterAgent(
                agent_id=idx,
                max_speed=float(self.mission.hunter_max_speed),
                safe_dis=float(hunter_cfg.safe_dis),
                control_mode=str(hunter_cfg.control_mode).lower(),
                max_acc=float(hunter_cfg.max_acc),
                max_turn_angle=float(hunter_cfg.max_turn_angle),
                min_turn_limit_velo=float(hunter_cfg.min_turn_limit_velo),
                policy_type="learn",
                block_length=0.0,
            )
            init_pos = self._sample_position()
            agent.reset(init_pos)
            speed = float(self.rng.uniform(0.0, float(hunter_cfg.max_velo)))
            self.hunters.append(
                HunterRuntime(
                    agent=agent,
                    standby_speed=speed,
                    total_endurance=float(self.mission.hunter_total_endurance),
                    remaining_endurance=float(self.mission.hunter_total_endurance),
                    resume_pos=np.asarray(init_pos, dtype=np.float32).copy(),
                )
            )

        target_cfg = getattr(self.cfg, "Target")
        target_init_positions: List[np.ndarray] = []
        policy_plan: List[str] = []
        unique_choices = []
        for c in self.target_policy_choices:
            cc = str(c).lower()
            if cc not in unique_choices:
                unique_choices.append(cc)
        if len(unique_choices) > 1 and int(self.mission.targets) >= len(unique_choices):
            perm = list(unique_choices)
            self.rng.shuffle(perm)
            policy_plan.extend(perm)
        while len(policy_plan) < int(self.mission.targets):
            policy_plan.append(self._sample_target_policy_type())

        for idx in range(int(self.mission.targets)):
            policy_source = str(policy_plan[idx]).lower()
            patrol_route = self.patrol_routes[idx % len(self.patrol_routes)] if len(self.patrol_routes) > 0 else None
            agent = TargetAgent(
                agent_id=idx,
                max_speed=float(self.mission.target_max_speed),
                safe_dis=float(target_cfg.safe_dis),
                control_mode=str(target_cfg.control_mode).lower(),
                max_acc=float(target_cfg.max_acc),
                max_turn_angle=float(target_cfg.max_turn_angle),
                min_turn_limit_velo=float(target_cfg.min_turn_limit_velo),
                policy_type=policy_source,
                patrol_waypoints=patrol_route,
                patrol_routes=self.patrol_routes,
                switch_interval=max(1, int(self.cfg.env.target_switch_interval)),
                control_dt=float(self.mission.dt),
                world_size=float(self.mission.world_size),
                escape_dis=float(getattr(self.cfg.reward, "escape_radius", 30.0)),
                escape_gap_angle_bins=int(getattr(self.cfg.reward, "escape_gap_angle_bins", 360)),
                escape_gap_hunter_reward_scale=0.0,
                escape_gap_target_reward_scale=0.0,
                escape_gap_encircle_hunter_reward_scale=0.0,
                escape_gap_encircle_target_reward_scale=0.0,
                escape_gap_intercept_hunter_reward_scale=0.0,
                escape_gap_intercept_target_reward_scale=0.0,
                escape_gap_min_speed=float(getattr(self.cfg.reward, "escape_gap_min_speed", 0.2)),
                boundary_avoid_enable=bool(getattr(self.cfg.env, "target_boundary_avoid_enable", True)),
                boundary_influence_ratio=float(getattr(self.cfg.env, "target_boundary_influence_ratio", 0.30)),
                boundary_enter_ratio=float(getattr(self.cfg.env, "target_boundary_enter_ratio", 0.15)),
                boundary_exit_ratio=float(getattr(self.cfg.env, "target_boundary_exit_ratio", 0.22)),
                boundary_wall_gain=float(getattr(self.cfg.env, "target_boundary_wall_gain", 1.2)),
                boundary_corner_tangent_gain=float(getattr(self.cfg.env, "target_boundary_corner_tangent_gain", 0.8)),
                boundary_smooth_alpha=float(getattr(self.cfg.env, "target_boundary_smooth_alpha", 0.25)),
                boundary_lookahead_steps=int(getattr(self.cfg.env, "target_boundary_lookahead_steps", 5)),
            )
            init_pos = self._sample_position_with_separation(
                existed=target_init_positions,
                min_dist=float(self.mission.target_min_init_separation),
            )
            if policy_source == "patrol" and len(self.patrol_routes) > 0:
                route_index = int(idx % len(self.patrol_routes))
                waypoint_count = len(self.patrol_routes[route_index])
                waypoint_idx = self._select_patrol_start_waypoint_index(
                    route=self.patrol_routes[route_index],
                    existed=target_init_positions,
                    min_dist=float(self.mission.target_min_init_separation),
                ) if waypoint_count > 0 else 0
                agent.reset(init_pos, force_route_index=route_index)
                if waypoint_count > 0:
                    agent.patrol_index = waypoint_idx
                    agent.position = np.asarray(
                        self.patrol_routes[route_index][waypoint_idx], dtype=np.float32
                    ).copy()
                    agent.trajectory = [agent.position.copy()]
            else:
                agent.reset(init_pos)
            target_init_positions.append(np.asarray(agent.position, dtype=np.float32).copy())
            self.target_actor.reset_target(idx)
            value = float(self.rng.uniform(1.0, 10.0))
            required = int(np.clip(int(round(value / 2.0)), 1, 5))
            self.targets.append(
                TargetRuntime(
                    agent=agent,
                    value=value,
                    required_hunters=required,
                    alive=True,
                    policy_type=str(policy_source).lower(),
                )
            )

    def apply_task_settings(self, mission: MissionConfig, weights: AssignmentWeights):
        """
        功能:
            应用任务设置并重置世界。
        输入:
            mission (MissionConfig): 新任务参数。
            weights (AssignmentWeights): 分配权重参数。
        输出:
            无。
        """
        self.mission = mission
        self.weights = weights
        self.reset_world()

    def plan_routes(self):
        """
        功能:
            执行搜索航线与待命航线预规划。
        输入:
            无。
        输出:
            无。
        """
        explorer_paths = self._build_explorer_search_paths(
            world_size=float(self.mission.world_size),
            num_explorers=int(self.mission.explorers),
            overlap_rate=float(self.mission.overlap_rate),
        )
        for idx, runtime in enumerate(self.explorers):
            runtime.path = explorer_paths[idx]
            runtime.path_index = 0
            runtime.resume_path_index = 0
            runtime.state = "SEARCH"
            runtime.assigned_target = -1
            if len(runtime.path) > 0:
                runtime.agent.reset(runtime.path[0].copy())

        if str(self.mission.hunters_wait_mode).lower() == "split":
            hunter_paths = self._build_hunter_split_paths(
                world_size=float(self.mission.world_size),
                num_hunters=int(self.mission.hunters),
            )
            for idx, runtime in enumerate(self.hunters):
                runtime.standby_mode = "split"
                runtime.zone_explorer = -1
                runtime.standby_path = hunter_paths[idx]
                runtime.standby_index = 0
                runtime.standby_direction = 1
                runtime.assigned_target = -1
                if len(runtime.standby_path) > 0:
                    runtime.agent.reset(runtime.standby_path[0].copy())
        else:
            self._assign_zone_groups()

        self.planned = True

    def dispatch_execute(self):
        """
        功能:
            将任务切换到执行状态（预规划后生效）。
        输入:
            无。
        输出:
            无。
        """
        if not self.planned:
            return
        self.executing = True

    def step_once(self):
        """
        功能:
            推进仿真一步，执行搜索、发现、分配与追捕。
        输入:
            无。
        输出:
            无。
        """
        if not self.executing:
            return
        if self.time_step >= int(self.mission.max_steps):
            self.executing = False
            return

        self.time_step += 1
        self._move_targets()
        self._move_explorers_and_hunters()
        self._update_discovery_pool()
        self._assignment_if_needed()
        self._step_pursuit_subtasks()
        self._update_pursuit_progress()

        alive_count = sum(1 for t in self.targets if t.alive)
        if alive_count <= 0:
            self.executing = False

    def get_summary(self) -> Dict[str, float]:
        """
        功能:
            获取运行状态汇总指标。
        输入:
            无。
        输出:
            Dict[str,float]: 指标字典。
        """
        alive_targets = sum(1 for t in self.targets if t.alive)
        in_pool = sum(1 for t in self.targets if t.alive and t.in_pool)
        pursuing = sum(1 for t in self.targets if t.alive and t.pursuit_started)
        free_hunters = sum(1 for h in self.hunters if h.assigned_target < 0)
        free_explorers = sum(1 for e in self.explorers if e.assigned_target < 0 and e.state == "SEARCH")
        captured = int(len(self.targets) - alive_targets)
        return {
            "step": float(self.time_step),
            "alive_targets": float(alive_targets),
            "pool_targets": float(in_pool),
            "pursuing_targets": float(pursuing),
            "captured_targets": float(captured),
            "free_hunters": float(free_hunters),
            "free_explorers": float(free_explorers),
        }

    def _sample_position(self) -> np.ndarray:
        """
        功能:
            在地图内随机采样位置。
        输入:
            无。
        输出:
            np.ndarray: shape=(2,)。
        """
        ws = float(self.mission.world_size)
        return self.rng.uniform(-ws * 0.9, ws * 0.9, size=(2,)).astype(np.float32)

    def _sample_position_with_separation(self, existed: List[np.ndarray], min_dist: float) -> np.ndarray:
        """
        功能:
            采样与既有位置保持最小间隔的新位置。
        输入:
            existed (List[np.ndarray]): 已占用位置列表。
            min_dist (float): 最小间隔距离。
        输出:
            np.ndarray: 新采样位置。
        """
        min_d = float(max(0.0, min_dist))
        candidate = self._sample_position()
        for _ in range(50):
            ok = True
            for p in existed:
                if float(np.linalg.norm(candidate - np.asarray(p, dtype=np.float32))) < min_d:
                    ok = False
                    break
            if ok:
                return candidate
            candidate = self._sample_position()
        return candidate

    def _sample_target_policy_type(self) -> str:
        """
        功能:
            按独立配置采样Target策略类型。
        输入:
            无。
        输出:
            str: 策略类型。
        """
        choices = [str(x).lower() for x in self.target_policy_choices]
        if len(choices) <= 0:
            return str(self.cfg.env.target_policy_source).lower()

        probs = np.asarray([float(max(0.0, self.target_policy_probs.get(c, 0.0))) for c in choices], dtype=np.float32)
        if float(np.sum(probs)) <= 1e-8:
            return choices[int(self.rng.randint(0, len(choices)))]
        probs = probs / float(np.sum(probs))
        return str(self.rng.choice(choices, p=probs))

    def _select_patrol_start_waypoint_index(
        self,
        route: List[np.ndarray],
        existed: List[np.ndarray],
        min_dist: float,
    ) -> int:
        """
        功能:
            在巡逻航点中选择与既有目标尽量分散的起始航点索引。
        输入:
            route (List[np.ndarray]): 巡逻航点列表。
            existed (List[np.ndarray]): 已占用目标位置列表。
            min_dist (float): 目标最小间隔期望。
        输出:
            int: 起始航点索引。
        """
        if len(route) <= 0:
            return 0
        if len(existed) <= 0:
            return int(self.rng.randint(0, len(route)))

        best_idx = 0
        best_score = -1e9
        min_d_req = float(max(0.0, min_dist))
        for idx, wp in enumerate(route):
            pos = np.asarray(wp, dtype=np.float32)
            dmin = min(
                float(np.linalg.norm(pos - np.asarray(ep, dtype=np.float32))) for ep in existed
            )
            score = dmin if dmin >= min_d_req else (dmin - min_d_req)
            if score > best_score:
                best_score = score
                best_idx = idx
        return int(best_idx)

    def _move_targets(self):
        """
        功能:
            推进所有存活Target一步，策略行为复用TargetAgent。
        输入:
            无。
        输出:
            无。
        """
        for tid, target in enumerate(self.targets):
            if not target.alive:
                continue
            if target.pursuit_started and tid in self.pursuit_tasks:
                continue
            active_hunters, active_mask = self._active_hunters_for_target(tid)
            action_from_policy = None
            if str(target.agent.policy_type).lower() == "learn":
                action_from_policy = np.zeros(2, dtype=np.float32)
            action = target.agent.select_action(
                step_count=int(self.time_step),
                action_from_policy=action_from_policy,
                rng=self.rng,
                hunters=active_hunters,
                active_hunter_mask=active_mask,
            )
            target.agent.step(action, dt=float(self.mission.dt), world_size=float(self.mission.world_size))

    def _move_explorers_and_hunters(self):
        """
        功能:
            推进Explorer与Hunter状态机运动。
        输入:
            无。
        输出:
            无。
        """
        for ex in self.explorers:
            if ex.remaining_endurance <= 0.0:
                ex.agent.velocity[:] = 0.0
                continue
            if ex.state == "SEARCH":
                self._move_along_path(ex.agent, ex.path, ex)
            elif ex.state == "RETURN":
                self._move_along_path(ex.agent, ex.path, ex)
                if ex.path_index == ex.resume_path_index:
                    ex.state = "SEARCH"
            elif ex.state == "TRACK":
                tid = int(ex.assigned_target)
                if tid < 0 or tid >= len(self.targets) or (not self.targets[tid].alive):
                    ex.state = "RETURN"
                    ex.assigned_target = -1
                    continue
                target = self.targets[tid]
                speed = float(ex.agent.max_speed) * float(self.mission.explorer_track_speed_scale)
                self._move_towards(ex.agent, target.agent.position, speed=speed)

            self._consume_endurance(ex, speed=float(np.linalg.norm(ex.agent.velocity)))

        for h in self.hunters:
            if h.remaining_endurance <= 0.0:
                h.state = "EXHAUSTED"
                h.assigned_target = -1
                h.agent.velocity[:] = 0.0
                continue
            if h.assigned_target < 0:
                if h.state == "RETURN":
                    arrived = self._move_towards(h.agent, h.resume_pos, speed=float(h.agent.max_speed))
                    if arrived:
                        h.state = "STANDBY"
                    self._consume_endurance(h, speed=float(np.linalg.norm(h.agent.velocity)))
                    continue
                if h.standby_mode == "split":
                    self._move_hunter_split_standby(h)
                else:
                    self._move_hunter_zone_standby(h)
                self._consume_endurance(h, speed=float(np.linalg.norm(h.agent.velocity)))
                continue

            tid = int(h.assigned_target)
            if tid < 0 or tid >= len(self.targets) or (not self.targets[tid].alive):
                self._release_hunter(h)
                continue

            target = self.targets[tid]
            if not target.pursuit_started:
                if target.assigned_explorer >= 0:
                    anchor = self.explorers[target.assigned_explorer].agent.position
                    self._move_towards(h.agent, anchor, speed=float(h.agent.max_speed))
                    self._consume_endurance(h, speed=float(np.linalg.norm(h.agent.velocity)))
                continue

            if tid in self.pursuit_tasks:
                continue

            chase_action = self._build_hunter_chase_action(h.agent, target.agent.position)
            h.agent.step(
                action_norm=chase_action,
                dt=float(self.mission.dt),
                world_size=float(self.mission.world_size),
            )
            self._consume_endurance(h, speed=float(np.linalg.norm(h.agent.velocity)))

    def _consume_endurance(self, runtime, speed: float):
        """
        功能:
            按 speed * dt + c 扣减续航。
        输入:
            runtime (ExplorerRuntime|HunterRuntime): 智能体运行时对象。
            speed (float): 当前速度模长。
        输出:
            无。
        """
        cost = float(max(0.0, speed) * float(self.mission.dt) + float(max(0.0, self.mission.endurance_idle_cost)))
        runtime.remaining_endurance = float(max(0.0, float(runtime.remaining_endurance) - cost))

    def _step_pursuit_subtasks(self):
        """
        功能:
            推进所有已启动的追捕子任务（观测与step均复用UAVPursuitEnv）。
        输入:
            无。
        输出:
            无。
        """
        for tid, runtime in list(self.pursuit_tasks.items()):
            if tid < 0 or tid >= len(self.targets):
                continue
            target = self.targets[tid]
            if (not target.alive) or (not target.pursuit_started):
                continue

            env = runtime.env
            if not bool(runtime.started):
                self._sync_subenv_from_global(target_id=tid)
                runtime.started = True
            actions = np.zeros((env.agent_num, 2), dtype=np.float32)
            need_obs = False
            if self.hunter_actor is not None and bool(self.hunter_actor.actor_path):
                need_obs = True
            if str(env.target.policy_type).lower() == "learn":
                need_obs = True
            obs_all = None
            if need_obs:
                team_sees_target = bool(env._team_sees_target())
                obs_all = env._build_obs(team_sees_target=team_sees_target)

            for local_hid, global_hid in enumerate(runtime.hunter_ids):
                if global_hid < 0 or global_hid >= len(self.hunters):
                    continue
                action = None
                if self.hunter_actor is not None and obs_all is not None:
                    hunter_obs = np.asarray(obs_all[local_hid], dtype=np.float32)
                    action = self.hunter_actor.act(int(global_hid), hunter_obs)
                if action is None:
                    action = self._build_hunter_chase_action(
                        hunter=env.hunters[local_hid],
                        target_pos=env.target.position,
                    )
                actions[local_hid] = action

            if str(env.target.policy_type).lower() == "learn" and obs_all is not None:
                target_obs = np.asarray(obs_all[env.target_index], dtype=np.float32)
                actions[env.target_index] = self.target_actor.act(tid, target_obs)

            env.step(actions)
            self._sync_target_task_from_subenv(tid)

    def _sync_subenv_from_global(self, target_id: int):
        """
        功能:
            将全局状态同步到追捕子环境（用于子任务正式启动前的首次对齐）。
        输入:
            target_id (int): 目标ID。
        输出:
            无。
        """
        runtime = self.pursuit_tasks.get(target_id, None)
        if runtime is None:
            return
        env = runtime.env
        target = self.targets[target_id]

        for local_hid, global_hid in enumerate(runtime.hunter_ids):
            if global_hid < 0 or global_hid >= len(self.hunters):
                continue
            global_h = self.hunters[global_hid].agent
            local_h = env.hunters[local_hid]
            local_h.position = np.asarray(global_h.position, dtype=np.float32).copy()
            local_h.velocity = np.asarray(global_h.velocity, dtype=np.float32).copy()
            local_h.heading = np.asarray(global_h.heading, dtype=np.float32).copy()
            local_h.alive = bool(global_h.alive)

        local_t = env.target
        global_t = target.agent
        local_t.position = np.asarray(global_t.position, dtype=np.float32).copy()
        local_t.velocity = np.asarray(global_t.velocity, dtype=np.float32).copy()
        local_t.heading = np.asarray(global_t.heading, dtype=np.float32).copy()
        local_t.alive = bool(global_t.alive)

    def _update_discovery_pool(self):
        """
        功能:
            更新目标发现池与共享信息。
        输入:
            无。
        输出:
            无。
        """
        for tid, target in enumerate(self.targets):
            if not target.alive:
                continue
            any_seen = False
            assigned_seen = False
            assigned_dist = None
            for ex_idx, ex in enumerate(self.explorers):
                dist = float(np.linalg.norm(ex.agent.position - target.agent.position))
                perc = float(getattr(self.cfg.Explorer, "perception_radius", -1))
                perc = float(self.mission.world_size * 2.0) if perc <= 0 else perc
                if dist <= perc:
                    any_seen = True
                    target.discovered = True
                    target.in_pool = True
                    target.last_seen_step = int(self.time_step)
                    target.last_seen_pos = target.agent.position.copy()
                    target.last_seen_vel = target.agent.velocity.copy()
                if int(target.assigned_explorer) >= 0 and int(ex_idx) == int(target.assigned_explorer):
                    assigned_seen = True
                    assigned_dist = dist
            if target.assigned_explorer >= 0:
                if assigned_seen:
                    target.seen_streak = int(target.seen_streak) + 1
                else:
                    target.seen_streak = 0
                perc = float(getattr(self.cfg.Explorer, "perception_radius", -1))
                perc = float(self.mission.world_size * 2.0) if perc <= 0 else perc
                if assigned_seen and assigned_dist is not None:
                    if float(assigned_dist) <= float(perc) * 0.8 and int(target.seen_streak) > 5:
                        if not bool(target.pursuit_started):
                            self._start_pursuit_for_target(int(tid))
                        target.pursuit_started = True

    def _assignment_if_needed(self):
        """
        功能:
            按任务池与空闲资源触发目标分配。
        输入:
            无。
        输出:
            无。
        """
        if self.pending_assignment is not None:
            return
        assignments = self._compute_assignment()
        if len(assignments) == 0:
            return
        if str(self.assign_mode).lower() == "in-loop":
            self.pending_assignment = assignments
            self.executing = False
            return
        self._apply_assignment(assignments)

    def _compute_assignment(self) -> List[Tuple[int, int, List[int]]]:
        """
        功能:
            计算Explorer/Hunter分配方案，不直接下发。
        输入:
            无。
        输出:
            List[Tuple[int,int,List[int]]]: (explorer_id, target_id, hunter_ids)
        """
        idle_explorer_ids = [
            idx for idx, ex in enumerate(self.explorers)
            if float(ex.remaining_endurance) > 0.0
        ]
        idle_hunter_ids = [
            idx
            for idx, h in enumerate(self.hunters)
            if h.state != "EXHAUSTED" and float(h.remaining_endurance) > 0.0
        ]
        candidate_target_ids = [
            idx for idx, t in enumerate(self.targets)
            if t.alive and t.in_pool
        ]
        if len(idle_explorer_ids) == 0 or len(idle_hunter_ids) == 0 or len(candidate_target_ids) == 0:
            return []

        explorer_cost = self._build_explorer_cost_matrix(idle_explorer_ids, candidate_target_ids)
        explorer_pairs = MinCostMatcher.solve(explorer_cost)

        assigned_target_by_explorer: Dict[int, int] = {}
        accepted_targets: List[int] = []
        max_cost = 1e6
        for row, col in explorer_pairs:
            if row >= len(idle_explorer_ids) or col >= len(candidate_target_ids):
                continue
            if float(explorer_cost[row, col]) >= max_cost:
                continue
            eid = idle_explorer_ids[row]
            tid = candidate_target_ids[col]
            assigned_target_by_explorer[eid] = tid
            accepted_targets.append(tid)

        if len(accepted_targets) == 0:
            return []

        expanded_slots: List[int] = []
        for tid in accepted_targets:
            req = int(np.clip(self.targets[tid].required_hunters, 1, 5))
            expanded_slots.extend([tid] * req)

        if len(expanded_slots) == 0:
            return []

        hunter_cost = self._build_hunter_cost_matrix(idle_hunter_ids, expanded_slots)
        hunter_pairs = MinCostMatcher.solve(hunter_cost)

        target_hunter_map: Dict[int, List[int]] = {tid: [] for tid in accepted_targets}
        for row, col in hunter_pairs:
            if row >= len(idle_hunter_ids) or col >= len(expanded_slots):
                continue
            if float(hunter_cost[row, col]) >= max_cost:
                continue
            hid = idle_hunter_ids[row]
            tid = expanded_slots[col]
            target_hunter_map[tid].append(hid)

        assignments: List[Tuple[int, int, List[int]]] = []
        for eid, tid in assigned_target_by_explorer.items():
            need = int(np.clip(self.targets[tid].required_hunters, 1, 5))
            picked_hunters = target_hunter_map.get(tid, [])
            if len(picked_hunters) < need:
                continue
            use_hunters = picked_hunters[:need]
            assignments.append((int(eid), int(tid), [int(x) for x in use_hunters]))
        return assignments

    def _apply_assignment(self, assignments: List[Tuple[int, int, List[int]]]):
        """
        功能:
            下发分配方案并更新状态。
        输入:
            assignments (List[Tuple[int,int,List[int]]]): 分配方案。
        输出:
            无。
        """
        if len(assignments) == 0:
            return
        # Close existing pursuit tasks (will be recreated if still assigned)
        for runtime in self.pursuit_tasks.values():
            runtime.env.close()
        self.pursuit_tasks = {}

        prev_pursuit_started = {idx: bool(t.pursuit_started) for idx, t in enumerate(self.targets)}

        new_explorer_target: Dict[int, int] = {}
        new_hunter_target: Dict[int, int] = {}
        new_target_hunters: Dict[int, List[int]] = {}
        for eid, tid, use_hunters in assignments:
            new_explorer_target[int(eid)] = int(tid)
            new_target_hunters[int(tid)] = [int(x) for x in use_hunters]
            for hid in use_hunters:
                new_hunter_target[int(hid)] = int(tid)

        # Reset target assignment fields
        for target in self.targets:
            target.assigned_explorer = -1
            target.assigned_hunters = []
            target.assign_step = -1
            target.pursuit_started = False

        # Update explorers
        for eid, ex in enumerate(self.explorers):
            if int(eid) in new_explorer_target:
                tid = int(new_explorer_target[eid])
                ex.state = "TRACK"
                ex.assigned_target = int(tid)
            else:
                if ex.assigned_target >= 0:
                    ex.state = "RETURN"
                ex.assigned_target = -1

        # Update hunters
        for hid, hunter in enumerate(self.hunters):
            if int(hid) in new_hunter_target:
                tid = int(new_hunter_target[hid])
                hunter.last_target = int(hunter.assigned_target)
                hunter.resume_pos = np.asarray(hunter.agent.position, dtype=np.float32).copy()
                hunter.assigned_target = int(tid)
                hunter.state = "PURSUIT"
            else:
                if hunter.assigned_target >= 0:
                    self._release_hunter(hunter)

        # Apply targets and create pursuit envs
        for tid, hids in new_target_hunters.items():
            if tid < 0 or tid >= len(self.targets):
                continue
            target = self.targets[tid]
            target.assigned_explorer = int(
                next((eid for eid, t in new_explorer_target.items() if int(t) == int(tid)), -1)
            )
            target.assigned_hunters = [int(x) for x in hids]
            target.assign_step = int(self.time_step)
            if bool(prev_pursuit_started.get(int(tid), False)):
                target.pursuit_started = True
            self._create_pursuit_task_env(target_id=int(tid), hunter_ids=hids)

    def _update_pursuit_progress(self):
        """
        功能:
            更新追捕子任务终止、失败回收与资源释放。
        输入:
            无。
        输出:
            无。
        """
        capture_dist = float(self.cfg.env.capture_dis)
        for tid, target in enumerate(self.targets):
            if not target.alive:
                continue
            if target.assigned_explorer < 0:
                continue

            ex = self.explorers[target.assigned_explorer]
            perc = float(getattr(self.cfg.Explorer, "perception_radius", -1))
            perc = float(self.mission.world_size * 2.0) if perc <= 0 else perc
            seen_now = float(np.linalg.norm(ex.agent.position - target.agent.position)) <= perc
            if seen_now:
                target.last_seen_step = int(self.time_step)
                target.last_seen_pos = target.agent.position.copy()
                target.last_seen_vel = target.agent.velocity.copy()

            if (not target.pursuit_started) and (int(self.time_step) - int(target.assign_step) > int(self.mission.loss_timeout_steps)):
                self._abort_target_task(tid)
                continue

            if target.pursuit_started and (int(self.time_step) - int(target.last_seen_step) > int(self.mission.loss_timeout_steps)):
                self._abort_target_task(tid)
                continue

            active_capture = 0
            for hid in list(target.assigned_hunters):
                if hid < 0 or hid >= len(self.hunters):
                    continue
                h = self.hunters[hid]
                dist = float(np.linalg.norm(h.agent.position - target.agent.position))
                if dist <= capture_dist:
                    active_capture += 1

            if active_capture >= int(target.required_hunters):
                target.alive = False
                target.in_pool = False
                target.discovered = False
                self._release_target_resources(tid)

    def _abort_target_task(self, target_id: int):
        """
        功能:
            中止目标子任务并释放资源。
        输入:
            target_id (int): 目标ID。
        输出:
            无。
        """
        if target_id < 0 or target_id >= len(self.targets):
            return
        target = self.targets[target_id]
        target.in_pool = False
        target.discovered = False
        self._release_target_resources(target_id)

    def _release_target_resources(self, target_id: int):
        """
        功能:
            释放目标对应Explorer/Hunter资源并回到待命状态。
        输入:
            target_id (int): 目标ID。
        输出:
            无。
        """
        target = self.targets[target_id]
        if target.assigned_explorer >= 0:
            ex = self.explorers[target.assigned_explorer]
            ex.state = "RETURN"
            ex.assigned_target = -1
            ex.path_index = int(ex.resume_path_index)
        for hid in list(target.assigned_hunters):
            if hid < 0 or hid >= len(self.hunters):
                continue
            self._release_hunter(self.hunters[hid])
        if target_id in self.pursuit_tasks:
            self.pursuit_tasks[target_id].env.close()
            self.pursuit_tasks.pop(target_id, None)
        target.assigned_explorer = -1
        target.assigned_hunters = []
        target.pursuit_started = False

    def _create_pursuit_task_env(self, target_id: int, hunter_ids: List[int]):
        """
        功能:
            为指定目标创建追捕子任务环境，并将全局状态同步为子环境初始状态。
        输入:
            target_id (int): 目标ID。
            hunter_ids (List[int]): 分配到该目标的全局Hunter ID列表。
        输出:
            无。
        """
        if target_id < 0 or target_id >= len(self.targets):
            return
        if len(hunter_ids) <= 0:
            return

        target = self.targets[target_id]
        sub_cfg = copy.deepcopy(self.cfg)
        sub_cfg.env.max_hunters_num = int(len(hunter_ids))
        sub_cfg.env.world_size = float(self.mission.world_size)
        sub_cfg.env.episode_length = int(max(1, self.mission.max_steps))
        sub_cfg.env.target_policy_source = str(target.agent.policy_type).lower()

        sub_env = UAVPursuitEnv(sub_cfg)
        sub_env.seed(int(self.rng.randint(1, 10**9)))
        sub_env.reset(mode="initial")

        init_positions = np.zeros((sub_env.agent_num, 2), dtype=np.float32)
        for local_hid, global_hid in enumerate(hunter_ids):
            init_positions[local_hid] = self.hunters[int(global_hid)].agent.position.copy()
        init_positions[sub_env.target_index] = target.agent.position.copy()
        sub_env._reset_to_positions(init_positions)

        for local_hid, global_hid in enumerate(hunter_ids):
            global_hunter = self.hunters[int(global_hid)].agent
            local_hunter = sub_env.hunters[local_hid]
            local_hunter.velocity = np.asarray(global_hunter.velocity, dtype=np.float32).copy()
            local_hunter.heading = np.asarray(global_hunter.heading, dtype=np.float32).copy()
            local_hunter.alive = bool(global_hunter.alive)
        sub_env.target.velocity = np.asarray(target.agent.velocity, dtype=np.float32).copy()
        sub_env.target.heading = np.asarray(target.agent.heading, dtype=np.float32).copy()
        sub_env.target.alive = bool(target.agent.alive)
        sub_env.target.policy_type = str(target.agent.policy_type).lower()
        if str(sub_env.target.policy_type).lower() == "patrol":
            sub_env.target.patrol_routes = list(target.agent.patrol_routes)
            sub_env.target.patrol_waypoints = list(target.agent.patrol_waypoints)
            sub_env.target.route_index = int(target.agent.route_index)
            sub_env.target.patrol_index = int(target.agent.patrol_index)
            sub_env.target.route_episode_count = int(target.agent.route_episode_count)

        if target_id in self.pursuit_tasks:
            self.pursuit_tasks[target_id].env.close()
        self.pursuit_tasks[target_id] = PursuitRuntime(
            env=sub_env,
            hunter_ids=[int(x) for x in hunter_ids],
        )
        if self.hunter_actor is not None and bool(self.hunter_actor.actor_path):
            team_sees_target = bool(sub_env._team_sees_target())
            obs_all = sub_env._build_obs(team_sees_target=team_sees_target)
            if obs_all is not None and len(obs_all) > 0:
                self.hunter_actor._ensure_policy(int(np.asarray(obs_all[0]).shape[0]))
            for hid in hunter_ids:
                self.hunter_actor.reset_hunter(int(hid))

    def _sync_target_task_from_subenv(self, target_id: int):
        """
        功能:
            将追捕子环境中的Hunter/Target状态回写到全局仿真。
        输入:
            target_id (int): 目标ID。
        输出:
            无。
        """
        runtime = self.pursuit_tasks.get(target_id, None)
        if runtime is None:
            return
        env = runtime.env

        for local_hid, global_hid in enumerate(runtime.hunter_ids):
            if global_hid < 0 or global_hid >= len(self.hunters):
                continue
            local_h = env.hunters[local_hid]
            global_runtime = self.hunters[global_hid]
            global_h = global_runtime.agent
            global_h.position = np.asarray(local_h.position, dtype=np.float32).copy()
            global_h.velocity = np.asarray(local_h.velocity, dtype=np.float32).copy()
            global_h.heading = np.asarray(local_h.heading, dtype=np.float32).copy()
            global_h.alive = bool(local_h.alive)
            global_h.trajectory.append(global_h.position.copy())
            self._consume_endurance(global_runtime, speed=float(np.linalg.norm(global_h.velocity)))
            if global_runtime.remaining_endurance <= 0.0:
                global_runtime.state = "EXHAUSTED"
                global_runtime.assigned_target = -1
                global_h.alive = False

        target = self.targets[target_id]
        global_t = target.agent
        local_t = env.target
        global_t.position = np.asarray(local_t.position, dtype=np.float32).copy()
        global_t.velocity = np.asarray(local_t.velocity, dtype=np.float32).copy()
        global_t.heading = np.asarray(local_t.heading, dtype=np.float32).copy()
        global_t.alive = bool(local_t.alive)
        global_t.trajectory.append(global_t.position.copy())
        if str(global_t.policy_type).lower() == "patrol":
            global_t.patrol_index = int(local_t.patrol_index)
            global_t.route_index = int(local_t.route_index)
            global_t.route_episode_count = int(local_t.route_episode_count)

        if not bool(local_t.alive):
            target.alive = False
            target.in_pool = False
            target.discovered = False
            self._release_target_resources(target_id)

    def _release_hunter(self, hunter: HunterRuntime):
        """
        功能:
            释放单个Hunter到待命状态。
        输入:
            hunter (HunterRuntime): Hunter运行态。
        输出:
            无。
        """
        hunter.last_target = int(hunter.assigned_target)
        hunter.assigned_target = -1
        hunter.state = "RETURN"
        hunter.agent.velocity[:] = 0.0
        hunter.pursuit_traj_start = -1

    def _move_along_path(self, agent: ExplorerAgent, path: List[np.ndarray], runtime: ExplorerRuntime):
        """
        功能:
            按规划航点推进Explorer（简化运动，不启用动力学约束）。
        输入:
            agent (ExplorerAgent): Explorer对象。
            path (List[np.ndarray]): 航线航点。
            runtime (ExplorerRuntime): 运行状态。
        输出:
            无。
        """
        if len(path) == 0:
            return
        idx = int(runtime.path_index) % len(path)
        target = path[idx]
        speed = float(agent.max_speed)
        arrived = self._move_towards(agent, target, speed)
        if arrived:
            runtime.path_index = (idx + 1) % len(path)
            if runtime.state == "SEARCH":
                runtime.resume_path_index = int(runtime.path_index)

    def _move_hunter_split_standby(self, runtime: HunterRuntime):
        """
        功能:
            按split待命航线推进Hunter。
        输入:
            runtime (HunterRuntime): Hunter运行状态。
        输出:
            无。
        """
        if len(runtime.standby_path) == 0:
            return
        idx = int(runtime.standby_index)
        target = runtime.standby_path[idx]
        speed = float(max(0.0, runtime.standby_speed))
        arrived = self._move_towards(runtime.agent, target, speed=speed)
        if not arrived:
            return
        if idx <= 0:
            runtime.standby_direction = 1
        elif idx >= len(runtime.standby_path) - 1:
            runtime.standby_direction = -1
        runtime.standby_index = int(np.clip(idx + runtime.standby_direction, 0, len(runtime.standby_path) - 1))

    def _move_hunter_zone_standby(self, runtime: HunterRuntime):
        """
        功能:
            按zone待命模式跟随对应Explorer。
        输入:
            runtime (HunterRuntime): Hunter运行状态。
        输出:
            无。
        """
        if runtime.zone_explorer < 0 or runtime.zone_explorer >= len(self.explorers):
            return
        anchor = self.explorers[runtime.zone_explorer].agent.position + runtime.zone_offset
        runtime.agent.position = np.clip(anchor, -self.mission.world_size, self.mission.world_size).astype(np.float32)
        runtime.agent.velocity[:] = 0.0
        runtime.agent.trajectory.append(runtime.agent.position.copy())

    def _move_towards(self, agent, target_pos: np.ndarray, speed: float) -> bool:
        """
        功能:
            以给定速度将agent推进至目标点。
        输入:
            agent (BaseAgent): Agent对象。
            target_pos (np.ndarray): 目标位置 shape=(2,)。
            speed (float): 速度上限（米/秒）。
        输出:
            bool: 是否到达目标点。
        """
        dt = float(self.mission.dt)
        vec = np.asarray(target_pos, dtype=np.float32) - np.asarray(agent.position, dtype=np.float32)
        dist = float(np.linalg.norm(vec))
        if dist <= 1e-6:
            agent.velocity[:] = 0.0
            agent.trajectory.append(agent.position.copy())
            return True
        step_dist = float(max(0.0, speed) * dt)
        if step_dist >= dist:
            new_pos = np.asarray(target_pos, dtype=np.float32)
            agent.velocity = (vec / max(dt, 1e-6)).astype(np.float32)
            agent.position = np.clip(new_pos, -self.mission.world_size, self.mission.world_size).astype(np.float32)
            agent.trajectory.append(agent.position.copy())
            return True
        direction = vec / dist
        agent.velocity = (direction * float(max(0.0, speed))).astype(np.float32)
        agent.position = np.clip(
            agent.position + agent.velocity * dt,
            -self.mission.world_size,
            self.mission.world_size,
        ).astype(np.float32)
        agent.trajectory.append(agent.position.copy())
        return False

    def _start_pursuit_for_target(self, target_id: int):
        """
        功能:
            标记追捕开始并记录追捕轨迹起点。
        输入:
            target_id (int): 目标ID。
        输出:
            无。
        """
        if target_id < 0 or target_id >= len(self.targets):
            return
        target = self.targets[target_id]
        for hid in list(target.assigned_hunters):
            if hid < 0 or hid >= len(self.hunters):
                continue
            hunter = self.hunters[hid]
            if len(hunter.agent.trajectory) == 0:
                hunter.agent.trajectory.append(hunter.agent.position.copy())
            hunter.pursuit_traj_start = int(max(0, len(hunter.agent.trajectory) - 1))

    def _build_hunter_chase_action(self, hunter: HunterAgent, target_pos: np.ndarray) -> np.ndarray:
        """
        功能:
            根据目标位置构造Hunter追捕动作（兼容velocity/acceleration）。
        输入:
            hunter (HunterAgent): Hunter对象。
            target_pos (np.ndarray): 目标位置。
        输出:
            np.ndarray: shape=(2,) 归一化动作。
        """
        vec = np.asarray(target_pos, dtype=np.float32) - np.asarray(hunter.position, dtype=np.float32)
        dist = float(np.linalg.norm(vec))
        if dist <= 1e-6:
            return np.zeros(2, dtype=np.float32)
        direction = vec / dist
        if str(hunter.control_mode).lower() == "acceleration":
            desired_vel = direction * float(hunter.max_speed)
            acc = desired_vel - np.asarray(hunter.velocity, dtype=np.float32)
            if float(hunter.max_acc) <= 1e-8:
                return np.zeros(2, dtype=np.float32)
            return np.clip(acc / float(hunter.max_acc), -1.0, 1.0).astype(np.float32)
        return np.clip(direction, -1.0, 1.0).astype(np.float32)

    def _build_explorer_search_paths(self, world_size: float, num_explorers: int, overlap_rate: float) -> List[List[np.ndarray]]:
        """
        功能:
            生成弓字覆盖航线并切分为M条子航线。
        输入:
            world_size (float): 地图半边长。
            num_explorers (int): Explorer数量。
            overlap_rate (float): 感知重叠率。
        输出:
            List[List[np.ndarray]]: 每个Explorer对应航线。
        """
        perc = float(getattr(self.cfg.Explorer, "perception_radius", -1))
        if perc <= 0:
            perc = float(max(5.0, self.cfg.Explorer.safe_dis))
        spacing = float(max(2.0, 2.0 * perc - perc * float(np.clip(overlap_rate, 0.0, 0.95))))
        margin = float(max(2.0, perc * 0.5))
        y_vals = []
        y = float(world_size - margin)
        while y >= -float(world_size - margin):
            y_vals.append(y)
            y -= spacing
        if len(y_vals) == 0:
            y_vals = [0.0]

        x_min = -float(world_size - margin)
        x_max = float(world_size - margin)
        full_route: List[np.ndarray] = []
        left_to_right = True
        for yy in y_vals:
            if left_to_right:
                full_route.append(np.array([x_min, yy], dtype=np.float32))
                full_route.append(np.array([x_max, yy], dtype=np.float32))
            else:
                full_route.append(np.array([x_max, yy], dtype=np.float32))
                full_route.append(np.array([x_min, yy], dtype=np.float32))
            left_to_right = not left_to_right

        # 先规划单Explorer全局覆盖航线，再按Explorer数量做连续等分。
        paths: List[List[np.ndarray]] = []
        total_wp = len(full_route)
        if total_wp <= 0:
            return [[self._sample_position()] for _ in range(max(1, num_explorers))]

        boundaries = [int(math.floor(i * total_wp / max(1, num_explorers))) for i in range(num_explorers)]
        boundaries.append(total_wp)
        for eid in range(num_explorers):
            start = int(boundaries[eid])
            end = int(boundaries[eid + 1])
            indices = list(range(start, end))
            # 连接相邻Explorer的航线端点，确保覆盖路径连续。
            if end < total_wp:
                indices.append(int(end))
            if len(indices) == 0:
                idx = int(min(start, total_wp - 1))
                indices = [idx]
                if idx + 1 < total_wp:
                    indices.append(idx + 1)
            chunk = [full_route[i].copy() for i in indices]
            if len(chunk) == 0:
                chunk = [self._sample_position()]

            # 为避免段尾回环产生斜线，构建往返序列。
            if len(chunk) >= 2:
                bounce = [wp.copy() for wp in chunk[-2:0:-1]]
                chunk = chunk + bounce
            paths.append(chunk)
        return paths

    def _build_hunter_split_paths(self, world_size: float, num_hunters: int) -> List[List[np.ndarray]]:
        """
        功能:
            构建split模式下Hunter纵向待命航线。
        输入:
            world_size (float): 地图半边长。
            num_hunters (int): Hunter数量。
        输出:
            List[List[np.ndarray]]: 每个Hunter的待命航线。
        """
        margin = max(2.0, world_size * 0.05)
        xs = np.linspace(-world_size + margin, world_size - margin, num=max(1, num_hunters)).astype(np.float32)
        y1 = -world_size + margin
        y2 = world_size - margin
        out = []
        for hid, xx in enumerate(xs):
            # 相邻航线起点交错：偶数从底部出发，奇数从顶部出发。
            if int(hid) % 2 == 0:
                out.append([
                    np.array([xx, y1], dtype=np.float32),
                    np.array([xx, y2], dtype=np.float32),
                ])
            else:
                out.append([
                    np.array([xx, y2], dtype=np.float32),
                    np.array([xx, y1], dtype=np.float32),
                ])
        return out

    def _assign_zone_groups(self):
        """
        功能:
            构建zone模式下Hunter编组与方阵偏移。
        输入:
            无。
        输出:
            无。
        """
        for hid, runtime in enumerate(self.hunters):
            eid = int(hid % max(1, len(self.explorers)))
            runtime.standby_mode = "zone"
            runtime.zone_explorer = eid

        group_map: Dict[int, List[int]] = {}
        for hid, runtime in enumerate(self.hunters):
            group_map.setdefault(runtime.zone_explorer, []).append(hid)

        spacing = max(2.0, float(self.cfg.Hunter.safe_dis) * 0.5)
        for eid, member_ids in group_map.items():
            n = len(member_ids)
            side = int(math.ceil(math.sqrt(max(1, n))))
            slots = []
            for r in range(side):
                for c in range(side):
                    x = (c - (side - 1) / 2.0) * spacing
                    y = -(r + 1) * spacing
                    slots.append(np.array([x, y], dtype=np.float32))
            self.rng.shuffle(slots)
            for i, hid in enumerate(member_ids):
                runtime = self.hunters[hid]
                runtime.zone_offset = slots[i].copy()
                anchor = self.explorers[eid].agent.position + runtime.zone_offset
                runtime.agent.reset(np.clip(anchor, -self.mission.world_size, self.mission.world_size).astype(np.float32))

    def _build_explorer_cost_matrix(self, explorer_ids: List[int], target_ids: List[int]) -> np.ndarray:
        """
        功能:
            构建Explorer-Target匹配代价矩阵。
        输入:
            explorer_ids (List[int]): 空闲Explorer列表。
            target_ids (List[int]): 待分配Target列表。
        输出:
            np.ndarray: shape=(E,T) 代价矩阵。
        """
        high = 1e6
        mat = np.full((len(explorer_ids), len(target_ids)), high, dtype=np.float32)
        for i, eid in enumerate(explorer_ids):
            ex = self.explorers[eid]
            for j, tid in enumerate(target_ids):
                tgt = self.targets[tid]
                if not tgt.alive:
                    continue
                dist = float(np.linalg.norm(ex.agent.position - tgt.last_seen_pos))
                if dist > float(self.weights.max_assign_dist):
                    continue
                value_term = (10.0 - float(tgt.value))
                endurance_term = 0.0
                if int(ex.assigned_target) < 0 or int(ex.assigned_target) == int(tid):
                    switch_term = 0.0
                else:
                    switch_term = 1.0
                cost = (
                    float(self.weights.distance_weight) * dist
                    + float(self.weights.value_weight) * value_term
                    + float(self.weights.endurance_weight) * endurance_term
                    + float(self.weights.switch_weight) * switch_term
                )
                mat[i, j] = float(cost)
        return mat

    def _build_hunter_cost_matrix(self, hunter_ids: List[int], target_slots: List[int]) -> np.ndarray:
        """
        功能:
            构建Hunter-扩展Target槽位匹配代价矩阵。
        输入:
            hunter_ids (List[int]): 空闲Hunter列表。
            target_slots (List[int]): 扩展后的Target槽位列表。
        输出:
            np.ndarray: shape=(H,S) 代价矩阵。
        """
        high = 1e6
        mat = np.full((len(hunter_ids), len(target_slots)), high, dtype=np.float32)
        for i, hid in enumerate(hunter_ids):
            hunter = self.hunters[hid]
            for j, tid in enumerate(target_slots):
                tgt = self.targets[tid]
                if not tgt.alive:
                    continue

                # 距离越远，代价越大
                dist = float(np.linalg.norm(hunter.agent.position - tgt.last_seen_pos))
                if dist > float(self.weights.max_assign_dist):
                    continue

                # 处理优先级越高，代价越低
                value_term = (10.0 - float(tgt.value))

                
                endurance_term = 0.0
                if int(hunter.assigned_target) < 0 or int(hunter.assigned_target) == int(tid):
                    switch_term = 0.0
                else:
                    switch_term = 1.0
                cost = (
                    float(self.weights.distance_weight) * dist
                    + float(self.weights.value_weight) * value_term
                    + float(self.weights.endurance_weight) * endurance_term
                    + float(self.weights.switch_weight) * switch_term
                )
                mat[i, j] = float(cost)
        return mat

    def _active_hunters_for_target(self, target_id: int) -> Tuple[List[HunterAgent], np.ndarray]:
        """
        功能:
            收集指定目标当前关联的Hunter列表与激活掩码。
        输入:
            target_id (int): 目标ID。
        输出:
            Tuple[List[HunterAgent], np.ndarray]: (Hunter对象列表, 激活掩码)。
        """
        target = self.targets[target_id]
        hunter_objs: List[HunterAgent] = []
        mask: List[bool] = []
        for hid in target.assigned_hunters:
            if hid < 0 or hid >= len(self.hunters):
                continue
            hunter_objs.append(self.hunters[hid].agent)
            mask.append(True)
        return hunter_objs, np.asarray(mask, dtype=bool)

    def _build_target_learn_obs(self, target_id: int, hunters: List[HunterAgent]) -> np.ndarray:
        """
        功能:
            构建learn-target推理观测（简化版本）。
        输入:
            target_id (int): 目标ID。
            hunters (List[HunterAgent]): 当前关联Hunter列表。
        输出:
            np.ndarray: shape=(20,)。
        """
        target = self.targets[target_id].agent
        obs = np.zeros(20, dtype=np.float32)
        ws = float(max(1.0, self.mission.world_size))
        obs[0:2] = target.position / ws
        obs[2:4] = target.velocity / max(1e-6, float(target.max_speed))
        cap = min(4, len(hunters))
        for idx in range(cap):
            rel = hunters[idx].position - target.position
            obs[4 + idx * 4: 6 + idx * 4] = rel / ws
            obs[6 + idx * 4: 8 + idx * 4] = hunters[idx].velocity / max(1e-6, float(hunters[idx].max_speed))
        return obs

    def _load_patrol_routes(self, route_path: str, route_names: List[str]) -> List[List[np.ndarray]]:
        """
        功能:
            从JSON读取巡逻路线并转换为航点列表。
        输入:
            route_path (str): 巡逻文件路径。
            route_names (List[str]): 指定路线名，支持'all'。
        输出:
            List[List[np.ndarray]]: 巡逻路线集合。
        """
        path = Path(route_path)
        if not path.is_absolute():
            path = project_root / path
        routes: List[List[np.ndarray]] = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                routes = self._parse_route_json(data, route_names)
            except Exception:
                routes = []
        if len(routes) > 0:
            return routes
        return [
            [
                np.array([-30.0, -30.0], dtype=np.float32),
                np.array([30.0, -30.0], dtype=np.float32),
                np.array([30.0, 30.0], dtype=np.float32),
                np.array([-30.0, 30.0], dtype=np.float32),
            ]
        ]

    def _parse_route_json(self, data, route_names: List[str]) -> List[List[np.ndarray]]:
        """
        功能:
            解析巡逻文件内容为标准路线列表。
        输入:
            data (Any): JSON加载结果。
            route_names (List[str]): 路线名过滤。
        输出:
            List[List[np.ndarray]]: 路线集合。
        """
        raw_routes: Dict[str, List] = {}
        if isinstance(data, dict):
            if "routes" in data and isinstance(data["routes"], dict):
                raw_routes = dict(data["routes"])
            else:
                for k, v in data.items():
                    if isinstance(v, list):
                        raw_routes[str(k)] = v
        elif isinstance(data, list):
            for idx, route in enumerate(data):
                raw_routes[f"route_{idx}"] = route

        names = [str(x) for x in route_names]
        selected_keys = list(raw_routes.keys()) if "all" in [x.lower() for x in names] else [k for k in raw_routes.keys() if k in names]
        if len(selected_keys) == 0:
            selected_keys = list(raw_routes.keys())

        out: List[List[np.ndarray]] = []
        for key in selected_keys:
            points = raw_routes.get(key, [])
            route: List[np.ndarray] = []
            for point in points:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    route.append(np.array([float(point[0]), float(point[1])], dtype=np.float32))
                elif isinstance(point, dict) and "x" in point and "y" in point:
                    route.append(np.array([float(point["x"]), float(point["y"])], dtype=np.float32))
            if len(route) > 0:
                out.append(route)
        return out


class SwarmSimGUI:
    """
    功能:
        提供任务控制、参数配置与可视化界面。
    输入:
        sim (SwarmSimulationCore): 仿真核心实例。
    输出:
        无。
    """

    def __init__(self, sim: SwarmSimulationCore, sim_config_path: Optional[Path] = None, model_config_path: Optional[Path] = None):
        if tk is None:
            raise RuntimeError("tkinter is unavailable in current environment")
        self.sim = sim
        self.sim_config_path = sim_config_path
        self.model_config_path = model_config_path
        self.model_cfg = {"hunters": {}, "targets": {}}
        self.root = tk.Tk()
        self.root.title("Swarm Search + Pursuit Simulator")
        self.running = False
        self.use_cjk_font = self._configure_matplotlib_font()
        self._load_model_config()

        self._build_ui()
        self._refresh_model_dropdowns()
        self._schedule_loop()

    def _configure_matplotlib_font(self) -> bool:
        """
        功能:
            配置Matplotlib中文字体，返回是否找到可用CJK字体。
        输入:
            无。
        输出:
            bool: 是否配置成功。
        """
        candidates = [
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "WenQuanYi Micro Hei",
            "Microsoft YaHei",
            "SimHei",
            "PingFang SC",
            "Heiti SC",
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["axes.unicode_minus"] = False
                return True
        return False

    def _t(self, cn_text: str, en_text: str) -> str:
        """
        功能:
            根据字体配置返回中文/英文显示文本。
        输入:
            cn_text (str): 中文文本。
            en_text (str): 英文文本。
        输出:
            str: 选择后的文本。
        """
        return cn_text if self.use_cjk_font else en_text

    def _get_explorer_phase(self, ex: ExplorerRuntime) -> str:
        """
        功能:
            获取Explorer视觉状态（空闲/预备/追捕/耗尽）。
        输入:
            ex (ExplorerRuntime): Explorer运行态。
        输出:
            str: 状态标签。
        """
        if float(ex.remaining_endurance) <= 0.0 or (not bool(ex.agent.alive)):
            return "EXH"
        if int(ex.assigned_target) < 0:
            return "IDLE"
        tid = int(ex.assigned_target)
        if 0 <= tid < len(self.sim.targets) and bool(self.sim.targets[tid].pursuit_started):
            return "PUR"
        return "PREP"

    def _get_hunter_phase(self, hunter: HunterRuntime) -> str:
        """
        功能:
            获取Hunter视觉状态（空闲/预备/追捕/耗尽）。
        输入:
            hunter (HunterRuntime): Hunter运行态。
        输出:
            str: 状态标签。
        """
        if float(hunter.remaining_endurance) <= 0.0 or str(hunter.state).upper() == "EXHAUSTED":
            return "EXH"
        if int(hunter.assigned_target) < 0:
            return "IDLE"
        tid = int(hunter.assigned_target)
        if 0 <= tid < len(self.sim.targets) and bool(self.sim.targets[tid].pursuit_started):
            return "PUR"
        return "PREP"

    def _build_ui(self):
        """
        功能:
            构建GUI布局、参数控件与绘图区域。
        输入:
            无。
        输出:
            无。
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        left = ttk.Frame(self.root, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self.root, padding=8)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.inputs: Dict[str, tk.StringVar] = {}

        notebook = ttk.Notebook(left)
        notebook.pack(fill=tk.X, pady=(0, 6))
        env_tab = ttk.Frame(notebook, padding=4)
        plan_tab = ttk.Frame(notebook, padding=4)
        assign_tab = ttk.Frame(notebook, padding=4)
        pursuit_tab = ttk.Frame(notebook, padding=4)
        notebook.add(env_tab, text="环境配置")
        notebook.add(plan_tab, text="预规划")
        notebook.add(assign_tab, text="任务分配")
        notebook.add(pursuit_tab, text="目标追捕")

        self._add_input(env_tab, "world_size", str(self.sim.mission.world_size))
        self._add_input(env_tab, "hunters", str(self.sim.mission.hunters))
        self._add_input(env_tab, "explorers", str(self.sim.mission.explorers))
        self._add_input(env_tab, "targets", str(self.sim.mission.targets))
        self._add_input(env_tab, "max_steps", str(self.sim.mission.max_steps))
        self._add_input(env_tab, "dt", str(self.sim.mission.dt))
        self._add_input(env_tab, "seed", str(self.sim.base_seed))
        self._add_input(env_tab, "explorer_max_speed", str(self.sim.mission.explorer_max_speed))
        self._add_input(env_tab, "hunter_max_speed", str(self.sim.mission.hunter_max_speed))
        self._add_input(env_tab, "target_max_speed", str(self.sim.mission.target_max_speed))
        self._add_select(env_tab, "plan_mode(in-loop/on-loop)", ["in-loop", "on-loop"], "on-loop")
        self._add_select(env_tab, "assign_mode(in-loop/on-loop)", ["in-loop", "on-loop"], "on-loop")

        self._add_input(plan_tab, "overlap_rate", str(self.sim.mission.overlap_rate))
        self._add_input(plan_tab, "wait_mode(split/zone)", str(self.sim.mission.hunters_wait_mode))
        self._add_input(plan_tab, "track_speed_scale", str(self.sim.mission.explorer_track_speed_scale))
        self._add_input(plan_tab, "loss_timeout", str(self.sim.mission.loss_timeout_steps))

        self._add_input(assign_tab, "w_distance", str(self.sim.weights.distance_weight))
        self._add_input(assign_tab, "w_value", str(self.sim.weights.value_weight))
        self._add_input(assign_tab, "w_endurance", str(self.sim.weights.endurance_weight))
        self._add_input(assign_tab, "w_switch", str(self.sim.weights.switch_weight))
        self._add_input(assign_tab, "max_assign_dist", str(self.sim.weights.max_assign_dist))
        self._add_input(assign_tab, "explorer_endurance", str(self.sim.mission.explorer_total_endurance))
        self._add_input(assign_tab, "hunter_endurance", str(self.sim.mission.hunter_total_endurance))
        self._add_input(assign_tab, "idle_endurance_cost", str(self.sim.mission.endurance_idle_cost))

        # pursuit tab: model selectors and management buttons
        self._add_model_selectors(pursuit_tab)

        env_buttons = ttk.Frame(left)
        env_buttons.pack(fill=tk.X, pady=(8, 4))
        ttk.Button(env_buttons, text="重置", command=self._on_env_reset).pack(side=tk.LEFT, padx=2)
        ttk.Button(env_buttons, text="随机", command=self._on_env_random_reset).pack(side=tk.LEFT, padx=2)
        ttk.Button(env_buttons, text="暂停", command=self._on_pause).pack(side=tk.LEFT, padx=2)
        ttk.Button(env_buttons, text="开始", command=self._on_start).pack(side=tk.LEFT, padx=2)

        plan_buttons = ttk.Frame(left)
        plan_buttons.pack(fill=tk.X, pady=(4, 4))
        ttk.Button(plan_buttons, text="预规划", command=self._on_plan).pack(side=tk.LEFT, padx=2)
        ttk.Button(plan_buttons, text="下发", command=self._on_dispatch).pack(side=tk.LEFT, padx=2)

        assign_buttons = ttk.Frame(left)
        assign_buttons.pack(fill=tk.X, pady=(4, 4))
        ttk.Button(assign_buttons, text="任务分配", command=self._on_manual_assignment).pack(side=tk.LEFT, padx=2)
        ttk.Button(assign_buttons, text="下发分配方案", command=self._on_apply_assignment).pack(side=tk.LEFT, padx=2)
        ttk.Button(assign_buttons, text="保存倾向", command=self._on_save_profile).pack(side=tk.LEFT, padx=2)
        ttk.Button(assign_buttons, text="加载倾向", command=self._on_load_profile).pack(side=tk.LEFT, padx=2)

        save_buttons = ttk.Frame(left)
        save_buttons.pack(fill=tk.X, pady=(4, 4))
        ttk.Button(save_buttons, text="保存环境配置", command=self._on_save_env_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(save_buttons, text="保存预规划", command=self._on_save_plan_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(save_buttons, text="保存任务分配", command=self._on_save_assign_config).pack(side=tk.LEFT, padx=2)

        row_vis = ttk.Frame(left)
        row_vis.pack(fill=tk.X, pady=(4, 4))
        self.show_explorer_perception_var = tk.BooleanVar(value=True)
        self.show_target_capture_var = tk.BooleanVar(value=True)
        self.show_all_targets_var = tk.BooleanVar(value=False)
        self.show_pursuit_trace_var = tk.BooleanVar(value=False)
        self.show_assignment_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row_vis,
            text="显示Explorer感知圈",
            variable=self.show_explorer_perception_var,
            command=self._draw_scene,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(
            row_vis,
            text="显示Target捕获圈",
            variable=self.show_target_capture_var,
            command=self._draw_scene,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(
            row_vis,
            text="显示全部目标",
            variable=self.show_all_targets_var,
            command=self._draw_scene,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(
            row_vis,
            text="显示追捕轨迹",
            variable=self.show_pursuit_trace_var,
            command=self._draw_scene,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(
            row_vis,
            text="显示分配结果",
            variable=self.show_assignment_var,
            command=self._draw_scene,
        ).pack(side=tk.LEFT, padx=2)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status_var, wraplength=320).pack(fill=tk.X, pady=(10, 0))

        ttk.Label(left, text="任务池 / 资源池", anchor="w").pack(fill=tk.X, pady=(8, 2))
        self.pool_text = tk.Text(left, width=48, height=20)
        self.pool_text.pack(fill=tk.BOTH, expand=False)
        self.pool_text.configure(state=tk.DISABLED)

        self.figure = Figure(figsize=(8, 8), dpi=100)
        grid = self.figure.add_gridspec(1, 2, width_ratios=[3.4, 1.4])
        self.ax = self.figure.add_subplot(grid[0, 0])
        self.ax_pool = self.figure.add_subplot(grid[0, 1])
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_scene()

    def _add_input(self, parent, key: str, default: str):
        """
        功能:
            添加一行参数输入控件。
        输入:
            parent (ttk.Frame): 父容器。
            key (str): 参数名。
            default (str): 默认值。
        输出:
            无。
        """
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=1)
        ttk.Label(frame, text=key, width=18).pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        self.inputs[key] = var
        ttk.Entry(frame, textvariable=var, width=16).pack(side=tk.RIGHT)

    def _add_select(self, parent, key: str, options: List[str], default: str):
        """
        功能:
            添加下拉选择控件。
        输入:
            parent (ttk.Frame): 父容器。
            key (str): 参数名。
            options (List[str]): 选项列表。
            default (str): 默认值。
        输出:
            无。
        """
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=1)
        ttk.Label(frame, text=key, width=18).pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        self.inputs[key] = var
        combo = ttk.Combobox(frame, textvariable=var, values=options, state="readonly", width=14)
        combo.pack(side=tk.RIGHT)

    def _add_model_selectors(self, parent):
        """
        功能:
            构建追捕模型选择与管理控件。
        输入:
            parent (ttk.Frame): 父容器。
        输出:
            无。
        """
        ttk.Label(parent, text="Hunter模型", anchor="w").pack(fill=tk.X, pady=(2, 2))
        hunter_row = ttk.Frame(parent)
        hunter_row.pack(fill=tk.X, pady=1)
        self.hunter_model_var = tk.StringVar()
        self.hunter_combo = ttk.Combobox(hunter_row, textvariable=self.hunter_model_var, state="readonly", width=24)
        self.hunter_combo.pack(side=tk.LEFT, padx=2)
        ttk.Button(hunter_row, text="新增", command=lambda: self._on_add_model("hunters")).pack(side=tk.LEFT, padx=2)
        ttk.Button(hunter_row, text="删除", command=lambda: self._on_delete_model("hunters")).pack(side=tk.LEFT, padx=2)
        ttk.Button(hunter_row, text="重命名", command=lambda: self._on_rename_model("hunters")).pack(side=tk.LEFT, padx=2)

        ttk.Label(parent, text="Target模型", anchor="w").pack(fill=tk.X, pady=(6, 2))
        target_row = ttk.Frame(parent)
        target_row.pack(fill=tk.X, pady=1)
        self.target_model_var = tk.StringVar()
        self.target_combo = ttk.Combobox(target_row, textvariable=self.target_model_var, state="readonly", width=24)
        self.target_combo.pack(side=tk.LEFT, padx=2)
        ttk.Button(target_row, text="新增", command=lambda: self._on_add_model("targets")).pack(side=tk.LEFT, padx=2)
        ttk.Button(target_row, text="删除", command=lambda: self._on_delete_model("targets")).pack(side=tk.LEFT, padx=2)
        ttk.Button(target_row, text="重命名", command=lambda: self._on_rename_model("targets")).pack(side=tk.LEFT, padx=2)

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill=tk.X, pady=(6, 2))
        ttk.Button(btn_row, text="批量加载", command=self._on_batch_load_models).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="下发追捕模型", command=self._on_apply_pursuit_models).pack(side=tk.LEFT, padx=2)

    def _read_mission_and_weights(self) -> Tuple[MissionConfig, AssignmentWeights]:
        """
        功能:
            从输入框读取任务参数与权重参数。
        输入:
            无。
        输出:
            Tuple[MissionConfig, AssignmentWeights]: 配置对象。
        """
        mission = MissionConfig(
            world_size=float(self.inputs["world_size"].get()),
            dt=float(self.inputs["dt"].get()),
            max_steps=int(float(self.inputs["max_steps"].get())),
            hunters=max(1, int(float(self.inputs["hunters"].get()))),
            explorers=max(1, int(float(self.inputs["explorers"].get()))),
            targets=max(1, int(float(self.inputs["targets"].get()))),
            explorer_max_speed=float(self.inputs["explorer_max_speed"].get()),
            hunter_max_speed=float(self.inputs["hunter_max_speed"].get()),
            target_max_speed=float(self.inputs["target_max_speed"].get()),
            overlap_rate=float(self.inputs["overlap_rate"].get()),
            hunters_wait_mode=str(self.inputs["wait_mode(split/zone)"].get()).strip().lower(),
            explorer_track_speed_scale=float(self.inputs["track_speed_scale"].get()),
            loss_timeout_steps=max(1, int(float(self.inputs["loss_timeout"].get()))),
            explorer_total_endurance=float(self.inputs["explorer_endurance"].get()),
            hunter_total_endurance=float(self.inputs["hunter_endurance"].get()),
            endurance_idle_cost=float(self.inputs["idle_endurance_cost"].get()),
        )
        if mission.hunters_wait_mode not in ("split", "zone"):
            mission.hunters_wait_mode = "split"

        weights = AssignmentWeights(
            distance_weight=float(self.inputs["w_distance"].get()),
            value_weight=float(self.inputs["w_value"].get()),
            endurance_weight=float(self.inputs["w_endurance"].get()),
            switch_weight=float(self.inputs["w_switch"].get()),
            max_assign_dist=float(self.inputs["max_assign_dist"].get()),
        )
        return mission, weights

    def _read_modes_and_seed(self) -> Tuple[str, str, int]:
        """
        功能:
            读取预规划/分配模式与随机种子。
        输入:
            无。
        输出:
            Tuple[str,str,int]: (plan_mode, assign_mode, seed)
        """
        plan_mode = str(self.inputs["plan_mode(in-loop/on-loop)"].get()).strip().lower()
        assign_mode = str(self.inputs["assign_mode(in-loop/on-loop)"].get()).strip().lower()
        if plan_mode not in ("in-loop", "on-loop"):
            plan_mode = "on-loop"
        if assign_mode not in ("in-loop", "on-loop"):
            assign_mode = "on-loop"
        seed = int(float(self.inputs["seed"].get()))
        return plan_mode, assign_mode, seed

    def _read_env_config(self) -> dict:
        """
        功能:
            读取环境配置字段并返回swarm_sim.mission配置段。
        输入:
            无。
        输出:
            dict: mission配置字段。
        """
        mission, _ = self._read_mission_and_weights()
        return {
            "world_size": float(mission.world_size),
            "dt": float(mission.dt),
            "max_steps": int(mission.max_steps),
            "hunters": int(mission.hunters),
            "explorers": int(mission.explorers),
            "targets": int(mission.targets),
            "explorer_max_speed": float(mission.explorer_max_speed),
            "hunter_max_speed": float(mission.hunter_max_speed),
            "target_max_speed": float(mission.target_max_speed),
            "explorer_total_endurance": float(mission.explorer_total_endurance),
            "hunter_total_endurance": float(mission.hunter_total_endurance),
            "endurance_idle_cost": float(mission.endurance_idle_cost),
        }

    def _read_plan_config(self) -> dict:
        """
        功能:
            读取预规划配置字段并返回swarm_sim.mission配置段。
        输入:
            无。
        输出:
            dict: mission配置字段。
        """
        mission, _ = self._read_mission_and_weights()
        return {
            "overlap_rate": float(mission.overlap_rate),
            "hunters_wait_mode": str(mission.hunters_wait_mode).lower(),
            "explorer_track_speed_scale": float(mission.explorer_track_speed_scale),
            "loss_timeout_steps": int(mission.loss_timeout_steps),
        }

    def _read_assign_config(self) -> dict:
        """
        功能:
            读取任务分配配置字段并返回swarm_sim.assignment配置段。
        输入:
            无。
        输出:
            dict: assignment配置字段。
        """
        _, weights = self._read_mission_and_weights()
        return {
            "distance_weight": float(weights.distance_weight),
            "value_weight": float(weights.value_weight),
            "endurance_weight": float(weights.endurance_weight),
            "switch_weight": float(weights.switch_weight),
            "max_assign_dist": float(weights.max_assign_dist),
        }

    def _save_sim_config_section(self, mission_updates: Optional[dict] = None, assignment_updates: Optional[dict] = None):
        """
        功能:
            将配置更新写入 --sim_config_file。
        输入:
            mission_updates (Optional[dict]): mission更新字段。
            assignment_updates (Optional[dict]): assignment更新字段。
        输出:
            无。
        """
        if self.sim_config_path is None:
            if messagebox is not None:
                messagebox.showerror("保存失败", "未指定 --sim_config_file")
            return
        cfg_path = Path(self.sim_config_path)
        payload = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                payload = yaml.safe_load(f) or {}
        if not isinstance(payload, dict):
            payload = {}
        swarm_sim = payload.get("swarm_sim", {})
        if not isinstance(swarm_sim, dict):
            swarm_sim = {}
        mission_cfg = swarm_sim.get("mission", {})
        if not isinstance(mission_cfg, dict):
            mission_cfg = {}
        assignment_cfg = swarm_sim.get("assignment", {})
        if not isinstance(assignment_cfg, dict):
            assignment_cfg = {}
        if isinstance(mission_updates, dict):
            mission_cfg.update(mission_updates)
        if isinstance(assignment_updates, dict):
            assignment_cfg.update(assignment_updates)
        swarm_sim["mission"] = mission_cfg
        swarm_sim["assignment"] = assignment_cfg
        payload["swarm_sim"] = swarm_sim
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

    def _on_save_env_config(self):
        """
        功能:
            保存环境配置到sim_config文件。
        输入:
            无。
        输出:
            无。
        """
        try:
            mission_updates = self._read_env_config()
            self._save_sim_config_section(mission_updates=mission_updates)
            self.status_var.set("已保存环境配置")
        except Exception as e:
            if messagebox is not None:
                messagebox.showerror("保存失败", str(e))

    def _on_save_plan_config(self):
        """
        功能:
            保存预规划配置到sim_config文件。
        输入:
            无。
        输出:
            无。
        """
        try:
            mission_updates = self._read_plan_config()
            self._save_sim_config_section(mission_updates=mission_updates)
            self.status_var.set("已保存预规划配置")
        except Exception as e:
            if messagebox is not None:
                messagebox.showerror("保存失败", str(e))

    def _on_save_assign_config(self):
        """
        功能:
            保存任务分配配置到sim_config文件。
        输入:
            无。
        输出:
            无。
        """
        try:
            assignment_updates = self._read_assign_config()
            self._save_sim_config_section(assignment_updates=assignment_updates)
            self.status_var.set("已保存任务分配配置")
        except Exception as e:
            if messagebox is not None:
                messagebox.showerror("保存失败", str(e))

    def _apply_modes(self):
        """
        功能:
            将GUI的决策模式同步到仿真核心。
        输入:
            无。
        输出:
            无。
        """
        plan_mode, assign_mode, _ = self._read_modes_and_seed()
        self.plan_mode = plan_mode
        self.sim.assign_mode = assign_mode

    def _on_env_reset(self):
        """
        功能:
            使用固定seed重置环境。
        输入:
            无。
        输出:
            无。
        """
        try:
            mission, weights = self._read_mission_and_weights()
            self._apply_modes()
            _, _, seed = self._read_modes_and_seed()
            self.running = False
            self.sim.mission = mission
            self.sim.weights = weights
            self.sim.base_seed = int(seed)
            self.sim.reset_world_with_seed(int(seed))
            self.status_var.set(f"Reset with seed={int(seed)}")
            self._draw_scene()
        except Exception as e:
            if messagebox is not None:
                messagebox.showerror("参数错误", str(e))

    def _on_env_random_reset(self):
        """
        功能:
            使用seed+1重置环境。
        输入:
            无。
        输出:
            无。
        """
        try:
            mission, weights = self._read_mission_and_weights()
            self._apply_modes()
            _, _, seed = self._read_modes_and_seed()
            seed = int(seed) + 1
            self.inputs["seed"].set(str(seed))
            self.running = False
            self.sim.mission = mission
            self.sim.weights = weights
            self.sim.base_seed = int(seed)
            self.sim.reset_world_with_seed(int(seed))
            self.status_var.set(f"Random reset with seed={int(seed)}")
            self._draw_scene()
        except Exception as e:
            if messagebox is not None:
                messagebox.showerror("参数错误", str(e))

    def _on_plan(self):
        """
        功能:
            执行预规划。
        输入:
            无。
        输出:
            无。
        """
        try:
            mission, weights = self._read_mission_and_weights()
            self._apply_modes()
            self.sim.mission.overlap_rate = float(mission.overlap_rate)
            self.sim.mission.hunters_wait_mode = str(mission.hunters_wait_mode).lower()
            self.sim.mission.explorer_track_speed_scale = float(mission.explorer_track_speed_scale)
            self.sim.mission.loss_timeout_steps = int(mission.loss_timeout_steps)
            self.sim.weights = weights
        except Exception as e:
            if messagebox is not None:
                messagebox.showerror("参数错误", str(e))
            return
        self.sim.plan_routes()
        if getattr(self, "plan_mode", "on-loop") == "on-loop":
            self.sim.dispatch_execute()
            self.running = True
            self.status_var.set("Planned -> Dispatched")
        else:
            self.running = False
            self.sim.executing = False
            self.status_var.set("Planned (pending dispatch)")
        self._draw_scene()

    def _on_dispatch(self):
        """
        功能:
            下发执行。
        输入:
            无。
        输出:
            无。
        """
        self.sim.dispatch_execute()
        self.running = True
        self.status_var.set("Dispatched")

    def _on_start(self):
        """
        功能:
            开始连续仿真。
        输入:
            无。
        输出:
            无。
        """
        self._apply_modes()
        self.running = True

    def _on_pause(self):
        """
        功能:
            暂停连续仿真。
        输入:
            无。
        输出:
            无。
        """
        self.running = False

    def _on_step(self):
        """
        功能:
            手动执行一步仿真。
        输入:
            无。
        输出:
            无。
        """
        self.sim.step_once()
        self._draw_scene()

    def _on_manual_assignment(self):
        """
        功能:
            手动触发任务分配。
        输入:
            无。
        输出:
            无。
        """
        self._apply_modes()
        try:
            _, weights = self._read_mission_and_weights()
            self.sim.weights = weights
        except Exception as e:
            if messagebox is not None:
                messagebox.showerror("参数错误", str(e))
            return
        assignments = self.sim._compute_assignment()
        if len(assignments) == 0:
            self.status_var.set("No assignment")
            self._draw_scene()
            return
        self.sim.pending_assignment = assignments
        self.running = False
        self.sim.executing = False
        self.status_var.set("Assignment pending")
        self._draw_scene()

    def _on_apply_assignment(self):
        """
        功能:
            下发待定任务分配方案。
        输入:
            无。
        输出:
            无。
        """
        if self.sim.pending_assignment is None:
            self.status_var.set("No pending assignment")
            return
        self.sim._apply_assignment(self.sim.pending_assignment)
        self.sim.pending_assignment = None
        self.sim.executing = True
        self.sim.planned = True
        self.running = True
        self.status_var.set("Assignment dispatched")

    def _on_save_profile(self):
        """
        功能:
            保存当前分配权重为JSON文件。
        输入:
            无。
        输出:
            无。
        """
        _, weights = self._read_mission_and_weights()
        out_dir = project_root / "results" / "swarm_profiles"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "latest_profile.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(weights.__dict__, f, ensure_ascii=False, indent=2)
        self.status_var.set(f"Saved profile: {out_path}")

    def _on_load_profile(self):
        """
        功能:
            加载分配权重配置到输入框。
        输入:
            无。
        输出:
            无。
        """
        in_path = project_root / "results" / "swarm_profiles" / "latest_profile.json"
        if not in_path.exists():
            self.status_var.set("No profile file")
            return
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapping = {
            "distance_weight": "w_distance",
            "value_weight": "w_value",
            "endurance_weight": "w_endurance",
            "switch_weight": "w_switch",
            "max_assign_dist": "max_assign_dist",
        }
        for key, input_key in mapping.items():
            if key in data and input_key in self.inputs:
                self.inputs[input_key].set(str(data[key]))
        self.status_var.set(f"Loaded profile: {in_path}")

    def _schedule_loop(self):
        """
        功能:
            GUI事件循环中的定时更新逻辑。
        输入:
            无。
        输出:
            无。
        """
        if self.running:
            self.sim.step_once()
            if not self.sim.executing:
                self.running = False
            self._draw_scene()
        self.root.after(50, self._schedule_loop)

    def _draw_scene(self):
        """
        功能:
            绘制当前仿真场景与状态信息。
        输入:
            无。
        输出:
            无。
        """
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        self.ax.clear()
        ws = float(self.sim.mission.world_size)
        target_capture_dis = float(self.sim.cfg.env.capture_dis)
        explorer_perception_radius = float(getattr(self.sim.cfg.Explorer, "perception_radius", -1))
        show_explorer_perception = bool(getattr(self, "show_explorer_perception_var", None).get())
        show_target_capture = bool(getattr(self, "show_target_capture_var", None).get())
        show_all_targets = bool(getattr(self, "show_all_targets_var", None).get())
        show_pursuit_trace = bool(getattr(self, "show_pursuit_trace_var", None).get())
        show_assignment = bool(getattr(self, "show_assignment_var", None).get())
        self.ax.set_xlim(-ws, ws)
        self.ax.set_ylim(-ws, ws)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.2)

        explorer_colors = {
            "IDLE": "tab:green",
            "PREP": "tab:orange",
            "PUR": "tab:red",
            "EXH": "tab:gray",
        }
        hunter_colors = {
            "IDLE": "tab:blue",
            "PREP": "tab:orange",
            "PUR": "tab:red",
            "EXH": "tab:gray",
        }

        for ex in self.sim.explorers:
            p = ex.agent.position
            phase = self._get_explorer_phase(ex)
            color = explorer_colors.get(phase, "tab:green")
            if show_explorer_perception and explorer_perception_radius > 0:
                ex_circle = plt.Circle(
                    (float(p[0]), float(p[1])),
                    explorer_perception_radius,
                    facecolor="tab:green",
                    edgecolor="none",
                    alpha=0.08,
                )
                self.ax.add_patch(ex_circle)
            self.ax.scatter([p[0]], [p[1]], c=color, s=45, marker="^")
            if len(ex.path) > 1:
                path_np = np.asarray(ex.path)
                self.ax.plot(path_np[:, 0], path_np[:, 1], color="tab:green", alpha=0.15)

        for h in self.sim.hunters:
            p = h.agent.position
            phase = self._get_hunter_phase(h)
            color = hunter_colors.get(phase, "tab:blue")
            self.ax.scatter([p[0]], [p[1]], c=color, s=35, marker="o")
            if h.standby_mode == "split" and len(h.standby_path) > 1:
                path_np = np.asarray(h.standby_path)
                self.ax.plot(path_np[:, 0], path_np[:, 1], color="tab:blue", alpha=0.12)
            if show_pursuit_trace and int(h.pursuit_traj_start) >= 0:
                traj = h.agent.trajectory
                start_idx = int(h.pursuit_traj_start)
                if len(traj) > start_idx + 1:
                    seg = np.asarray(traj[start_idx:], dtype=np.float32)
                    self.ax.plot(seg[:, 0], seg[:, 1], color="tab:red", alpha=0.45, linewidth=1.2)

        for tid, t in enumerate(self.sim.targets):
            if not t.alive:
                continue
            if not show_all_targets and (not bool(t.in_pool)) and (not bool(t.pursuit_started)):
                continue
            p = t.agent.position
            policy_name = str(t.policy_type).lower()
            if policy_name == "patrol" and len(getattr(t.agent, "patrol_waypoints", [])) > 1:
                patrol_np = np.asarray(t.agent.patrol_waypoints, dtype=np.float32)
                self.ax.plot(
                    patrol_np[:, 0],
                    patrol_np[:, 1],
                    color="tab:purple",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.25,
                )
            if show_target_capture:
                cap_circle = plt.Circle(
                    (float(p[0]), float(p[1])),
                    target_capture_dis,
                    facecolor="tab:red",
                    edgecolor="tab:red",
                    alpha=0.10,
                    linewidth=1.0,
                )
                self.ax.add_patch(cap_circle)
            color = "tab:purple" if t.in_pool else "black"
            self.ax.scatter([p[0]], [p[1]], c=color, s=40, marker="x")
            self.ax.text(
                float(p[0]) + 1.0,
                float(p[1]) + 1.0,
                f"T{tid}:{policy_name}",
                fontsize=8,
            )

        if show_assignment:
            # Applied assignment visualization (solid)
            for tid, target in enumerate(self.sim.targets):
                if not target.alive:
                    continue
                if int(target.assigned_explorer) >= 0:
                    eid = int(target.assigned_explorer)
                    if 0 <= eid < len(self.sim.explorers):
                        ex_pos = self.sim.explorers[eid].agent.position
                        tgt_pos = target.agent.position
                        self.ax.plot(
                            [ex_pos[0], tgt_pos[0]],
                            [ex_pos[1], tgt_pos[1]],
                            color="tab:orange",
                            linestyle="-",
                            linewidth=1.3,
                            alpha=0.9,
                        )
                for hid in list(target.assigned_hunters):
                    if hid < 0 or hid >= len(self.sim.hunters):
                        continue
                    h_pos = self.sim.hunters[hid].agent.position
                    tgt_pos = target.agent.position
                    self.ax.plot(
                        [h_pos[0], tgt_pos[0]],
                        [h_pos[1], tgt_pos[1]],
                        color="tab:red",
                        linestyle="-",
                        linewidth=1.1,
                        alpha=0.8,
                    )

            # Pending assignment visualization (dashed)
            if self.sim.pending_assignment is not None and len(self.sim.pending_assignment) > 0:
                for eid, tid, hids in self.sim.pending_assignment:
                    if eid < 0 or eid >= len(self.sim.explorers):
                        continue
                    if tid < 0 or tid >= len(self.sim.targets):
                        continue
                    ex_pos = self.sim.explorers[eid].agent.position
                    tgt_pos = self.sim.targets[tid].agent.position
                    self.ax.plot(
                        [ex_pos[0], tgt_pos[0]],
                        [ex_pos[1], tgt_pos[1]],
                        color="tab:orange",
                        linestyle="--",
                        linewidth=1.1,
                        alpha=0.85,
                    )
                    for hid in hids:
                        if hid < 0 or hid >= len(self.sim.hunters):
                            continue
                        h_pos = self.sim.hunters[hid].agent.position
                        self.ax.plot(
                            [h_pos[0], tgt_pos[0]],
                            [h_pos[1], tgt_pos[1]],
                            color="tab:red",
                            linestyle="--",
                            linewidth=1.0,
                            alpha=0.7,
                        )

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor=explorer_colors["IDLE"],
                markeredgecolor=explorer_colors["IDLE"],
                markersize=7,
                label=self._t("Explorer-空闲", "Explorer-Idle"),
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor=explorer_colors["PREP"],
                markeredgecolor=explorer_colors["PREP"],
                markersize=7,
                label=self._t("Explorer-预备", "Explorer-Prep"),
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor=explorer_colors["PUR"],
                markeredgecolor=explorer_colors["PUR"],
                markersize=7,
                label=self._t("Explorer-追捕", "Explorer-Pursuit"),
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=hunter_colors["IDLE"],
                markeredgecolor=hunter_colors["IDLE"],
                markersize=7,
                label=self._t("Hunter-空闲", "Hunter-Idle"),
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=hunter_colors["PREP"],
                markeredgecolor=hunter_colors["PREP"],
                markersize=7,
                label=self._t("Hunter-预备", "Hunter-Prep"),
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=hunter_colors["PUR"],
                markeredgecolor=hunter_colors["PUR"],
                markersize=7,
                label=self._t("Hunter-追捕", "Hunter-Pursuit"),
            ),
            Line2D([0], [0], marker="x", color="black", markersize=8, label="Target"),
        ]
        if show_assignment:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="tab:orange",
                    linestyle="-",
                    linewidth=1.2,
                    label=self._t("已下发(Explorer)", "Assigned-Explorer"),
                )
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="tab:red",
                    linestyle="-",
                    linewidth=1.0,
                    label=self._t("已下发(Hunter)", "Assigned-Hunter"),
                )
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="tab:orange",
                    linestyle="--",
                    linewidth=1.1,
                    label=self._t("待下发(Explorer)", "Pending-Explorer"),
                )
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.0,
                    label=self._t("待下发(Hunter)", "Pending-Hunter"),
                )
            )
        if show_explorer_perception:
            legend_handles.append(Patch(facecolor="tab:green", alpha=0.12, label=self._t("Explorer感知范围", "Explorer-Perception")))
        if show_target_capture:
            legend_handles.append(Patch(facecolor="tab:red", alpha=0.12, label=self._t("Target捕获范围", "Target-Capture")))
        has_pending = bool(show_assignment and self.sim.pending_assignment is not None and len(self.sim.pending_assignment) > 0)
        if not has_pending:
            filtered = []
            for h in legend_handles:
                label = str(getattr(h, "get_label", lambda: "")())
                if ("Pending" in label) or ("待下发" in label):
                    continue
                filtered.append(h)
            legend_handles = filtered
        self.ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=2,
            fontsize=8,
            framealpha=0.9,
            borderaxespad=0.0,
        )

        summary = self.sim.get_summary()
        status_text = (
            f"step={int(summary['step'])}, alive={int(summary['alive_targets'])}, "
            f"pool={int(summary['pool_targets'])}, pursuing={int(summary['pursuing_targets'])}, "
            f"captured={int(summary['captured_targets'])}, "
            f"freeH={int(summary['free_hunters'])}, freeE={int(summary['free_explorers'])}"
        )
        self.ax.set_title(status_text)
        self._draw_pool_panel()
        self.canvas.draw_idle()
        self._update_pool_text()

    def _draw_pool_panel(self):
        """
        功能:
            绘制右侧任务池/资源池图形面板（目标状态+无人机续航/任务状态）。
        输入:
            无。
        输出:
            无。
        """
        if not hasattr(self, "ax_pool"):
            return

        ax = self.ax_pool
        ax.clear()
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")

        # Step 1: 目标池状态统计条形图
        total_targets = max(1, len(self.sim.targets))
        captured = sum(1 for t in self.sim.targets if not t.alive)
        pursuing = sum(1 for t in self.sim.targets if t.alive and t.pursuit_started)
        discovered = sum(1 for t in self.sim.targets if t.alive and t.in_pool and not t.pursuit_started)
        undiscovered = max(0, total_targets - captured - pursuing - discovered)

        ax.text(0.02, 0.98, self._t("任务池", "Targets"), va="top", fontsize=9, fontweight="bold")
        target_stats = [
            (self._t("未发现", "Undisc"), undiscovered, "#9e9e9e"),
            (self._t("已发现", "Discov"), discovered, "#9467bd"),
            (self._t("追捕中", "Pursue"), pursuing, "#ff7f0e"),
            (self._t("已捕获", "Capt"), captured, "#2ca02c"),
        ]
        y0 = 0.92
        for idx, (name, value, color) in enumerate(target_stats):
            y = y0 - idx * 0.055
            ratio = float(value) / float(max(1, total_targets))
            ax.add_patch(plt.Rectangle((0.02, y - 0.02), 0.62 * ratio, 0.028, color=color, alpha=0.75))
            ax.add_patch(plt.Rectangle((0.02, y - 0.02), 0.62, 0.028, fill=False, edgecolor="#cccccc", linewidth=0.6))
            ax.text(0.67, y - 0.006, f"{name}:{int(value)}", fontsize=7, va="center")

        # Step 2: 无人机资源池续航与任务状态
        ax.text(0.02, 0.67, self._t("资源池", "Resources"), va="top", fontsize=9, fontweight="bold")
        explorer_colors = {
            "IDLE": "#2ca02c",
            "PREP": "#ff7f0e",
            "PUR": "#d62728",
            "EXH": "#7f7f7f",
        }
        hunter_colors = {
            "IDLE": "#1f77b4",
            "PREP": "#ff7f0e",
            "PUR": "#d62728",
            "EXH": "#7f7f7f",
        }
        drone_rows = []
        for idx, ex in enumerate(self.sim.explorers):
            state = self._get_explorer_phase(ex)
            drone_rows.append(
                (
                    f"E{int(idx)}",
                    float(ex.remaining_endurance),
                    float(max(1e-6, ex.total_endurance)),
                    state,
                    explorer_colors.get(state, "#2ca02c"),
                )
            )
        for idx, h in enumerate(self.sim.hunters):
            state = self._get_hunter_phase(h)
            drone_rows.append(
                (
                    f"H{int(idx)}",
                    float(h.remaining_endurance),
                    float(max(1e-6, h.total_endurance)),
                    state,
                    hunter_colors.get(state, "#1f77b4"),
                )
            )

        max_rows = 12
        shown_rows = drone_rows[:max_rows]
        base_y = 0.63
        row_h = 0.045
        for ridx, (name, remain, total, state, color) in enumerate(shown_rows):
            y = base_y - ridx * row_h
            if y < 0.05:
                break
            ratio = float(np.clip(remain / max(1e-6, total), 0.0, 1.0))
            ax.text(0.02, y, name, fontsize=7, va="center", color=color)
            ax.add_patch(plt.Rectangle((0.15, y - 0.012), 0.42, 0.022, fill=False, edgecolor="#cccccc", linewidth=0.5))
            ax.add_patch(plt.Rectangle((0.15, y - 0.012), 0.42 * ratio, 0.022, color=color, alpha=0.72))
            ax.text(0.59, y, state, fontsize=6.5, va="center")

        if len(drone_rows) > max_rows:
            ax.text(0.02, 0.02, f"+{len(drone_rows)-max_rows} more", fontsize=7, color="#666666")

        # Step 3: 目标分配明细（E/H映射）
        ax.text(0.02, 0.16, self._t("目标分配", "Assignments"), va="top", fontsize=9, fontweight="bold")
        assign_y = 0.13
        for tid, target in enumerate(self.sim.targets[:6]):
            if not target.alive:
                status = "CAP"
            elif target.pursuit_started:
                status = "PUR"
            elif target.in_pool:
                status = "DIS"
            else:
                status = "UND"
            ex_tag = int(target.assigned_explorer)
            hs_tag = [int(x) for x in list(target.assigned_hunters)]
            ax.text(
                0.02,
                assign_y,
                f"T{int(tid)}[{status}] p={str(target.policy_type)[:3]} E{ex_tag} H{hs_tag}",
                fontsize=6.6,
                va="center",
            )
            assign_y -= 0.03
            if assign_y < 0.01:
                break

    def _format_endurance_bar(self, remain: float, total: float, width: int = 10) -> str:
        """
        功能:
            生成续航文本进度条。
        输入:
            remain (float): 剩余续航。
            total (float): 总续航。
            width (int): 进度条宽度。
        输出:
            str: 文本进度条。
        """
        total_safe = float(max(1e-6, total))
        ratio = float(np.clip(remain / total_safe, 0.0, 1.0))
        filled = int(round(ratio * int(max(1, width))))
        return "[" + "#" * filled + "-" * (int(max(1, width)) - filled) + "]"

    def _update_pool_text(self):
        """
        功能:
            更新任务目标池与无人机资源池文本面板。
        输入:
            无。
        输出:
            无。
        """
        if not hasattr(self, "pool_text"):
            return

        lines: List[str] = []
        lines.append(self._t("[任务目标池]", "[Target Pool]"))
        for tid, target in enumerate(self.sim.targets):
            if not target.alive:
                status = self._t("已捕获", "CAPTURED")
            elif target.pursuit_started:
                status = self._t("追捕中", "PURSUIT")
            elif target.in_pool:
                status = self._t("已发现", "DISCOVERED")
            else:
                status = self._t("未发现", "UNDISCOVERED")
            lines.append(
                "T{} | {} | policy={} | req_h={} | ex={} | hs={}".format(
                    int(tid),
                    status,
                    str(target.policy_type).lower(),
                    int(target.required_hunters),
                    int(target.assigned_explorer),
                    [int(x) for x in list(target.assigned_hunters)],
                )
            )

        lines.append("")
        lines.append(self._t("[无人机资源池 - Explorer]", "[Resources - Explorer]"))
        for eid, ex in enumerate(self.sim.explorers):
            bar = self._format_endurance_bar(ex.remaining_endurance, ex.total_endurance)
            phase = self._get_explorer_phase(ex)
            lines.append(
                "E{} | state={} | task={} | end={:.1f}/{:.1f} {}".format(
                    int(eid),
                    phase,
                    int(ex.assigned_target),
                    float(ex.remaining_endurance),
                    float(ex.total_endurance),
                    bar,
                )
            )

        lines.append("")
        lines.append(self._t("[无人机资源池 - Hunter]", "[Resources - Hunter]"))
        for hid, hunter in enumerate(self.sim.hunters):
            bar = self._format_endurance_bar(hunter.remaining_endurance, hunter.total_endurance)
            phase = self._get_hunter_phase(hunter)
            lines.append(
                "H{} | state={} | mode={} | task={} | end={:.1f}/{:.1f} {}".format(
                    int(hid),
                    phase,
                    str(hunter.standby_mode),
                    int(hunter.assigned_target),
                    float(hunter.remaining_endurance),
                    float(hunter.total_endurance),
                    bar,
                )
            )

        text_val = "\n".join(lines)
        self.pool_text.configure(state=tk.NORMAL)
        self.pool_text.delete("1.0", tk.END)
        self.pool_text.insert("1.0", text_val)
        self.pool_text.configure(state=tk.DISABLED)

    def _load_model_config(self):
        """
        功能:
            读取模型配置文件到内存。
        输入:
            无。
        输出:
            无。
        """
        if self.model_config_path is None:
            return
        cfg_path = Path(self.model_config_path)
        if not cfg_path.exists():
            return
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return
        self.model_cfg["hunters"] = dict(data.get("hunters", {})) if isinstance(data.get("hunters", {}), dict) else {}
        self.model_cfg["targets"] = dict(data.get("targets", {})) if isinstance(data.get("targets", {}), dict) else {}

    def _save_model_config(self):
        """
        功能:
            写入模型配置文件。
        输入:
            无。
        输出:
            无。
        """
        if self.model_config_path is None:
            if messagebox is not None:
                messagebox.showerror("保存失败", "未指定 --model_config_file")
            return
        cfg_path = Path(self.model_config_path)
        payload = {
            "hunters": self.model_cfg.get("hunters", {}),
            "targets": self.model_cfg.get("targets", {}),
        }
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

    def _refresh_model_dropdowns(self):
        """
        功能:
            刷新模型下拉列表。
        输入:
            无。
        输出:
            无。
        """
        if not hasattr(self, "hunter_combo"):
            return
        hunter_names = list(self.model_cfg.get("hunters", {}).keys())
        target_names = list(self.model_cfg.get("targets", {}).keys())
        self.hunter_combo["values"] = hunter_names
        self.target_combo["values"] = target_names
        if self.hunter_model_var.get() not in hunter_names and len(hunter_names) > 0:
            self.hunter_model_var.set(hunter_names[0])
        if self.target_model_var.get() not in target_names and len(target_names) > 0:
            self.target_model_var.set(target_names[0])

    def _on_batch_load_models(self):
        """
        功能:
            批量加载模型配置YAML。
        输入:
            无。
        输出:
            无。
        """
        if filedialog is None:
            return
        path = filedialog.askopenfilename(
            title="选择模型配置YAML",
            filetypes=[("YAML", "*.yaml *.yml")],
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return
        hunters = data.get("hunters", {})
        targets = data.get("targets", {})
        if isinstance(hunters, dict):
            self.model_cfg["hunters"].update(hunters)
        if isinstance(targets, dict):
            self.model_cfg["targets"].update(targets)
        self._save_model_config()
        self._refresh_model_dropdowns()
        self.status_var.set("批量加载完成")

    def _on_add_model(self, kind: str):
        """
        功能:
            新增模型路径。
        输入:
            kind (str): hunters/targets。
        输出:
            无。
        """
        if filedialog is None or simpledialog is None:
            return
        path = filedialog.askopenfilename(title="选择Actor模型")
        if not path:
            return
        alias = simpledialog.askstring("模型别称", "请输入模型别称")
        if not alias:
            return
        self.model_cfg.setdefault(kind, {})
        self.model_cfg[kind][str(alias)] = str(path)
        self._save_model_config()
        self._refresh_model_dropdowns()
        self.status_var.set(f"已添加{kind}模型")

    def _on_delete_model(self, kind: str):
        """
        功能:
            删除当前选择模型。
        输入:
            kind (str): hunters/targets。
        输出:
            无。
        """
        current = self.hunter_model_var.get() if kind == "hunters" else self.target_model_var.get()
        if not current:
            return
        if current in self.model_cfg.get(kind, {}):
            self.model_cfg[kind].pop(current, None)
            self._save_model_config()
            self._refresh_model_dropdowns()
            self.status_var.set(f"已删除{kind}模型")

    def _on_rename_model(self, kind: str):
        """
        功能:
            重命名当前选择模型。
        输入:
            kind (str): hunters/targets。
        输出:
            无。
        """
        if simpledialog is None:
            return
        current = self.hunter_model_var.get() if kind == "hunters" else self.target_model_var.get()
        if not current:
            return
        new_name = simpledialog.askstring("重命名", "请输入新别称")
        if not new_name:
            return
        path = self.model_cfg.get(kind, {}).get(current, None)
        if path is None:
            return
        self.model_cfg[kind].pop(current, None)
        self.model_cfg[kind][str(new_name)] = str(path)
        self._save_model_config()
        self._refresh_model_dropdowns()
        self.status_var.set(f"已重命名{kind}模型")

    def _on_apply_pursuit_models(self):
        """
        功能:
            下发当前选择的追捕模型。
        输入:
            无。
        输出:
            无。
        """
        hunter_name = self.hunter_model_var.get()
        target_name = self.target_model_var.get()
        hunter_path = self.model_cfg.get("hunters", {}).get(hunter_name, None)
        target_path = self.model_cfg.get("targets", {}).get(target_name, None)
        if hunter_path:
            self.sim.hunter_actor = LearnHunterActor(self.sim.cfg, hunter_path)
        if target_path:
            self.sim.target_actor = LearnTargetActor(self.sim.cfg, target_path, obs_dim=20)
        # warm up hunter actor with current pursuit env if any
        if self.sim.hunter_actor is not None and len(self.sim.pursuit_tasks) > 0:
            any_env = next(iter(self.sim.pursuit_tasks.values())).env
            team_sees_target = bool(any_env._team_sees_target())
            obs_all = any_env._build_obs(team_sees_target=team_sees_target)
            if obs_all is not None and len(obs_all) > 0:
                self.sim.hunter_actor._ensure_policy(int(np.asarray(obs_all[0]).shape[0]))
            for hid in range(len(self.sim.hunters)):
                self.sim.hunter_actor.reset_hunter(int(hid))
        self.status_var.set("追捕模型已下发")

    def run(self):
        """
        功能:
            启动GUI主循环。
        输入:
            无。
        输出:
            无。
        """
        self.root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    """
    功能:
        构建命令行解析器。
    输入:
        无。
    输出:
        argparse.ArgumentParser: 参数解析器。
    """
    parser = argparse.ArgumentParser(description="Swarm simulation GUI entry")
    parser.add_argument("--config_file", type=str, required=True, help="Path to yaml config file")
    parser.add_argument(
        "--sim_config_file",
        type=str,
        default=None,
        help="Path to standalone swarm sim yaml file",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--target_actor", type=str, default=None, help="Optional actor for target learn policy")
    parser.add_argument("--hunter_actor", type=str, default=None, help="Optional actor for hunter pursuit policy")
    parser.add_argument(
        "--model_config_file",
        type=str,
        default=None,
        help="Path to hunter/target actor model config yaml",
    )
    return parser


def load_sim_overrides(sim_config_file: Optional[str]) -> dict:
    """
    功能:
        读取独立仿真配置文件，并返回swarm_sim配置段。
    输入:
        sim_config_file (Optional[str]): 独立仿真配置路径。
    输出:
        dict: 仿真配置覆盖字典。
    """
    if sim_config_file is None:
        return {}
    cfg_path = Path(str(sim_config_file))
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"sim_config_file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("sim_config_file root must be a dict")
    if "swarm_sim" in data:
        payload = data.get("swarm_sim", {})
        if not isinstance(payload, dict):
            raise ValueError("sim_config_file key swarm_sim must be a dict")
        return dict(payload)
    return dict(data)


def main():
    """
    功能:
        程序主入口，加载配置并启动GUI。
    输入:
        无。
    输出:
        无。
    """
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config_file)
    sim_overrides = load_sim_overrides(args.sim_config_file)

    sim = SwarmSimulationCore(
        cfg=cfg,
        seed=args.seed,
        target_actor_path=args.target_actor,
        hunter_actor_path=args.hunter_actor,
        sim_overrides=sim_overrides,
    )
    sim_cfg_path = Path(args.sim_config_file) if args.sim_config_file is not None else None
    model_cfg_path = Path(args.model_config_file) if args.model_config_file is not None else None
    app = SwarmSimGUI(sim, sim_config_path=sim_cfg_path, model_config_path=model_cfg_path)
    app.run()


if __name__ == "__main__":
    main()
