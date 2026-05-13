"""
Rule-based baseline demo on the same eval_envs pipeline as train/train.py.

支持规则:
- angelani
- janosov
- simple_chase
"""

import argparse
import csv
import json
import os
import time
import sys
import importlib.util
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)

from utils.util import load_config

_TRAIN_PY_PATH = Path(parent_dir) / "train" / "train.py"
_TRAIN_SPEC = importlib.util.spec_from_file_location("light_mappo_train_entry", str(_TRAIN_PY_PATH))
_TRAIN_MOD = importlib.util.module_from_spec(_TRAIN_SPEC)
_TRAIN_SPEC.loader.exec_module(_TRAIN_MOD)

_load_eval_task_specs = _TRAIN_MOD._load_eval_task_specs
_override_eval_specs_hunters_in_zone = _TRAIN_MOD._override_eval_specs_hunters_in_zone
_resolve_eval_max_hunters_num = _TRAIN_MOD._resolve_eval_max_hunters_num
make_eval_env = _TRAIN_MOD.make_eval_env


def _wrap_to_pi(angle):
    """
    功能:
        将角度包裹到[-pi, pi]。
    输入:
        angle (float): 任意实数角度（弧度）。
    输出:
        float: 包裹后的角度。
    """
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _norm_vec(vec):
    """
    功能:
        对二维向量做单位化。
    输入:
        vec (np.ndarray): shape=(2,)。
    输出:
        np.ndarray: shape=(2,) 单位向量；零向量返回零。
    """
    v = np.asarray(vec, dtype=np.float32).reshape(2)
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return np.zeros(2, dtype=np.float32)
    return (v / n).astype(np.float32)


def _clip_action(action):
    """
    功能:
        将动作裁剪到[-1,1]并控制模长不超过1。
    输入:
        action (np.ndarray): shape=(2,) 动作向量。
    输出:
        np.ndarray: shape=(2,) 归一化动作。
    """
    a = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
    n = float(np.linalg.norm(a))
    if n > 1.0:
        a = a / n
    return a.astype(np.float32)


def _compute_alive_rate(env_infos, max_hunters_num):
    """
    功能:
        计算与Runner一致的hunter非碰撞可用率。
    输入:
        env_infos (list[dict]): 单环境infos。
        max_hunters_num (int): 最大hunter数量。
    输出:
        float: [0,1]范围alive_rate。
    """
    if env_infos is None or len(env_infos) == 0:
        return 0.0
    non_collided = 0.0
    active = 0.0
    for hid in range(int(max_hunters_num)):
        if hid >= len(env_infos):
            break
        info = env_infos[hid]
        if not isinstance(info, dict):
            continue
        if not bool(info.get("active_agent", True)):
            continue
        active += 1.0
        if bool(info.get("alive", False)) and (not bool(info.get("collided", False))):
            non_collided += 1.0
    if active <= 0.0:
        return 0.0
    return float(non_collided / active)


def _extract_active_hunter_count(env_infos, max_hunters_num):
    """
    功能:
        提取单环境active hunter数量。
    输入:
        env_infos (list[dict]): 单环境infos。
        max_hunters_num (int): 最大hunter数量。
    输出:
        int: active hunter数量（至少1）。
    """
    if env_infos is None or len(env_infos) == 0:
        return int(max(1, int(max_hunters_num)))
    cnt = 0
    for hid in range(int(max_hunters_num)):
        if hid >= len(env_infos):
            break
        info = env_infos[hid]
        if not isinstance(info, dict):
            continue
        if bool(info.get("active_agent", True)):
            cnt += 1
    return int(max(1, cnt))


def _to_env_action_frame(core_env, agent, action_global):
    """
    功能:
        将全局动作映射到环境要求的动作坐标系。
    输入:
        core_env (EnvCore): 底层环境实例。
        agent (BaseAgent): 当前agent。
        action_global (np.ndarray): shape=(2,) 全局动作。
    输出:
        np.ndarray: shape=(2,) 按env.action_frame要求的动作。
    """
    a = _clip_action(action_global)
    if str(core_env.action_frame) == "global":
        return a
    return _clip_action(core_env._global_action_to_local(a, agent.heading))


def _angelani_action(core_env, hid, rule_cfg):
    """
    功能:
        计算Angelani风格追捕动作（追击+分离+边界回避）。
    输入:
        core_env (EnvCore): 底层环境实例。
        hid (int): hunter索引。
    输出:
        np.ndarray: shape=(2,) 全局动作。
    """
    hunter = core_env.hunters[int(hid)]
    target = core_env.target
    cfg = rule_cfg

    to_target = _norm_vec(np.asarray(target.position, dtype=np.float32) - np.asarray(hunter.position, dtype=np.float32))
    sep = np.zeros(2, dtype=np.float32)
    sep_radius = float(max(float(hunter.safe_dis) * 2.0, core_env.capture_dis * 1.5))

    for oid in range(int(core_env.num_hunters)):
        if oid == int(hid):
            continue
        if not bool(core_env._hunter_alive_for_team_ops(oid)):
            continue
        other = core_env.hunters[int(oid)]
        diff = np.asarray(hunter.position, dtype=np.float32) - np.asarray(other.position, dtype=np.float32)
        dist = float(np.linalg.norm(diff))
        if dist <= 1e-6 or dist > sep_radius:
            continue
        sep += diff / float((dist * dist) + 1e-6)
    sep = _norm_vec(sep)

    ws = float(core_env.world_size)
    pos = np.asarray(hunter.position, dtype=np.float32)
    wall = np.zeros(2, dtype=np.float32)
    margin = max(float(hunter.safe_dis), 1e-6)
    if (ws - pos[0]) < margin:
        wall[0] -= (margin - (ws - pos[0])) / margin
    if (pos[0] + ws) < margin:
        wall[0] += (margin - (pos[0] + ws)) / margin
    if (ws - pos[1]) < margin:
        wall[1] -= (margin - (ws - pos[1])) / margin
    if (pos[1] + ws) < margin:
        wall[1] += (margin - (pos[1] + ws)) / margin
    wall = _norm_vec(wall)

    action_global = (
        float(cfg.angelani_chase_weight) * to_target
        + float(cfg.angelani_separation_weight) * sep
        + float(cfg.angelani_wall_weight) * wall
    )
    return _clip_action(action_global)


def _janosov_action(core_env, hid, alive_ids, rule_cfg):
    """
    功能:
        计算Janosov风格围捕动作（径向约束+切向排布+追击补偿）。
    输入:
        core_env (EnvCore): 底层环境实例。
        hid (int): hunter索引。
        alive_ids (list[int]): 本环境active/alive/未碰撞hunter列表。
    输出:
        np.ndarray: shape=(2,) 全局动作。
    """
    cfg = rule_cfg
    hunter = core_env.hunters[int(hid)]
    target = core_env.target
    target_pos = np.asarray(target.position, dtype=np.float32)
    rel = np.asarray(hunter.position, dtype=np.float32) - target_pos
    r = float(np.linalg.norm(rel))
    radial = _norm_vec(rel)
    if r <= 1e-8:
        radial = np.array([1.0, 0.0], dtype=np.float32)
    tangential = np.array([-radial[1], radial[0]], dtype=np.float32)

    angles = []
    for oid in alive_ids:
        p = np.asarray(core_env.hunters[int(oid)].position, dtype=np.float32) - target_pos
        angles.append((int(oid), float(np.arctan2(float(p[1]), float(p[0])))))
    angles_sorted = sorted(angles, key=lambda x: x[1])
    rank = 0
    for i, (oid, _) in enumerate(angles_sorted):
        if int(oid) == int(hid):
            rank = int(i)
            break

    target_vel = np.asarray(target.velocity, dtype=np.float32)
    phase = float(np.arctan2(float(target_vel[1]), float(target_vel[0]))) if float(np.linalg.norm(target_vel)) > 1e-6 else 0.0
    n = max(1, len(alive_ids))
    desired_theta = float(phase + 2.0 * np.pi * (float(rank) / float(n)))
    current_theta = float(np.arctan2(float(rel[1]), float(rel[0])))
    angle_err = _wrap_to_pi(desired_theta - current_theta)

    escape_radius = float(max(1e-6, float(getattr(target, "escape_dis", core_env.capture_dis * 2.0))))
    desired_radius = float(max(core_env.capture_dis * 2.0, escape_radius * 0.65))
    radial_term = -float(cfg.janosov_radial_weight) * (r - desired_radius) * radial
    tangent_term = float(cfg.janosov_tangent_weight) * angle_err * tangential
    chase_term = float(cfg.janosov_chase_weight) * (-radial)
    action_global = radial_term + tangent_term + chase_term
    return _clip_action(action_global)


def _simple_chase_action(core_env, hid):
    """
    功能:
        计算简单直接追击动作：尽量以最大速度朝Target运动，
        若一步内可到达则按剩余距离给出更小速度指令。
    输入:
        core_env (EnvCore): 底层环境实例。
        hid (int): hunter索引。
    输出:
        np.ndarray: shape=(2,) 全局归一化动作。
    """
    hunter = core_env.hunters[int(hid)]
    target = core_env.target

    hunter_pos = np.asarray(hunter.position, dtype=np.float32)
    target_pos = np.asarray(target.position, dtype=np.float32)
    to_target = target_pos - hunter_pos
    dist = float(np.linalg.norm(to_target))
    if dist <= 1e-8:
        return np.zeros(2, dtype=np.float32)

    direction = to_target / dist
    dt = float(max(1e-8, core_env.dt))
    step_reachable_dist = float(max(1e-8, float(hunter.max_speed) * dt))
    speed_ratio = float(min(1.0, dist / step_reachable_dist))
    action_global = direction * speed_ratio
    return _clip_action(action_global)


def _build_actions_for_env(core_env, rule_name, rule_cfg):
    """
    功能:
        为单环境构建所有agent动作。
    输入:
        core_env (EnvCore): 底层环境实例。
        rule_name (str): 规则名称（angelani/janosov/simple_chase）。
    输出:
        np.ndarray: shape=(agent_num,2) 动作数组。
    """
    actions = np.zeros((int(core_env.agent_num), 2), dtype=np.float32)
    alive_ids = [
        int(hid)
        for hid in range(int(core_env.num_hunters))
        if bool(core_env._hunter_alive_for_team_ops(int(hid)))
    ]
    for hid in range(int(core_env.num_hunters)):
        hunter = core_env.hunters[int(hid)]
        if not bool(core_env._hunter_alive_for_team_ops(int(hid))):
            actions[int(hid)] = 0.0
            continue
        if str(rule_name) == "angelani":
            action_global = _angelani_action(core_env, int(hid), rule_cfg=rule_cfg)
        elif str(rule_name) == "janosov":
            action_global = _janosov_action(core_env, int(hid), alive_ids, rule_cfg=rule_cfg)
        elif str(rule_name) == "simple_chase":
            action_global = _simple_chase_action(core_env, int(hid))
        else:
            raise ValueError("Unsupported rule: {}".format(str(rule_name)))
        actions[int(hid)] = _to_env_action_frame(core_env, hunter, action_global)
    return actions


def _extract_terminal_frame(env_infos):
    """
    功能:
        从单环境infos中提取DummyVecEnv注入的终止帧。
    输入:
        env_infos (list[dict]): 单环境infos。
    输出:
        np.ndarray | None: 终止RGB帧。
    """
    if env_infos is None:
        return None
    for agent_info in env_infos:
        if not isinstance(agent_info, dict):
            continue
        frame = agent_info.get("terminal_frame", None)
        if isinstance(frame, np.ndarray):
            return frame
    return None


def _extract_target_info(env_infos, target_index):
    """
    功能:
        从单环境infos中提取Target的info字典。
    输入:
        env_infos (list[dict]): 单环境infos。
        target_index (int): Target在agent维度上的索引。
    输出:
        dict: Target对应的info字典。
    """
    if env_infos is None or len(env_infos) <= int(target_index):
        raise KeyError("Target info missing for target_index={}".format(int(target_index)))
    target_info = env_infos[int(target_index)]
    if not isinstance(target_info, dict):
        raise TypeError("Target info must be dict, got {}".format(type(target_info).__name__))
    return target_info


def _env_infos_has_capture(env_infos):
    """
    功能:
        判断单环境infos中是否记录了本步捕获。
    输入:
        env_infos (list[dict]): 单环境infos。
    输出:
        bool: 任一agent info标记captured=True时返回True。
    """
    if env_infos is None:
        return False
    for agent_info in env_infos:
        if not isinstance(agent_info, dict):
            continue
        if bool(agent_info["captured"]):
            return True
    return False


def _build_gif_header_text(episode_id, capture_step, alive_rate):
    """
    功能:
        构建GIF/PNG顶部摘要标题。
    输入:
        episode_id (int | None): 评估编号。
        capture_step (int | None): 捕获步数。
        alive_rate (float | None): 终止时alive_rate。
    输出:
        str: 标题文本。
    """
    ep_text = "NA" if episode_id is None else str(int(episode_id))
    cap_text = "NA" if capture_step is None else str(int(capture_step))
    alive_text = "NA" if alive_rate is None else "{:.1f}%".format(100.0 * float(alive_rate))
    return "Episode {} | Capture Step {} | Alive Rate {}".format(ep_text, cap_text, alive_text)


def _overlay_gif_header(frame, header_text):
    """
    功能:
        给单帧RGB图像叠加顶部标题横幅。
    输入:
        frame (np.ndarray): RGB帧。
        header_text (str): 标题文本。
    输出:
        np.ndarray: 叠加后的RGB帧。
    """
    frame_arr = np.asarray(frame)
    if frame_arr.dtype != np.uint8:
        frame_arr = np.clip(frame_arr, 0, 255).astype(np.uint8)
    if frame_arr.ndim == 2:
        frame_arr = np.repeat(frame_arr[..., None], repeats=3, axis=2)
    if frame_arr.shape[2] > 3:
        frame_arr = frame_arr[:, :, :3]

    h, w = frame_arr.shape[0], frame_arr.shape[1]
    header_h = max(28, int(round(h * 0.08)))
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, (h + header_h) / dpi), dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[header_h, h], hspace=0.0)
    ax_h = fig.add_subplot(gs[0])
    ax_i = fig.add_subplot(gs[1])
    ax_h.set_facecolor("#111111")
    ax_h.text(
        0.01,
        0.5,
        str(header_text),
        color="white",
        fontsize=10,
        va="center",
        ha="left",
        transform=ax_h.transAxes,
    )
    ax_h.set_xticks([])
    ax_h.set_yticks([])
    for spine in ax_h.spines.values():
        spine.set_visible(False)
    ax_i.imshow(frame_arr)
    ax_i.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig.canvas.draw()
    out = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    out = out.reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4)[..., :3].copy()
    plt.close(fig)
    return out


def _save_gif(frames, gif_path, episode_id, capture_step, alive_rate, frame_duration, hold_sec):
    """
    功能:
        保存GIF（含顶部摘要和终止帧停留）。
    输入:
        frames (list[np.ndarray]): 帧列表。
        gif_path (Path): 输出GIF路径。
        episode_id/capture_step/alive_rate: 标题信息。
        frame_duration (float): 每帧时长。
        hold_sec (float): 终止帧额外停留秒数。
    输出:
        无。
    """
    if frames is None or len(frames) == 0:
        return
    header_text = _build_gif_header_text(episode_id, capture_step, alive_rate)
    seq = [_overlay_gif_header(f, header_text) for f in frames]
    hold_n = max(1, int(round(float(hold_sec) / max(float(frame_duration), 1e-6))))
    seq.extend([seq[-1].copy() for _ in range(hold_n)])
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(gif_path), seq, duration=float(frame_duration), loop=0)


def _save_png(frame, png_path, episode_id, capture_step, alive_rate):
    """
    功能:
        保存PNG（含顶部摘要）。
    输入:
        frame (np.ndarray): 终止帧。
        png_path (Path): 输出PNG路径。
        episode_id/capture_step/alive_rate: 标题信息。
    输出:
        无。
    """
    if frame is None:
        return
    header_text = _build_gif_header_text(episode_id, capture_step, alive_rate)
    out = _overlay_gif_header(frame, header_text)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(png_path), out)


def _evaluate_rule_on_env(
    eval_envs,
    eval_episode_length,
    num_hunters,
    rule_name,
    rule_cfg,
    bucket,
    episode,
    gif_output_dir,
):
    """
    功能:
        在给定eval_envs上运行一次规则评估，并按Runner口径输出指标。
    输入:
        eval_envs (DummyVecEnv): 评估向量环境。
        eval_episode_length (int): 最大评估步数。
        num_hunters (int): 最大hunter数量。
        rule_name (str): 规则名称。
    输出:
        dict: 包含总指标与按hunter数量分桶指标。
    """
    obs = eval_envs.reset(mode="recover")
    n_env = int(eval_envs.num_envs)
    num_agents = int(len(eval_envs.action_space))
    act_dim = int(eval_envs.action_space[0].shape[0])
    gif_output_dir = Path(gif_output_dir)
    save_gifs = bool(rule_cfg.save_gifs)
    save_pngs = bool(rule_cfg.save_png)
    gif_env_limit = int(rule_cfg.gif_env_limit)
    gif_frame_interval = int(rule_cfg.gif_frame_interval)
    gif_frame_duration = float(rule_cfg.gif_frame_duration)
    gif_final_hold_sec = float(rule_cfg.gif_final_hold_sec)

    finished = np.zeros(n_env, dtype=bool)
    env_episode_rewards = np.zeros(n_env, dtype=np.float32)
    env_captured = np.zeros(n_env, dtype=bool)
    env_capture_step = np.full(n_env, -1, dtype=np.int32)
    env_alive_rate = np.zeros(n_env, dtype=np.float32)
    env_active_hunter_count = np.full(n_env, int(max(1, num_hunters)), dtype=np.int32)
    env_capture_escape_gap_angle = np.full(n_env, np.nan, dtype=np.float32)
    env_capture_spread_reward = np.full(n_env, np.nan, dtype=np.float32)
    last_infos = [None for _ in range(n_env)]
    tracked_env_ids = set(range(max(0, min(int(gif_env_limit), int(n_env)))))
    frames_per_env = {int(i): [] for i in tracked_env_ids}
    terminal_png_frames = {int(i): None for i in tracked_env_ids}
    gif_saved = {int(i): False for i in tracked_env_ids}
    if save_gifs or save_pngs:
        eval_envs.capture_terminal_frame = True
    if save_gifs:
        for env_i in tracked_env_ids:
            frame0 = eval_envs.render(
                mode="rgb_array",
                env_id=int(env_i),
                title="RuleDemo({}) Episode {}".format(str(bucket), int(episode)),
            )
            if isinstance(frame0, np.ndarray):
                frames_per_env[int(env_i)].append(frame0.copy())

    steps_executed = 0

    for step in range(int(eval_episode_length)):
        active_idx = np.where(np.logical_not(finished))[0]
        if active_idx.size == 0:
            break
        steps_executed = int(step + 1)

        actions_env = np.zeros((n_env, num_agents, act_dim), dtype=np.float32)
        for env_i in active_idx:
            core_env = eval_envs.envs[int(env_i)].env
            actions_env[int(env_i), :, :] = _build_actions_for_env(
                core_env,
                rule_name,
                rule_cfg=rule_cfg,
            )

        obs, rewards, dones, infos = eval_envs.step(actions_env)
        infos = infos if infos is not None else np.array([[] for _ in range(n_env)], dtype=object)

        for env_i in active_idx:
            env_infos = infos[int(env_i)] if int(env_i) < len(infos) else []
            last_infos[int(env_i)] = env_infos
            env_active_hunter_count[int(env_i)] = int(
                _extract_active_hunter_count(env_infos, max_hunters_num=int(num_hunters))
            )

            hunter_step_rewards = np.asarray(rewards[int(env_i), : int(num_hunters), 0], dtype=np.float32)
            env_episode_rewards[int(env_i)] += float(np.sum(hunter_step_rewards) / max(1, int(num_hunters)))

            if (not bool(env_captured[int(env_i)])) and _env_infos_has_capture(env_infos):
                env_captured[int(env_i)] = True
                env_capture_step[int(env_i)] = int(step + 1)
                target_info = _extract_target_info(env_infos, int(num_agents - 1))
                metric_valid = bool(target_info["max_escape_gap_metric_valid"])
                metric_angle = float(target_info["max_escape_gap_angle"])
                if metric_valid and np.isfinite(metric_angle):
                    env_capture_escape_gap_angle[int(env_i)] = float(metric_angle)
                env_capture_spread_reward[int(env_i)] = float(target_info["capture_spread_reward"])

            done_env = bool(np.all(dones[int(env_i)]))
            if done_env:
                finished[int(env_i)] = True
                env_alive_rate[int(env_i)] = float(
                    _compute_alive_rate(env_infos, max_hunters_num=int(num_hunters))
                )
                if save_gifs and int(env_i) in tracked_env_ids and (not bool(gif_saved[int(env_i)])):
                    terminal_frame = _extract_terminal_frame(env_infos)
                    if isinstance(terminal_frame, np.ndarray):
                        frames_per_env[int(env_i)].append(terminal_frame.copy())
                    gif_path = gif_output_dir / "gifs" / "e-{}-{}-env-{}.gif".format(
                        str(bucket), int(episode), int(env_i)
                    )
                    _save_gif(
                        frames=frames_per_env[int(env_i)],
                        gif_path=gif_path,
                        episode_id=int(episode),
                        capture_step=None if int(env_capture_step[int(env_i)]) <= 0 else int(env_capture_step[int(env_i)]),
                        alive_rate=float(env_alive_rate[int(env_i)]),
                        frame_duration=float(gif_frame_duration),
                        hold_sec=float(gif_final_hold_sec),
                    )
                    frames_per_env[int(env_i)] = []
                    gif_saved[int(env_i)] = True
                elif save_pngs and int(env_i) in tracked_env_ids:
                    terminal_frame = _extract_terminal_frame(env_infos)
                    if terminal_frame is None:
                        terminal_frame = eval_envs.render(
                            mode="rgb_array",
                            env_id=int(env_i),
                            title="RuleDemo({}) Episode {}".format(str(bucket), int(episode)),
                        )
                    if isinstance(terminal_frame, np.ndarray):
                        terminal_png_frames[int(env_i)] = terminal_frame.copy()

        if save_gifs and (int(step + 1) % max(1, int(gif_frame_interval)) == 0):
            for env_i in tracked_env_ids:
                if bool(finished[int(env_i)]):
                    continue
                frame = eval_envs.render(
                    mode="rgb_array",
                    env_id=int(env_i),
                    title="RuleDemo({}) Episode {}".format(str(bucket), int(episode)),
                )
                if isinstance(frame, np.ndarray):
                    frames_per_env[int(env_i)].append(frame.copy())

    for env_i in range(n_env):
        if float(env_alive_rate[int(env_i)]) > 0.0:
            continue
        env_infos = last_infos[int(env_i)]
        if env_infos is None or len(env_infos) == 0:
            continue
        env_alive_rate[int(env_i)] = float(
            _compute_alive_rate(env_infos, max_hunters_num=int(num_hunters))
        )
        if save_gifs and int(env_i) in tracked_env_ids and (not bool(gif_saved[int(env_i)])):
            gif_path = gif_output_dir / "gifs" / "e-{}-{}-env-{}.gif".format(
                str(bucket), int(episode), int(env_i)
            )
            _save_gif(
                frames=frames_per_env[int(env_i)],
                gif_path=gif_path,
                episode_id=int(episode),
                capture_step=None if int(env_capture_step[int(env_i)]) <= 0 else int(env_capture_step[int(env_i)]),
                alive_rate=float(env_alive_rate[int(env_i)]),
                frame_duration=float(gif_frame_duration),
                hold_sec=float(gif_final_hold_sec),
            )
            frames_per_env[int(env_i)] = []
            gif_saved[int(env_i)] = True
        if save_pngs and int(env_i) in tracked_env_ids:
            if terminal_png_frames[int(env_i)] is None:
                terminal_frame = _extract_terminal_frame(env_infos)
                if terminal_frame is None:
                    terminal_frame = eval_envs.render(
                        mode="rgb_array",
                        env_id=int(env_i),
                        title="RuleDemo({}) Episode {}".format(str(bucket), int(episode)),
                    )
                if isinstance(terminal_frame, np.ndarray):
                    terminal_png_frames[int(env_i)] = terminal_frame.copy()
            if isinstance(terminal_png_frames[int(env_i)], np.ndarray):
                png_path = gif_output_dir / "gifs" / "e-{}-{}-env-{}.png".format(
                    str(bucket), int(episode), int(env_i)
                )
                _save_png(
                    frame=terminal_png_frames[int(env_i)],
                    png_path=png_path,
                    episode_id=int(episode),
                    capture_step=None if int(env_capture_step[int(env_i)]) <= 0 else int(env_capture_step[int(env_i)]),
                    alive_rate=float(env_alive_rate[int(env_i)]),
                )

    total_eval_episodes = int(n_env)
    captured_episodes = int(np.sum(env_captured))
    capture_rate = float(captured_episodes / max(1, total_eval_episodes))
    eval_reward = float(np.mean(env_episode_rewards)) if total_eval_episodes > 0 else 0.0

    capture_steps_arr = np.asarray(env_capture_step, dtype=np.int32).reshape(-1)
    captured_steps = [int(capture_steps_arr[i]) for i in range(n_env) if env_captured[i] and int(capture_steps_arr[i]) > 0]
    capture_steps = float(np.mean(captured_steps)) if len(captured_steps) > 0 else float(eval_episode_length)
    capture_steps_objective = float(
        np.mean(
            [
                int(capture_steps_arr[i]) if (env_captured[i] and int(capture_steps_arr[i]) > 0) else int(eval_episode_length)
                for i in range(n_env)
            ]
        )
    )
    alive_rate = float(np.mean(env_alive_rate)) if total_eval_episodes > 0 else 0.0
    captured_valid_mask = np.logical_and(env_captured, np.isfinite(env_capture_escape_gap_angle))
    if bool(np.any(captured_valid_mask)):
        max_escape_gap_angle = float(np.mean(env_capture_escape_gap_angle[captured_valid_mask]))
    else:
        max_escape_gap_angle = float("nan")
    spread_valid_mask = np.logical_and(env_captured, np.isfinite(env_capture_spread_reward))
    capture_spread_reward = (
        float(np.mean(env_capture_spread_reward[spread_valid_mask]))
        if bool(np.any(spread_valid_mask))
        else float("nan")
    )

    metrics = {
        "eval_reward": float(eval_reward),
        "capture_rate": float(capture_rate),
        "capture_steps": float(capture_steps),
        "capture_steps_objective": float(capture_steps_objective),
        "alive_rate": float(alive_rate),
        "max_escape_gap_angle": float(max_escape_gap_angle),
        "capture_spread_reward": float(capture_spread_reward),
        "captured_episodes": int(captured_episodes),
        "total_eval_episodes": int(total_eval_episodes),
    }

    by_hunter_count = {}
    active_counts = np.asarray(env_active_hunter_count, dtype=np.int32).reshape(-1)
    for h_num in sorted(set(int(x) for x in active_counts.tolist())):
        mask = active_counts == int(h_num)
        if not np.any(mask):
            continue
        local_n = int(np.sum(mask))
        local_cap = int(np.sum(env_captured[mask]))
        local_capture_steps = [
            int(capture_steps_arr[i])
            for i in range(n_env)
            if bool(mask[i]) and bool(env_captured[i]) and int(capture_steps_arr[i]) > 0
        ]
        local_capture_steps_obj = [
            int(capture_steps_arr[i]) if (bool(env_captured[i]) and int(capture_steps_arr[i]) > 0) else int(eval_episode_length)
            for i in range(n_env)
            if bool(mask[i])
        ]
        by_hunter_count[str(int(h_num))] = {
            "eval_reward": float(np.mean(env_episode_rewards[mask])) if local_n > 0 else 0.0,
            "capture_rate": float(local_cap / max(1, local_n)),
            "capture_steps": float(np.mean(local_capture_steps)) if len(local_capture_steps) > 0 else float(eval_episode_length),
            "capture_steps_objective": float(np.mean(local_capture_steps_obj)) if len(local_capture_steps_obj) > 0 else float(eval_episode_length),
            "alive_rate": float(np.mean(env_alive_rate[mask])) if local_n > 0 else 0.0,
            "max_escape_gap_angle": (
                float(np.mean(env_capture_escape_gap_angle[np.logical_and(mask, np.logical_and(env_captured, np.isfinite(env_capture_escape_gap_angle)))]))
                if bool(np.any(np.logical_and(mask, np.logical_and(env_captured, np.isfinite(env_capture_escape_gap_angle)))))
                else float("nan")
            ),
            "capture_spread_reward": (
                float(np.mean(env_capture_spread_reward[np.logical_and(mask, np.logical_and(env_captured, np.isfinite(env_capture_spread_reward)))]))
                if bool(np.any(np.logical_and(mask, np.logical_and(env_captured, np.isfinite(env_capture_spread_reward)))))
                else float("nan")
            ),
            "captured_episodes": int(local_cap),
            "total_eval_episodes": int(local_n),
        }

    return {
        "metrics": metrics,
        "by_hunter_count": by_hunter_count,
        "total_num_steps": int(max(0, steps_executed)) * int(n_env),
    }


def _write_eval_csv(rows, out_path):
    """
    功能:
        写出与常规训练eval.csv一致字段的结果CSV。
    输入:
        rows (list[dict]): 行数据。
        out_path (Path): 输出路径。
    输出:
        无。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "total_num_steps",
        "bucket",
        "eval_reward",
        "capture_rate",
        "capture_steps",
        "capture_steps_objective",
        "alive_rate",
        "max_escape_gap_angle",
        "capture_spread_reward",
        "captured_episodes",
        "total_eval_episodes",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _update_eval_hunter_bucket_plot(
    episode,
    total_num_steps,
    bucket_to_metrics,
    output_dir,
):
    """
    功能:
        输出与常规评估一致风格的“指标-随hunter数量变化”分桶曲线图与json。
    输入:
        episode (int): 评估编号。
        total_num_steps (int): 对齐eval.csv的累计步数。
        bucket_to_metrics (dict[str, dict]): 各bucket的by_hunter_count字典。
        output_dir (Path): 输出目录。
    输出:
        无。
    """
    out_dir = Path(output_dir)
    metrics_order = [
        "eval_reward",
        "capture_rate",
        "capture_steps",
        "alive_rate",
        "max_escape_gap_angle",
        "capture_spread_reward",
    ]
    metric_titles = {
        "eval_reward": "Eval Reward",
        "capture_rate": "Capture Rate",
        "capture_steps": "Capture Steps",
        "alive_rate": "Alive Rate",
        "max_escape_gap_angle": "Max Potential Escape Gap Angle (deg)",
        "capture_spread_reward": "Capture Spread Reward",
    }
    bucket_style = {
        "fixed": {"label": "fixed", "color": "#1f77b4", "marker": "o"},
        "fixed_zone_false": {"label": "fixed zone=false", "color": "#1f77b4", "marker": "o"},
        "fixed_zone_true": {"label": "fixed zone=true", "color": "#2ca02c", "marker": "^"},
    }
    bucket_plot_order = ["fixed", "fixed_zone_false", "fixed_zone_true"]

    fig, axes = plt.subplots(len(metrics_order), 1, figsize=(7.5, 17.0), dpi=120)
    plot_export = {
        "episode": int(episode),
        "total_num_steps": int(total_num_steps),
        "buckets": {},
    }
    for idx, metric_name in enumerate(metrics_order):
        ax = axes[idx]
        x_ticks = set()
        for bucket_name in bucket_plot_order:
            if bucket_name not in bucket_to_metrics:
                continue
            grouped = bucket_to_metrics[bucket_name]
            if grouped is None or len(grouped) == 0:
                continue
            x_vals = sorted([int(k) for k in grouped.keys()])
            y_vals = []
            for hc in x_vals:
                key = str(int(hc))
                metric_val = float(grouped[key].get(metric_name, float("nan")))
                if metric_name == "max_escape_gap_angle":
                    if np.isfinite(metric_val):
                        metric_val = float(np.degrees(metric_val))
                    else:
                        metric_val = 360.0
                y_vals.append(float(metric_val))
            x_ticks.update(x_vals)
            style = bucket_style[bucket_name]
            ax.plot(
                x_vals,
                y_vals,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=4.5,
            )
            if len(x_vals) > 0:
                if metric_name in ("eval_reward", "capture_rate", "alive_rate", "capture_spread_reward"):
                    best_pos = int(np.argmax(np.asarray(y_vals, dtype=np.float32)))
                else:
                    best_pos = int(np.argmin(np.asarray(y_vals, dtype=np.float32)))
                best_x = int(x_vals[best_pos])
                best_y = float(y_vals[best_pos])
                ax.scatter([best_x], [best_y], marker="*", s=70, color=style["color"], zorder=4)
                ax.annotate(
                    "{:.3f}".format(float(best_y)),
                    xy=(best_x, best_y),
                    xytext=(6, 6),
                    textcoords="offset points",
                    color=style["color"],
                    fontsize=8,
                )
            if bucket_name not in plot_export["buckets"]:
                plot_export["buckets"][bucket_name] = {}
            plot_export["buckets"][bucket_name][metric_name] = {
                "x": [int(x) for x in x_vals],
                "y": [float(y) for y in y_vals],
            }
        ax.set_title(metric_titles[metric_name])
        ax.set_xlabel("Number of Hunters")
        if len(x_ticks) > 0:
            ax.set_xticks(sorted([int(x) for x in x_ticks]))
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
        if metric_name in ("capture_rate", "alive_rate"):
            ax.set_ylim(0.0, 1.0)
        if metric_name == "max_escape_gap_angle":
            ax.set_ylim(0.0, 360.0)
        ax.set_ylabel("Value")
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend(loc="best")

    fig.suptitle(
        "Eval Metrics vs Hunters | episode={} | steps={}".format(int(episode), int(total_num_steps))
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    png_path = out_dir / "eval_hunter_bucket_metrics_ep_{}.png".format(int(episode))
    fig.savefig(str(png_path))
    plt.close(fig)

    json_path = out_dir / "eval_hunter_bucket_metrics_ep_{}.json".format(int(episode))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(plot_export, f, ensure_ascii=False, indent=2)


def main():
    """
    功能:
        运行独立规则baseline评估（复用train.py的eval_env创建流程）。
    输入:
        命令行参数。
    输出:
        无（写出json/csv结果）。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to merged-yaml input.")
    parser.add_argument(
        "--rule",
        type=str,
        required=True,
        choices=["angelani", "janosov", "simple_chase"],
        help="Rule policy name.",
    )
    parser.add_argument(
        "--fixed_tasks_file",
        type=str,
        required=False,
        help="Optional override for eval.fixed_tasks_file used by rule demo.",
    )
    parser.add_argument(
        "--task_file",
        type=str,
        required=False,
        help="Alias of --fixed_tasks_file, matching scripts/batch_eval_best_models.py.",
    )
    parser.add_argument("--output_dir", type=str, required=False, help="Optional output directory.")
    args = parser.parse_args()

    cfg = load_config(args.config_file)
    fixed_tasks_file = args.fixed_tasks_file
    if args.task_file is not None:
        if fixed_tasks_file is not None and str(args.task_file) != str(fixed_tasks_file):
            raise ValueError("--task_file and --fixed_tasks_file must match when both are provided.")
        fixed_tasks_file = str(args.task_file)
    if fixed_tasks_file is not None:
        cfg.eval.fixed_tasks_file = str(fixed_tasks_file)
    eval_task_specs, from_external_file = _load_eval_task_specs(cfg)
    if eval_task_specs is None:
        raise ValueError("Rule demo requires fixed eval tasks. Please set eval.fixed_tasks or eval.fixed_tasks_file.")
    if bool(from_external_file):
        cfg.exp.n_eval_rollout_threads = int(len(eval_task_specs))

    eval_max_hunters_num = _resolve_eval_max_hunters_num(cfg, eval_task_specs)
    eval_episode_length = int(cfg.eval.eval_episode_length)

    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = (
            Path(cfg.rule_demo.output_root)
            / str(cfg.env.env_name)
            / str(args.rule)
            / str(cfg.exp.experiment_name)
            / str(stamp)
        )
    else:
        out_dir = Path(str(args.output_dir))
    if not out_dir.is_absolute():
        root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_to_specs = {"fixed": eval_task_specs}
    if bool(cfg.rule_demo.eval_zone_split):
        bucket_to_specs["fixed_zone_false"] = _override_eval_specs_hunters_in_zone(eval_task_specs, hunters_in_zone=False)
        bucket_to_specs["fixed_zone_true"] = _override_eval_specs_hunters_in_zone(eval_task_specs, hunters_in_zone=True)

    eval_rows = []
    bucket_plot_metrics = {}
    max_total_num_steps = 0
    json_result = {
        "rule": str(args.rule),
        "config_file": str(args.config_file),
        "fixed_tasks_file": None if cfg.eval.fixed_tasks_file is None else str(cfg.eval.fixed_tasks_file),
        "eval_episode_length": int(eval_episode_length),
        "eval_max_hunters_num": int(eval_max_hunters_num),
        "buckets": {},
    }

    for bucket_name, task_specs in bucket_to_specs.items():
        eval_envs = make_eval_env(cfg, eval_task_specs=task_specs, eval_max_hunters_num=eval_max_hunters_num)
        try:
            bucket_result = _evaluate_rule_on_env(
                eval_envs=eval_envs,
                eval_episode_length=eval_episode_length,
                num_hunters=eval_max_hunters_num,
                rule_name=str(args.rule),
                rule_cfg=cfg.rule_demo,
                bucket=str(bucket_name),
                episode=0,
                gif_output_dir=out_dir,
            )
        finally:
            eval_envs.close()

        json_result["buckets"][str(bucket_name)] = bucket_result
        bucket_plot_metrics[str(bucket_name)] = dict(bucket_result["by_hunter_count"])
        metric = bucket_result["metrics"]
        total_num_steps = int(bucket_result.get("total_num_steps", 0))
        max_total_num_steps = max(max_total_num_steps, total_num_steps)
        eval_rows.append(
            {
                "episode": 0,
                "total_num_steps": int(total_num_steps),
                "bucket": str(bucket_name),
                "eval_reward": float(metric["eval_reward"]),
                "capture_rate": float(metric["capture_rate"]),
                "capture_steps": float(metric["capture_steps"]),
                "capture_steps_objective": float(metric["capture_steps_objective"]),
                "alive_rate": float(metric["alive_rate"]),
                "max_escape_gap_angle": float(metric["max_escape_gap_angle"]),
                "capture_spread_reward": float(metric["capture_spread_reward"]),
                "captured_episodes": int(metric["captured_episodes"]),
                "total_eval_episodes": int(metric["total_eval_episodes"]),
            }
        )
        print(
            "[RuleDemo] bucket={} | capture_rate={:.4f} | capture_steps={:.2f} | alive_rate={:.4f} | eval_reward={:.4f} | capture_spread={}".format(
                str(bucket_name),
                float(metric["capture_rate"]),
                float(metric["capture_steps"]),
                float(metric["alive_rate"]),
                float(metric["eval_reward"]),
                "{:.4f}".format(float(metric["capture_spread_reward"]))
                if np.isfinite(float(metric["capture_spread_reward"]))
                else "NA",
            )
        )

    json_path = out_dir / "rule_demo_result.json"
    csv_path = out_dir / "eval.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)
    _write_eval_csv(eval_rows, csv_path)
    _update_eval_hunter_bucket_plot(
        episode=0,
        total_num_steps=int(max_total_num_steps),
        bucket_to_metrics=bucket_plot_metrics,
        output_dir=out_dir,
    )
    print("[RuleDemo] result_json={}".format(str(json_path)))
    print("[RuleDemo] eval_csv={}".format(str(csv_path)))


if __name__ == "__main__":
    main()
