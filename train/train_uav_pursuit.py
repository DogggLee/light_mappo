"""
# @Time    : 2024/xx/xx
# @Author  : OpenAI
# @File    : train_uav_pursuit.py
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

import matplotlib.pyplot as plt
import numpy as np
import setproctitle
import torch

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from parameters import get_config, parse_args_with_yaml
from envs.env_wrappers import DummyVecEnv
from envs.uav_pursuit_env import MultiUavPursuitEnv
from runner.shared.uav_pursuit_runner import UavPursuitRunner


ROLE_NAMES = ("hunter", "blocker", "target")


def _get_role_params(all_args):
    return {
        "max_speed_hunter": all_args.max_speed_hunter,
        "max_speed_blocker": all_args.max_speed_blocker,
        "max_speed_target": all_args.max_speed_target,
        "perception_hunter": all_args.perception_hunter,
        "perception_blocker": all_args.perception_blocker,
        "perception_target": all_args.perception_target,
        "speed_penalty": all_args.speed_penalty,
    }


def _render_perception_preview(all_args):
    world_size = all_args.world_size
    fig, ax = plt.subplots(figsize=(7, 7), dpi=130)
    ax.set_xlim(-world_size, world_size)
    ax.set_ylim(-world_size, world_size)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("Perception Preview Before Training")

    positions = {
        "hunter": (-0.55 * world_size, 0.0),
        "blocker": (0.0, 0.0),
        "target": (0.55 * world_size, 0.0),
    }
    colors = {"hunter": "#1f77b4", "blocker": "#2ca02c", "target": "#d62728"}
    markers = {"hunter": "o", "blocker": "s", "target": "*"}

    for role in ROLE_NAMES:
        perception = getattr(all_args, f"perception_{role}")
        x, y = positions[role]
        ax.scatter([x], [y], c=colors[role], s=90, marker=markers[role], label=f"{role} (speed={getattr(all_args, f'max_speed_{role}'):.2f})")
        ax.add_patch(plt.Circle((x, y), perception, color=colors[role], alpha=0.12))
        ax.text(x, y, f" {role}\nR={perception:.2f}", fontsize=9, va="bottom")

    ax.legend(loc="upper right")
    plt.show(block=False)
    plt.pause(0.001)


def _interactive_confirm_perception(all_args):
    while True:
        _render_perception_preview(all_args)
        try:
            answer = input(
                "\n当前感知范围: "
                f"hunter={all_args.perception_hunter}, "
                f"blocker={all_args.perception_blocker}, "
                f"target={all_args.perception_target}. "
                "输入 y 确认开始训练；输入 n 重新修改: "
            ).strip().lower()
        except EOFError:
            print("未检测到交互输入，使用当前配置继续训练。")
            plt.close("all")
            return

        if answer == "y":
            plt.close("all")
            return

        if answer != "n":
            print("无效输入，请输入 y 或 n。")
            plt.close("all")
            continue

        for role in ROLE_NAMES:
            key = f"perception_{role}"
            while True:
                raw = input(f"请输入 {role} 的感知范围(当前 {getattr(all_args, key)}): ").strip()
                try:
                    val = float(raw)
                except ValueError:
                    print("请输入合法数字。")
                    continue
                if val <= 0.0:
                    print("感知范围必须大于 0。")
                    continue
                setattr(all_args, key, val)
                break
        plt.close("all")


def make_train_env(all_args):
    role_params = _get_role_params(all_args)

    def get_env_fn(rank):
        def init_env():
            env = MultiUavPursuitEnv(
                num_hunters=all_args.num_hunters,
                num_blockers=all_args.num_blockers,
                world_size=all_args.world_size,
                dt=all_args.dt,
                capture_radius=all_args.capture_radius,
                capture_steps=all_args.capture_steps,
                max_steps=all_args.episode_length,
                seed=all_args.seed + rank * 1000,
                target_policy_source=all_args.target_policy_source,
                target_patrol_path=all_args.target_patrol_path,
                target_patrol_names=all_args.target_patrol_names,
                **role_params,
            )
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    role_params = _get_role_params(all_args)

    def get_env_fn(rank):
        def init_env():
            env = MultiUavPursuitEnv(
                num_hunters=all_args.num_hunters,
                num_blockers=all_args.num_blockers,
                world_size=all_args.world_size,
                dt=all_args.dt,
                capture_radius=all_args.capture_radius,
                capture_steps=all_args.capture_steps,
                max_steps=all_args.episode_length,
                seed=all_args.seed + rank * 1000,
                target_policy_source=all_args.target_policy_source,
                target_patrol_path=all_args.target_patrol_path,
                target_patrol_names=all_args.target_patrol_names,
                **role_params,
            )
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def _resolve_config_path(config_path):
    if not config_path:
        return None
    path = Path(config_path)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parent.parent
    return (repo_root / path).resolve()



def _load_yaml_mapping(yaml_path):
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Scenario file must be a mapping: {yaml_path}")
    return data


def _load_scenario_suite(suite_path):
    # 支持目录模式（每个yaml一个场景）和旧版单文件模式。
    if not suite_path:
        return []
    resolved = _resolve_config_path(suite_path)
    required_fields = {
        "num_hunters", "num_blockers", "world_size", "dt", "capture_radius",
        "capture_steps", "episode_length", "seed", "initial_positions", "target_patrol_route_id",
    }
    scenarios = []
    if resolved.is_dir():
        files = sorted([p for p in resolved.iterdir() if p.suffix.lower() in {".yaml", ".yml"}], key=lambda x: x.stem)
        for fp in files:
            if fp.parent.name == "patrol_routes":
                continue
            cfg = _load_yaml_mapping(fp)
            cfg.setdefault("scenario_id", fp.stem)
            cfg.setdefault("scenario_file", str(fp))
            scenarios.append(cfg)
    else:
        data = _load_yaml_mapping(resolved)
        scenarios = data.get("scenarios", []) if isinstance(data, dict) else data

    normalized = []
    for idx, sc in enumerate(scenarios):
        sc = dict(sc)
        if "target_patrol_route_id" not in sc and "target_patrol_name" in sc:
            sc["target_patrol_route_id"] = sc.get("target_patrol_name")
        missing = sorted(required_fields - set(sc.keys()))
        if missing:
            raise ValueError(f"scenario entry missing fields: {', '.join(missing)}")
        sc.setdefault("scenario_id", f"scenario_{idx}")
        normalized.append(sc)
    return normalized


def _resolve_eval_suite_path(all_args):
    # 若显式传入 scenario_suite 则优先，否则根据 split 从 config 读取 val/test。
    if getattr(all_args, "scenario_suite", None):
        return all_args.scenario_suite
    split = str(getattr(all_args, "eval_dataset_split", "val")).lower()
    if split == "test":
        return getattr(all_args, "scenario_suite_test", None)
    return getattr(all_args, "scenario_suite_val", None)

def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="uav_pursuit")
    parser.add_argument("--scenario_suite", type=str, default=None)
    parser.add_argument("--scenario_suite_val", type=str, default="datasets/val")
    parser.add_argument("--scenario_suite_test", type=str, default="datasets/test")
    parser.add_argument("--eval_dataset_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--train_patrol_route_dir", type=str, default="datasets/val/patrol_routes")
    parser.add_argument("--num_hunters", type=int, default=3)
    parser.add_argument("--num_blockers", type=int, default=0)
    parser.add_argument("--world_size", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--capture_radius", type=float, default=0.12)
    parser.add_argument("--capture_steps", type=int, default=5)
    parser.add_argument("--gif_interval", type=int, default=10)
    parser.add_argument("--gif_frame_duration", type=float, default=0.1)
    parser.add_argument("--max_speed_hunter", type=float, default=1.2)
    parser.add_argument("--max_speed_blocker", type=float, default=0.9)
    parser.add_argument("--max_speed_target", type=float, default=1.0)
    parser.add_argument("--perception_hunter", type=float, default=0.8)
    parser.add_argument("--perception_blocker", type=float, default=1.2)
    parser.add_argument("--perception_target", type=float, default=0.8)
    parser.add_argument("--speed_penalty", type=float, default=0.05)
    parser.add_argument("--interactive_perception_confirm", type=lambda x: str(x).lower() in ["1", "true", "yes"], default=True)

    config_only = argparse.ArgumentParser(add_help=False)
    config_only.add_argument("--config", type=str, default=None)
    known_args, _ = config_only.parse_known_args(args)
    config_path = None
    if len(args) == 1 and not args[0].startswith("-"):
        config_path = args[0]
    else:
        config_path = known_args.config

    if config_path is None:
        config_path = "config/pursuit3v1.yaml"

    resolved_config = _resolve_config_path(config_path)
    all_args, _ = parse_args_with_yaml(
        [],
        parser,
        default_config=str(resolved_config),
        require_config=True,
    )
    all_args.num_agents = all_args.num_hunters + all_args.num_blockers + 1
    all_args.scenario_suite_data = _load_scenario_suite(_resolve_eval_suite_path(all_args))
    all_args.test_suite_data = _load_scenario_suite(getattr(all_args, "scenario_suite_test", None))
    return all_args, str(resolved_config)


def main(args):
    parser = get_config()
    all_args, config_path = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert all_args.use_recurrent_policy is False and all_args.use_naive_recurrent_policy is False, "check recurrent policy!"
    else:
        raise NotImplementedError

    if all_args.cuda and torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
            torch.cuda.init()
            _ = torch.randn(1, 1, device="cuda") @ torch.randn(1, 1, device="cuda")
            device = torch.device("cuda:0")
            if all_args.cuda_deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        except Exception as exc:
            device = torch.device("cpu")
            all_args.cuda = False
    else:
        device = torch.device("cpu")

    torch.set_num_threads(all_args.n_training_threads)

    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    os.makedirs(str(run_dir), exist_ok=True)

    exst_run_nums = [
        int(str(folder.name).split("run")[1])
        for folder in run_dir.iterdir()
        if str(folder.name).startswith("run")
    ]
    curr_run = "run1" if len(exst_run_nums) == 0 else f"run{max(exst_run_nums) + 1}"
    run_dir = run_dir / curr_run
    os.makedirs(str(run_dir), exist_ok=True)
    print(f"Run dir: {str(run_dir)}")
    print(f"TensorBoard: tensorboard --logdir {str(run_dir / 'logs')}")

    if config_path:
        config_file = Path(config_path)
        config_target = run_dir / config_file.name
        config_target.write_text(config_file.read_text(encoding="utf-8"), encoding="utf-8")

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    torch.manual_seed(all_args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    if all_args.interactive_perception_confirm:
        if sys.stdin.isatty():
            _interactive_confirm_perception(all_args)
        else:
            print("Non-interactive session detected; skipping perception confirmation.")

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    runner = UavPursuitRunner(config)
    runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
