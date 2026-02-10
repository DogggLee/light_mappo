"""
# @Time    : 2024/xx/xx
# @Author  : OpenAI
# @File    : train_uav_pursuit.py
"""

import os
import sys
from pathlib import Path

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

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="uav_pursuit")
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
    parser.add_argument("--interactive_perception_confirm", type=lambda x: str(x).lower() in ["1", "true", "yes"], default=True)

    default_config = Path(__file__).resolve().parent / "config" / "pursuit3v1.yaml"
    all_args, config_path = parse_args_with_yaml(
        args,
        parser,
        default_config=str(default_config),
        require_config=True,
    )
    all_args.num_agents = all_args.num_hunters + all_args.num_blockers + 1
    return all_args, config_path


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
            print("choose to use gpu...")
            device = torch.device("cuda:0")
            if all_args.cuda_deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        except Exception as exc:
            print(f"cuda init failed, fallback to cpu: {exc}")
            device = torch.device("cpu")
            all_args.cuda = False
    else:
        print("choose to use cpu...")
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
    print(f"Save results to {str(run_dir)}")

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
        _interactive_confirm_perception(all_args)

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
