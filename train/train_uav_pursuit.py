"""
# @Time    : 2024/xx/xx
# @Author  : OpenAI
# @File    : train_uav_pursuit.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch

from config import get_config, parse_args_with_yaml
from envs.env_wrappers import DummyVecEnv
from envs.uav_pursuit_env import MultiUavPursuitEnv
from runner.shared.uav_pursuit_runner import UavPursuitRunner


def make_train_env(all_args):
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
            )
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
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

    all_args, config_path = parse_args_with_yaml(args, parser)
    all_args.env_name = "MPE"
    all_args.num_agents = all_args.num_hunters + all_args.num_blockers + 1
    all_args.share_policy = True
    return all_args, config_path


def main(args):
    parser = get_config()
    all_args, config_path = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert (
            all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy
        ), "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
            all_args.use_recurrent_policy == False
            and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
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

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
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
