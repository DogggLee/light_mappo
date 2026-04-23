"""
MAPPO training entry for Multi-UAV Pursuit.

设计原则:
1) 外部配置始终使用分层结构（merged_cfg）。
2) 环境创建直接读取分层参数，不依赖扁平化参数。
3) 仅在Runner内部需要初始化算法组件时，才进行扁平化参数映射。
"""

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)

import argparse
import json
import subprocess
from pathlib import Path
import copy

import numpy as np
import setproctitle
import torch
import yaml

from utils.util import load_config
from utils.network_diagram import export_network_diagram
from envs.env_wrappers import DummyVecEnv


class _TeeStream:
    """
    将stdout/stderr双写到终端与文件。
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


def _to_plain_data(obj):
    """
    将EasyDict/嵌套容器转换为可yaml序列化的普通Python结构。
    """
    if isinstance(obj, dict):
        return {k: _to_plain_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_data(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_plain_data(v) for v in obj]
    return obj


def _load_eval_task_specs(merged_cfg):
    """
    构建评估线程对应的固定任务规格列表。

    输入:
        merged_cfg (EasyDict): 分层配置对象。
    输出:
        list[dict] | None: 长度为n_eval_rollout_threads的任务规格列表；若未配置则返回None。
    """
    fixed_tasks = list(merged_cfg.eval.fixed_tasks)
    fixed_tasks_file = merged_cfg.eval.fixed_tasks_file
    from_external_file = fixed_tasks_file is not None
    if fixed_tasks_file is not None:
        task_path = Path(str(fixed_tasks_file))
        if not task_path.is_absolute():
            root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            task_path = root / task_path
        if not task_path.exists():
            raise FileNotFoundError(f"eval.fixed_tasks_file not found: {task_path}")
        if task_path.suffix.lower() in [".yaml", ".yml"]:
            with open(task_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        elif task_path.suffix.lower() == ".json":
            with open(task_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported eval task file format: {task_path.suffix}")

        if isinstance(data, dict):
            if "tasks" not in data:
                raise ValueError("eval task file dict must contain key 'tasks'")
            fixed_tasks = list(data["tasks"])
        elif isinstance(data, list):
            fixed_tasks = list(data)
        else:
            raise ValueError("eval task file must be a list or a dict with key 'tasks'")
    if len(fixed_tasks) == 0:
        return None, bool(from_external_file)

    # 兼容策略:
    # 1) 外部任务文件：eval线程数自动对齐为任务数；
    # 2) 非外部任务文件：沿用配置线程数，并将fixed_tasks按线程数展开。
    if from_external_file:
        return [dict(x) for x in fixed_tasks], True

    n_env = int(merged_cfg.exp.n_eval_rollout_threads)
    if n_env <= 0:
        return None, False
    out = []
    for i in range(n_env):
        out.append(dict(fixed_tasks[i % len(fixed_tasks)]))
    return out, False


def _build_target_learn_eval_specs(eval_task_specs):
    """
    基于固定评估任务构建Target=learn版本任务列表。

    输入:
        eval_task_specs (list[dict]): 固定评估任务列表。
    输出:
        list[dict]: 将target_policy_source强制为learn后的任务列表。
    """
    out = []
    for spec in eval_task_specs:
        s = dict(spec)
        s["target_policy_source"] = "learn"
        out.append(s)
    return out


def _override_eval_specs_hunters_in_zone(eval_task_specs, hunters_in_zone):
    """
    基于固定评估任务列表，统一覆盖hunters_in_zone字段。

    输入:
        eval_task_specs (list[dict]): 固定评估任务列表。
        hunters_in_zone (bool): 统一覆盖后的zone模式。
    输出:
        list[dict]: 覆盖hunters_in_zone后的新任务列表。
    """
    out = []
    for spec in eval_task_specs:
        s = dict(spec)
        s["hunters_in_zone"] = bool(hunters_in_zone)
        out.append(s)
    return out


def _infer_max_num_hunters_from_eval_tasks(eval_task_specs, fallback_max_hunters_num):
    """
    从固定评估任务中推断所需的最大hunter数量。

    输入:
        eval_task_specs (list[dict] | None): 固定评估任务列表。
        fallback_max_hunters_num (int): 当任务未提供num_hunters时的回退值。
    输出:
        int: 评估任务要求的最大hunter数量（最小为1）。
    """
    # Step 1: 空任务回退到配置值。
    if eval_task_specs is None or len(eval_task_specs) == 0:
        return int(max(1, int(fallback_max_hunters_num)))

    # Step 2: 扫描所有任务的num_hunters并取最大值。
    max_hunters = int(max(1, int(fallback_max_hunters_num)))
    for spec in eval_task_specs:
        if not isinstance(spec, dict):
            continue
        num_h = spec.get("num_hunters", max_hunters)
        try:
            max_hunters = max(max_hunters, int(num_h))
        except Exception:
            continue
    return int(max(1, max_hunters))


def _resolve_train_max_hunters_num(merged_cfg):
    """
    解析训练环境使用的最大hunter数量。

    输入:
        merged_cfg (EasyDict): 分层配置对象。
    输出:
        int: train env最大hunter数量（最小为1）。
    """
    env_max = int(max(1, int(merged_cfg.env.max_hunters_num)))
    if bool(merged_cfg.curriculum.enable):
        curriculum_choices = []
        for stage in list(merged_cfg.curriculum.stages):
            curriculum_choices.extend([int(x) for x in list(stage["num_hunters_choices"])])
        curriculum_max = max(curriculum_choices) if len(curriculum_choices) > 0 else env_max
        return int(max(env_max, curriculum_max))

    choices = [int(x) for x in list(merged_cfg.domain_randomization.train_split.hunter_count_choices)]
    choice_max = max(choices) if len(choices) > 0 else env_max
    return int(max(env_max, choice_max))


def _resolve_eval_max_hunters_num(merged_cfg, eval_task_specs):
    """
    解析评估环境使用的最大hunter数量。

    输入:
        merged_cfg (EasyDict): 分层配置对象。
        eval_task_specs (list[dict] | None): 固定评估任务列表。
    输出:
        int: eval env最大hunter数量（最小为1）。
    """
    base_max = int(max(1, int(merged_cfg.env.max_hunters_num)))
    task_max = _infer_max_num_hunters_from_eval_tasks(
        eval_task_specs=eval_task_specs,
        fallback_max_hunters_num=base_max,
    )
    return int(max(base_max, int(task_max)))


def _print_domain_randomization_settings(
    merged_cfg,
    eval_task_specs,
    train_max_hunters_num,
    eval_max_hunters_num,
):
    """
    打印domain randomization配置，便于训练启动时快速确认。

    输入:
        merged_cfg (EasyDict): 分层配置对象。
        eval_task_specs (list[dict] | None): 展开后的评估固定任务列表。
    输出:
        无。
    """
    train_split = merged_cfg.domain_randomization.train_split
    zone_choices = list(getattr(train_split, "hunters_in_zone_choices", [merged_cfg.env.hunters_in_zone]))
    print(
        "[DomainRandConfig] train.enable={}, interval={}, prob={}, hunter_choices={}, hunters_in_zone_choices={}, seed_range={}, target_policies={}, patrol_pool={}".format(
            bool(train_split.enable),
            int(train_split.regen_interval_episode),
            float(train_split.regen_prob),
            list(train_split.hunter_count_choices),
            zone_choices,
            list(train_split.seed_range),
            list(train_split.target_policy_choices),
            list(train_split.patrol_name_choices),
        )
    )
    curriculum_cfg = merged_cfg.curriculum
    if bool(curriculum_cfg.enable):
        stage_desc = [
            "{}@ratio={}->update={}".format(
                str(stage["name"]),
                float(stage["start_ratio"]),
                int(stage["start_update"]),
            )
            for stage in list(curriculum_cfg.stages)
        ]
        print(
            "[CurriculumConfig] enable={}, progress_unit=update, stages={}".format(
                bool(curriculum_cfg.enable),
                stage_desc,
            )
        )
    eval_source = "inline"
    if merged_cfg.eval.fixed_tasks_file is not None:
        eval_source = str(merged_cfg.eval.fixed_tasks_file)
    print(
        "[EvalConfig] fixed_task_source={}, test_fixed_task_source={}, fixed_tasks={}, train_max_hunters_num={}, eval_max_hunters_num={}, eval_episode_length={}, dual_eval_target_learn={}".format(
            eval_source,
            merged_cfg.eval.test_fixed_tasks_file,
            0 if eval_task_specs is None else int(len(eval_task_specs)),
            int(train_max_hunters_num),
            int(eval_max_hunters_num),
            int(_resolve_eval_episode_length(merged_cfg)),
            str(merged_cfg.env.target_policy_source).lower() == "learn",
        )
    )


def _resolve_eval_episode_length(merged_cfg):
    """
    解析评估阶段episode最大步数。

    输入:
        merged_cfg (EasyDict): 分层配置对象。
    输出:
        int: 评估阶段episode最大步数（最小为1）。
    """
    eval_episode_length = getattr(merged_cfg.eval, "eval_episode_length", None)
    if eval_episode_length is None:
        return int(max(1, int(merged_cfg.env.episode_length)))
    return int(max(1, int(eval_episode_length)))


def _require_non_empty_choices(stage, field_name):
    """
    校验课程阶段离散choices字段非空。

    输入:
        stage (dict): 单个课程阶段配置。
        field_name (str): choices字段名。
    输出:
        list: 对应choices列表。
    """
    values = list(stage[field_name])
    if len(values) == 0:
        raise ValueError("curriculum stage '{}' has empty {}.".format(str(stage["name"]), str(field_name)))
    return values


def _validate_curriculum_config(merged_cfg):
    """
    校验课程学习配置，确保课程采样与domain_randomization互斥且阶段字段完整。

    输入:
        merged_cfg (EasyDict): 分层配置对象。
    输出:
        无。
    """
    curriculum_cfg = merged_cfg.curriculum
    if not bool(curriculum_cfg.enable):
        return

    if bool(merged_cfg.domain_randomization.train_split.enable):
        raise ValueError(
            "curriculum.enable=true requires domain_randomization.train_split.enable=false; "
            "curriculum owns train task sampling when enabled."
        )

    stages = list(curriculum_cfg.stages)
    if len(stages) == 0:
        raise ValueError("curriculum.stages must be non-empty when curriculum.enable=true.")

    required_choice_fields = [
        "world_size_choices",
        "num_hunters_choices",
        "hunters_in_zone_choices",
        "hunter_max_speed_choices",
        "target_max_speed_ratio_choices",
        "capture_dis_ratio_choices",
        "target_policy_source_choices",
        "patrol_name_choices",
    ]
    supported_target_policies = {"learn", "random", "patrol", "greedy", "escape"}
    start_ratios = []
    for stage in stages:
        stage_name = str(stage["name"])
        if "start_update" in stage:
            raise ValueError(
                "curriculum stage '{}' uses start_update, but curriculum stages must use start_ratio.".format(
                    stage_name
                )
            )
        start_ratio = float(stage["start_ratio"])
        if start_ratio < 0.0 or start_ratio > 1.0:
            raise ValueError("curriculum stage '{}' start_ratio must be in [0, 1].".format(stage_name))
        start_ratios.append(float(start_ratio))

        for field_name in required_choice_fields:
            _require_non_empty_choices(stage, field_name)

        for value in list(stage["world_size_choices"]):
            if float(value) <= 0.0:
                raise ValueError("curriculum stage '{}' world_size_choices must be > 0.".format(stage_name))
        for value in list(stage["num_hunters_choices"]):
            if int(value) <= 0:
                raise ValueError("curriculum stage '{}' num_hunters_choices must be > 0.".format(stage_name))
        for value in list(stage["hunter_max_speed_choices"]):
            if float(value) <= 0.0:
                raise ValueError("curriculum stage '{}' hunter_max_speed_choices must be > 0.".format(stage_name))
        for value in list(stage["target_max_speed_ratio_choices"]):
            if float(value) <= 0.0:
                raise ValueError("curriculum stage '{}' target_max_speed_ratio_choices must be > 0.".format(stage_name))
        for value in list(stage["capture_dis_ratio_choices"]):
            if float(value) <= 0.0:
                raise ValueError("curriculum stage '{}' capture_dis_ratio_choices must be > 0.".format(stage_name))
        for value in list(stage["hunters_in_zone_choices"]):
            if isinstance(value, (bool, np.bool_)):
                continue
            if isinstance(value, int) and int(value) in (0, 1):
                continue
            raise ValueError(
                "curriculum stage '{}' hunters_in_zone_choices must contain bool or 0/1 values.".format(
                    stage_name
                )
            )

        seed_range = list(stage["seed_range"])
        if len(seed_range) != 2:
            raise ValueError("curriculum stage '{}' seed_range must contain exactly two values.".format(stage_name))

        for raw_policy in list(stage["target_policy_source_choices"]):
            policy = str(raw_policy).lower()
            if policy not in supported_target_policies:
                raise ValueError(
                    "curriculum stage '{}' has unsupported target_policy_source '{}'. "
                    "Supported values: {}".format(stage_name, str(raw_policy), sorted(supported_target_policies))
                )

    if 0.0 not in start_ratios:
        raise ValueError("curriculum.stages must include one stage with start_ratio=0.0.")
    if len(set(start_ratios)) != len(start_ratios):
        raise ValueError("curriculum.stages start_ratio values must be unique.")


def _resolve_curriculum_stage_starts(merged_cfg):
    """
    功能:
        将用户配置的start_ratio解析为环境内部使用的start_update。
    输入:
        merged_cfg (EasyDict): 分层配置对象。
    输出:
        int: 总训练updates数量。
    """
    total_updates = (
        int(merged_cfg.exp.num_env_steps)
        // int(merged_cfg.env.episode_length)
        // int(merged_cfg.exp.n_rollout_threads)
    )
    total_updates = int(max(1, total_updates))
    if not bool(merged_cfg.curriculum.enable):
        return total_updates

    resolved_updates = []
    for stage in list(merged_cfg.curriculum.stages):
        start_ratio = float(stage["start_ratio"])
        start_update = int(np.floor(start_ratio * float(total_updates)))
        stage["start_update"] = int(max(0, start_update))
        resolved_updates.append(int(stage["start_update"]))
    if len(set(resolved_updates)) != len(resolved_updates):
        raise ValueError(
            "Resolved curriculum start_update values must be unique; got {} from start_ratio values.".format(
                resolved_updates
            )
        )
    return total_updates


def make_train_env(merged_cfg, train_max_hunters_num):
    """
    创建训练环境向量封装。

    输入:
        merged_cfg (EasyDict): 分层配置对象。
            - merged_cfg.exp.n_rollout_threads: 训练并行环境数（int）。
            - merged_cfg.exp.seed: 基础随机种子（int）。
    输出:
        DummyVecEnv: 向量化环境，接口满足现有MAPPO训练循环。
    """

    train_cfg = copy.deepcopy(merged_cfg)
    train_cfg.env.max_hunters_num = int(train_max_hunters_num)

    def get_env_fn(rank):
        """
        输入:
            rank (int): 当前环境线程编号。
        输出:
            callable: 延迟创建单个环境实例的函数。
        """

        def init_env():
            """
            输入:
                无。
            输出:
                ContinuousActionEnv: 单个连续动作环境实例。
            """
            from envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv(train_cfg)
            env.set_regen_scope("train")
            env.seed(int(train_cfg.exp.seed) + rank * 1000)
            return env

        return init_env

    vec_env = DummyVecEnv([get_env_fn(i) for i in range(int(train_cfg.exp.n_rollout_threads))])
    vec_env.set_auto_reset(mode="initial")
    return vec_env


def make_eval_env(merged_cfg, eval_task_specs, eval_max_hunters_num):
    """
    创建评估环境向量封装。

    输入:
        merged_cfg (EasyDict): 分层配置对象。
            - merged_cfg.exp.n_eval_rollout_threads: 评估并行环境数（int）。
            - merged_cfg.exp.seed: 基础随机种子（int）。
    输出:
        DummyVecEnv: 评估向量环境。
    """

    # Step 0: 使用外部传入的eval最大hunter数量构建评估环境。
    eval_cfg = copy.deepcopy(merged_cfg)
    eval_cfg.env.max_hunters_num = int(eval_max_hunters_num)
    eval_cfg.env.episode_length = int(_resolve_eval_episode_length(merged_cfg))

    def get_env_fn(rank):
        """
        输入:
            rank (int): 当前环境线程编号。
        输出:
            callable: 延迟创建单个环境实例的函数。
        """

        def init_env():
            """
            输入:
                无。
            输出:
                ContinuousActionEnv: 单个连续动作环境实例。
            """
            from envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv(eval_cfg)
            env.set_regen_scope("eval")
            env.seed(int(eval_cfg.exp.seed) * 10 + rank * 1000)
            return env

        return init_env

    eval_env = DummyVecEnv(
        [get_env_fn(i) for i in range(int(eval_cfg.exp.n_eval_rollout_threads))]
    )
    if eval_task_specs is None:
        raise ValueError("Eval requires fixed tasks. Please set eval.fixed_tasks or eval.fixed_tasks_file.")
    eval_env.reset_task(mode="regen", task_specs=eval_task_specs)
    eval_env.set_auto_reset(mode="recover", task_specs=eval_task_specs)
    return eval_env


parser = argparse.ArgumentParser(
    description="mappo-pursuit", formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--config_file",
    type=str,
    required=True,
    help="Path to YAML config file",
)
parser.add_argument(
    "--cuda",
    action="store_true",
    help="Use GPU or not (CLI flag has higher priority than yaml false)",
)
parser.add_argument(
    "--time_stat",
    action="store_true",
    help="Enable detailed per-episode runtime statistics and run runner.run_time_stat()",
)
parser.add_argument(
    "--final_test_eval",
    choices=["off", "sync", "async"],
    default="off",
    help="Run eval.test_fixed_tasks_file after training: off, sync, or async.",
)
parser.add_argument(
    "--final_test_eval_model_glob",
    type=str,
    default="best_eval_capture_rate",
    help="Model glob passed to train/eval.py for final test eval.",
)


def _launch_final_test_eval(args, merged_cfg, run_dir):
    """
    功能:
        训练结束后按需启动基于eval.test_fixed_tasks_file的完整测试集评估。
    输入:
        args (argparse.Namespace): 训练入口命令行参数。
        merged_cfg (EasyDict): 已合并配置。
        run_dir (Path): 当前训练run目录。
    输出:
        subprocess.Popen | None: async模式返回子进程对象，其余模式返回None。
    """
    mode = str(args.final_test_eval).lower()
    if mode == "off":
        return None
    if merged_cfg.eval.test_fixed_tasks_file is None:
        raise ValueError("--final_test_eval requires eval.test_fixed_tasks_file")

    log_path = Path(run_dir) / "final_test_eval.log"
    cmd = [
        sys.executable,
        "train/eval.py",
        "--run_dir",
        str(run_dir),
        "--model_glob",
        str(args.final_test_eval_model_glob),
    ]
    print(
        "[FinalTestEval] mode={}, task_file={}, model_glob={}, log={}".format(
            str(mode),
            str(merged_cfg.eval.test_fixed_tasks_file),
            str(args.final_test_eval_model_glob),
            str(log_path),
        )
    )
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)
    if mode == "sync":
        try:
            ret = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        finally:
            log_f.close()
        if int(ret.returncode) != 0:
            raise RuntimeError("Final test eval failed with returncode {}".format(int(ret.returncode)))
        print("[FinalTestEval] sync finished, log={}".format(str(log_path)))
        return None

    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    log_f.close()
    print("[FinalTestEval] async started pid={}, log={}".format(int(proc.pid), str(log_path)))
    return proc


def main(args):
    """
    训练入口。

    输入:
        args (argparse.Namespace):
            - config_file (str): 配置文件路径。
            - cuda (bool): 命令行是否强制启用GPU。
    输出:
        无（执行训练并写入日志/模型文件）。
    """
    merged_cfg = load_config(args.config_file)
    _validate_curriculum_config(merged_cfg)
    _resolve_curriculum_stage_starts(merged_cfg)

    # Step 1: 校验算法档案与后端可用性
    algo_name = str(merged_cfg.exp.algorithm_name)
    algo_backend = str(getattr(merged_cfg.exp, "algorithm_backend", ""))
    if algo_backend not in ("r_mappo", "maddpg", "matd3"):
        raise NotImplementedError(
            "Unsupported algorithm backend: {} (algorithm_name={})".format(
                str(algo_backend), str(algo_name)
            )
        )
    if algo_backend == "r_mappo":
        recurrent_enabled = bool(merged_cfg.model.use_recurrent_policy) or bool(
            merged_cfg.model.use_naive_recurrent_policy
        )
        if str(algo_name) in ("rmappo", "happo") and (not recurrent_enabled):
            raise ValueError("{} requires recurrent policy enabled by profile.".format(str(algo_name)))
        if str(algo_name) in ("mappo", "ippo") and recurrent_enabled:
            raise ValueError("{} requires recurrent policy disabled by profile.".format(str(algo_name)))

    # Step 2: 设备与线程设置
    use_cuda = (bool(args.cuda) or bool(merged_cfg.exp.cuda)) and torch.cuda.is_available()
    if use_cuda:
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(int(merged_cfg.exp.n_training_threads))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(int(merged_cfg.exp.n_training_threads))

    # Step 3: 结果目录构建
    run_root = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / str(merged_cfg.env.env_name)
        / str(merged_cfg.exp.algorithm_name)
        / str(merged_cfg.exp.experiment_name)
    )
    os.makedirs(str(run_root), exist_ok=True)

    exst_run_nums = [
        int(str(folder.name).split("run")[1])
        for folder in run_root.iterdir()
        if str(folder.name).startswith("run")
    ] if run_root.exists() else []
    curr_run = "run1" if len(exst_run_nums) == 0 else f"run{max(exst_run_nums) + 1}"
    run_dir = run_root / curr_run
    os.makedirs(str(run_dir), exist_ok=True)

    # Step 4: 初始化训练日志tee输出
    train_log_path = run_dir / "train.log"
    log_f = open(train_log_path, "a", encoding="utf-8", buffering=1)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _TeeStream(orig_stdout, log_f)
    sys.stderr = _TeeStream(orig_stderr, log_f)

    try:
        # Step 5: 进程名与随机种子
        setproctitle.setproctitle(
            f"{merged_cfg.exp.algorithm_name}-{merged_cfg.env.env_name}-{merged_cfg.exp.experiment_name}"
        )
        torch.manual_seed(int(merged_cfg.exp.seed))
        torch.cuda.manual_seed_all(int(merged_cfg.exp.seed))
        np.random.seed(int(merged_cfg.exp.seed))

        # Step 6: 构建环境前处理
        eval_task_specs, from_external_file = _load_eval_task_specs(merged_cfg) if bool(merged_cfg.eval.use_eval) else (None, False)
        if bool(merged_cfg.eval.use_eval) and bool(from_external_file) and eval_task_specs is not None:
            merged_cfg.exp.n_eval_rollout_threads = int(len(eval_task_specs))
            print(
                "[EvalConfig] override n_eval_rollout_threads={} (from external fixed task file)".format(
                    int(merged_cfg.exp.n_eval_rollout_threads)
                )
            )
        train_max_hunters_num = _resolve_train_max_hunters_num(merged_cfg)
        eval_max_hunters_num = _resolve_eval_max_hunters_num(merged_cfg, eval_task_specs)
        _print_domain_randomization_settings(
            merged_cfg,
            eval_task_specs,
            train_max_hunters_num=train_max_hunters_num,
            eval_max_hunters_num=eval_max_hunters_num,
        )

        # Step 7: 保存当前实际训练配置快照
        cfg_out_path = run_dir / "train_cfg.yaml"
        with open(cfg_out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                _to_plain_data(merged_cfg),
                f,
                allow_unicode=True,
                sort_keys=False,
            )
        print("[Config] saved merged training config to {}".format(str(cfg_out_path)))

        # Step 8: 构建环境
        envs = make_train_env(merged_cfg, train_max_hunters_num=train_max_hunters_num)
        eval_envs = None
        eval_envs_zone_false = None
        eval_envs_zone_true = None
        eval_envs_target_learn = None
        eval_envs_target_learn_zone_false = None
        eval_envs_target_learn_zone_true = None
        if bool(merged_cfg.eval.use_eval):
            use_dual_zone_eval = bool(from_external_file) and (eval_task_specs is not None)
            if use_dual_zone_eval:
                eval_task_specs_zone_false = _override_eval_specs_hunters_in_zone(
                    eval_task_specs,
                    hunters_in_zone=False,
                )
                eval_task_specs_zone_true = _override_eval_specs_hunters_in_zone(
                    eval_task_specs,
                    hunters_in_zone=True,
                )
                eval_envs_zone_false = make_eval_env(
                    merged_cfg,
                    eval_task_specs_zone_false,
                    eval_max_hunters_num=eval_max_hunters_num,
                )
                eval_envs_zone_true = make_eval_env(
                    merged_cfg,
                    eval_task_specs_zone_true,
                    eval_max_hunters_num=eval_max_hunters_num,
                )
                eval_envs = eval_envs_zone_false

                if str(merged_cfg.env.target_policy_source).lower() == "learn":
                    eval_task_specs_target_learn_zone_false = _build_target_learn_eval_specs(
                        eval_task_specs_zone_false
                    )
                    eval_task_specs_target_learn_zone_true = _build_target_learn_eval_specs(
                        eval_task_specs_zone_true
                    )
                    eval_envs_target_learn_zone_false = make_eval_env(
                        merged_cfg,
                        eval_task_specs_target_learn_zone_false,
                        eval_max_hunters_num=eval_max_hunters_num,
                    )
                    eval_envs_target_learn_zone_true = make_eval_env(
                        merged_cfg,
                        eval_task_specs_target_learn_zone_true,
                        eval_max_hunters_num=eval_max_hunters_num,
                    )
                    eval_envs_target_learn = eval_envs_target_learn_zone_false
            else:
                eval_envs = make_eval_env(
                    merged_cfg,
                    eval_task_specs,
                    eval_max_hunters_num=eval_max_hunters_num,
                )
                if str(merged_cfg.env.target_policy_source).lower() == "learn":
                    eval_task_specs_target_learn = _build_target_learn_eval_specs(eval_task_specs)
                    eval_envs_target_learn = make_eval_env(
                        merged_cfg,
                        eval_task_specs_target_learn,
                        eval_max_hunters_num=eval_max_hunters_num,
                    )

        # Step 9: 构建Runner配置
        num_agents = int(train_max_hunters_num) + int(merged_cfg.env.num_explorers) + 1
        runner_cfg = {
            "envs": envs,
            "eval_envs": eval_envs,
            "eval_envs_zone_false": eval_envs_zone_false,
            "eval_envs_zone_true": eval_envs_zone_true,
            "eval_envs_target_learn": eval_envs_target_learn,
            "eval_envs_target_learn_zone_false": eval_envs_target_learn_zone_false,
            "eval_envs_target_learn_zone_true": eval_envs_target_learn_zone_true,
            "device": device,
            "run_dir": run_dir,
            "num_agents": num_agents,
            "train_max_hunters_num": int(train_max_hunters_num),
            "eval_max_hunters_num": int(eval_max_hunters_num),
            "time_stat": bool(args.time_stat),
        }

        # Step 10: 使用Multi-UAV专用Runner（分层配置 + 角色共享策略）
        from runner.uav.role_runner import RoleBasedRunner as Runner

        runner = Runner(runner_cfg, merged_cfg)

        # Step 10.1: 训练开始前导出各角色Actor/Critic网络结构图
        netviz_dir = Path(run_dir) / "network_viz"
        for role_name, policy in runner.role_policies.items():
            actor_stem = netviz_dir / f"actor_{role_name}"
            critic_stem = netviz_dir / f"critic_{role_name}"
            actor_dot = export_network_diagram(
                policy.actor,
                output_stem=actor_stem,
                graph_name=f"actor_{role_name}",
            )
            critic_dot = export_network_diagram(
                policy.critic,
                output_stem=critic_stem,
                graph_name=f"critic_{role_name}",
            )
            print(
                "[NetViz] role={} actor={} critic={}".format(
                    role_name,
                    str(actor_dot),
                    str(critic_dot),
                )
            )

        if bool(args.time_stat):
            runner.run_time_stat()
        else:
            runner.run()

        # Step 11: 收尾
        envs.close()
        if bool(merged_cfg.eval.use_eval):
            eval_env_candidates = [
                eval_envs,
                eval_envs_zone_false,
                eval_envs_zone_true,
                eval_envs_target_learn,
                eval_envs_target_learn_zone_false,
                eval_envs_target_learn_zone_true,
            ]
            closed_env_ids = {id(envs)}
            for eval_env in eval_env_candidates:
                if eval_env is None:
                    continue
                if id(eval_env) in closed_env_ids:
                    continue
                eval_env.close()
                closed_env_ids.add(id(eval_env))

        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()

        _launch_final_test_eval(args=args, merged_cfg=merged_cfg, run_dir=run_dir)
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_f.close()


if __name__ == "__main__":
    cli_args = parser.parse_args()
    main(cli_args)
