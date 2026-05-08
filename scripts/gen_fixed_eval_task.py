#!/usr/bin/env python3
"""
生成固定评估任务文件（支持“基础环境 x hunter数量”组合生成）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    """
    功能:
        解析命令行参数。
    输入:
        无。
    输出:
        argparse.Namespace: 解析后的参数对象。
    """
    # Step 1: 构建参数解析器并注册基础参数
    parser = argparse.ArgumentParser(
        description="Generate fixed eval task specs JSON for UAV pursuit."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出任务JSON路径，例如 config/eval_tasks/fixed_eval_generated.json",
    )
    parser.add_argument(
        "--num_base_envs",
        type=int,
        default=None,
        help="基础环境数量N。最终任务数为 N * len(hunter_count_choices)。",
    )
    parser.add_argument(
        "--hunter_count_choices",
        type=str,
        default=None,
        help="参与横向对比的hunter数量集合，逗号分隔，例如 1,2,4,8。",
    )
    parser.add_argument(
        "--hunter_count_range",
        type=int,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="可选hunter数量闭区间[min,max]；与--hunter_count_choices二选一。",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=None,
        help="兼容旧参数：等价于--num_base_envs。",
    )
    parser.add_argument(
        "--max_hunters",
        type=int,
        default=None,
        help="兼容旧参数：若未提供hunter choices/range，则使用[1,max_hunters]。",
    )
    parser.add_argument(
        "--world_size",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="可选world_size采样区间[min,max]；未提供则不写入world_size字段。",
    )
    parser.add_argument(
        "--world_size_choices",
        type=str,
        default=None,
        help="离散world_size集合，逗号分隔；与--world_size二选一。",
    )
    parser.add_argument(
        "--capture_dis",
        type=float,
        default=None,
        help="写入每个任务的capture_dis。",
    )
    parser.add_argument(
        "--capture_dis_choices",
        type=str,
        default=None,
        help="离散capture_dis集合，逗号分隔；与--capture_dis/--capture_dis_ratio_choices二选一。",
    )
    parser.add_argument(
        "--capture_dis_ratio_choices",
        type=str,
        default=None,
        help="离散capture_dis/hunter_max_speed比例集合，逗号分隔。",
    )
    parser.add_argument(
        "--capture_step",
        type=int,
        default=None,
        help="写入每个任务的capture_step。",
    )
    parser.add_argument(
        "--capture_step_choices",
        type=str,
        default=None,
        help="离散capture_step集合，逗号分隔；与--capture_step二选一。",
    )
    parser.add_argument(
        "--collision_dis",
        type=float,
        required=True,
        help="写入每个任务的collision_dis。",
    )
    parser.add_argument(
        "--hunter_safe_dis",
        type=float,
        required=True,
        help="写入每个任务的hunter_safe_dis，用于固定zone初始化间距。",
    )
    parser.add_argument(
        "--hunter_max_speed",
        type=float,
        default=None,
        help="写入每个任务的hunter_max_speed。",
    )
    parser.add_argument(
        "--hunter_max_speed_choices",
        type=str,
        default=None,
        help="离散hunter_max_speed集合，逗号分隔；与--hunter_max_speed二选一。",
    )
    parser.add_argument(
        "--target_max_speed",
        type=float,
        default=None,
        help="写入每个任务的target_max_speed。",
    )
    parser.add_argument(
        "--target_max_speed_ratio_choices",
        type=str,
        default=None,
        help="离散target_max_speed/hunter_max_speed比例集合，逗号分隔。",
    )

    # Step 2: 注册策略与巡逻相关参数
    parser.add_argument(
        "--target_policy_choices",
        type=str,
        default="random,patrol",
        help="Target策略候选，逗号分隔，例如 random,patrol。",
    )
    parser.add_argument(
        "--target_patrol_paths",
        type=str,
        nargs="+",
        default=["datasets/patrol_routes.json"],
        help="patrol路线JSON路径列表（空格分隔）；也兼容单参数逗号分隔写法。",
    )
    parser.add_argument(
        "--target_route_id",
        type=int,
        default=0,
        help="写入任务的target_route_id（通常保持0）。",
    )
    parser.add_argument(
        "--hunters_in_zone_choices",
        type=str,
        default="false,true",
        help="基础环境中hunters_in_zone采样集合，逗号分隔，例如 false,true。",
    )
    parser.add_argument(
        "--target_avoid_hunter_zone",
        type=str,
        default=None,
        help="可选：写入每个任务的target_avoid_hunter_zone，取true/false；未提供则不写入该字段。",
    )
    parser.add_argument(
        "--target_hunter_zone_min_dis",
        type=float,
        required=True,
        help="写入每个任务的target_hunter_zone_min_dis。",
    )

    # Step 3: 注册随机种子控制参数
    parser.add_argument(
        "--seed_start",
        type=int,
        default=10000,
        help="任务seed起始值。",
    )
    parser.add_argument(
        "--seed_step",
        type=int,
        default=1,
        help="任务seed递增步长。",
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=2026,
        help="生成器随机种子（用于hunter/world_size/策略采样）。",
    )
    parser.add_argument(
        "--factorial",
        action="store_true",
        help="使用离散因子的笛卡尔组合生成base环境；未开启时按num_base_envs随机采样。",
    )
    return parser.parse_args()


def _parse_int_choices(raw_text: str, arg_name: str) -> list[int]:
    """
    功能:
        解析逗号分隔整数列表并去重排序。
    输入:
        raw_text (str): 逗号分隔字符串。
        arg_name (str): 参数名（用于报错信息）。
    输出:
        list[int]: 去重排序后的整数列表。
    """
    values: list[int] = []
    for token in str(raw_text).split(","):
        s = token.strip()
        if not s:
            continue
        values.append(int(s))
    uniq = sorted(set(values))
    if len(uniq) == 0:
        raise ValueError(f"No valid integer choices in {arg_name}")
    return uniq


def _parse_float_choices(raw_text: str, arg_name: str) -> list[float]:
    """
    功能:
        解析逗号分隔浮点列表并去重排序。
    输入:
        raw_text (str): 逗号分隔字符串。
        arg_name (str): 参数名（用于报错信息）。
    输出:
        list[float]: 去重排序后的浮点列表。
    """
    values: list[float] = []
    for token in str(raw_text).split(","):
        s = token.strip()
        if not s:
            continue
        values.append(float(s))
    uniq = sorted(set(values))
    if len(uniq) == 0:
        raise ValueError(f"No valid float choices in {arg_name}")
    return uniq


def _resolve_choice_pool(single_value, choices_text, arg_name: str, value_type):
    """
    功能:
        将单值参数或choices参数解析为候选池。
    输入:
        single_value: 单值参数。
        choices_text: 逗号分隔候选字符串。
        arg_name (str): 参数名（用于报错信息）。
        value_type: int或float。
    输出:
        list: 候选值列表。
    """
    if single_value is not None and choices_text is not None:
        raise ValueError(f"Use only one of --{arg_name} and --{arg_name}_choices")
    if choices_text is not None:
        if value_type is int:
            return _parse_int_choices(str(choices_text), f"--{arg_name}_choices")
        return _parse_float_choices(str(choices_text), f"--{arg_name}_choices")
    if single_value is not None:
        return [value_type(single_value)]
    return None


def _parse_bool_choices(raw_text: str, arg_name: str) -> list[bool]:
    """
    功能:
        解析逗号分隔布尔列表（true/false/1/0/yes/no）并去重。
    输入:
        raw_text (str): 逗号分隔字符串。
        arg_name (str): 参数名（用于报错信息）。
    输出:
        list[bool]: 去重后的布尔列表。
    """
    mapper = {
        "1": True,
        "true": True,
        "t": True,
        "yes": True,
        "y": True,
        "on": True,
        "0": False,
        "false": False,
        "f": False,
        "no": False,
        "n": False,
        "off": False,
    }
    values: list[bool] = []
    for token in str(raw_text).split(","):
        s = token.strip().lower()
        if not s:
            continue
        if s not in mapper:
            raise ValueError(f"Invalid bool choice '{token}' in {arg_name}")
        values.append(bool(mapper[s]))
    uniq: list[bool] = []
    for val in values:
        if val not in uniq:
            uniq.append(val)
    if len(uniq) == 0:
        raise ValueError(f"No valid bool choices in {arg_name}")
    return uniq


def _parse_bool_value(raw_text: str, arg_name: str) -> bool:
    """
    功能:
        解析单个布尔参数。
    输入:
        raw_text (str): 布尔字符串。
        arg_name (str): 参数名（用于报错信息）。
    输出:
        bool: 解析后的布尔值。
    """
    values = _parse_bool_choices(str(raw_text), arg_name)
    if len(values) != 1:
        raise ValueError(f"{arg_name} expects exactly one bool value")
    return bool(values[0])


def _resolve_hunter_count_choices(args: argparse.Namespace) -> list[int]:
    """
    功能:
        解析最终hunter数量候选集合（新参数优先，兼容旧参数）。
    输入:
        args (argparse.Namespace): 命令行参数对象。
    输出:
        list[int]: 去重排序后的hunter数量列表。
    """
    if args.hunter_count_choices is not None and args.hunter_count_range is not None:
        raise ValueError("Use only one of --hunter_count_choices and --hunter_count_range")

    if args.hunter_count_choices is not None:
        choices = _parse_int_choices(str(args.hunter_count_choices), "--hunter_count_choices")
    elif args.hunter_count_range is not None:
        h_min = int(args.hunter_count_range[0])
        h_max = int(args.hunter_count_range[1])
        if h_max < h_min:
            raise ValueError("--hunter_count_range MAX must be >= MIN")
        choices = list(range(int(h_min), int(h_max) + 1))
    else:
        if args.max_hunters is None:
            raise ValueError(
                "Please provide hunter counts via --hunter_count_choices/--hunter_count_range "
                "or legacy --max_hunters"
            )
        max_hunters = int(args.max_hunters)
        if max_hunters < 1:
            raise ValueError("--max_hunters must be >= 1")
        choices = list(range(1, max_hunters + 1))

    cleaned = sorted({int(x) for x in choices if int(x) >= 1})
    if len(cleaned) == 0:
        raise ValueError("No valid hunter count after filtering (must be >= 1)")
    return cleaned


def _resolve_num_base_envs(args: argparse.Namespace) -> int:
    """
    功能:
        解析基础环境数量（优先新参数，兼容旧参数）。
    输入:
        args (argparse.Namespace): 命令行参数对象。
    输出:
        int: 基础环境数量。
    """
    if args.num_base_envs is not None and args.num_tasks is not None:
        raise ValueError("Use only one of --num_base_envs and legacy --num_tasks")
    val = args.num_base_envs if args.num_base_envs is not None else args.num_tasks
    if val is None:
        raise ValueError("Please provide --num_base_envs (or legacy --num_tasks)")
    out = int(val)
    if out <= 0:
        raise ValueError("--num_base_envs/--num_tasks must be > 0")
    return out


def _load_route_names_from_json(route_path: Path) -> list[str]:
    """
    功能:
        从巡逻路线JSON中读取全部可用路线名。
    输入:
        route_path (Path): 路线JSON文件路径。
    输出:
        list[str]: 路线名列表。
    """
    # Step 1: 文件存在性校验与JSON读取
    if not route_path.exists():
        raise FileNotFoundError(f"Patrol route file not found: {route_path}")
    with route_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Step 2: 解析routes数组中的name字段
    names: list[str] = []
    for route in data.get("routes", []):
        if not isinstance(route, dict):
            continue
        name = route.get("name")
        if isinstance(name, str) and len(name.strip()) > 0:
            names.append(name.strip())
    return names


def _resolve_patrol_route_pool(target_patrol_paths_arg: list[str]) -> list[tuple[str, str]]:
    """
    功能:
        解析多个patrol路线文件，构建(路径, 路线名)候选池。
    输入:
        target_patrol_paths_arg (list[str]): 巡逻路线JSON路径列表（支持元素内逗号拼接）。
    输出:
        list[tuple[str, str]]: 可采样的(路径, 路线名)元组列表。
    """
    # Step 1: 解析路径列表（兼容空格分隔与逗号分隔）
    raw_items = target_patrol_paths_arg if isinstance(target_patrol_paths_arg, list) else [str(target_patrol_paths_arg)]
    path_list: list[str] = []
    for raw in raw_items:
        path_list.extend([x.strip() for x in str(raw).split(",") if x.strip()])
    if len(path_list) == 0:
        raise ValueError("No valid path in --target_patrol_paths")

    # Step 2: 逐路径读取全部route name并构建候选池
    route_pool: list[tuple[str, str]] = []
    for raw_path in path_list:
        names = _load_route_names_from_json(Path(raw_path))
        for name in names:
            route_pool.append((str(raw_path), str(name)))
    return route_pool


def build_tasks(args: argparse.Namespace) -> list[dict]:
    """
    功能:
        根据输入参数生成固定评估任务列表。
    输入:
        args (argparse.Namespace): 命令行参数对象。
    输出:
        list[dict]: 任务规格字典列表。
    """
    # Step 1: 参数有效性校验
    num_base_envs = _resolve_num_base_envs(args)
    hunter_count_choices = _resolve_hunter_count_choices(args)
    if args.world_size is not None and args.world_size_choices is not None:
        raise ValueError("Use only one of --world_size and --world_size_choices")
    world_size_range = None
    world_size_pool = None
    if args.world_size is not None:
        world_min = float(args.world_size[0])
        world_max = float(args.world_size[1])
        if world_max < world_min:
            raise ValueError("--world_size MAX must be >= MIN")
        world_size_range = (world_min, world_max)
    if args.world_size_choices is not None:
        world_size_pool = _parse_float_choices(str(args.world_size_choices), "--world_size_choices")
        if any(float(x) <= 0.0 for x in world_size_pool):
            raise ValueError("--world_size_choices values must be > 0")
    if bool(args.factorial) and world_size_range is not None:
        raise ValueError("--factorial requires discrete --world_size_choices instead of --world_size range")

    capture_dis_pool = _resolve_choice_pool(
        args.capture_dis,
        args.capture_dis_choices,
        "capture_dis",
        float,
    )
    capture_dis_ratio_pool = None
    if args.capture_dis_ratio_choices is not None:
        if capture_dis_pool is not None:
            raise ValueError("Use only one of --capture_dis/--capture_dis_choices and --capture_dis_ratio_choices")
        capture_dis_ratio_pool = _parse_float_choices(
            str(args.capture_dis_ratio_choices),
            "--capture_dis_ratio_choices",
        )
        if any(float(x) <= 0.0 for x in capture_dis_ratio_pool):
            raise ValueError("--capture_dis_ratio_choices values must be > 0")
    if capture_dis_pool is None and capture_dis_ratio_pool is None:
        raise ValueError("Please provide --capture_dis, --capture_dis_choices, or --capture_dis_ratio_choices")
    if capture_dis_pool is not None and any(float(x) <= 0.0 for x in capture_dis_pool):
        raise ValueError("--capture_dis values must be > 0")

    capture_step_pool = _resolve_choice_pool(
        args.capture_step,
        args.capture_step_choices,
        "capture_step",
        int,
    )
    if capture_step_pool is not None and any(int(x) <= 0 for x in capture_step_pool):
        raise ValueError("--capture_step values must be > 0")
    if float(args.collision_dis) < 0.0:
        raise ValueError("--collision_dis must be >= 0")
    if float(args.hunter_safe_dis) < 0.0:
        raise ValueError("--hunter_safe_dis must be >= 0")

    hunter_max_speed_pool = _resolve_choice_pool(
        args.hunter_max_speed,
        args.hunter_max_speed_choices,
        "hunter_max_speed",
        float,
    )
    if hunter_max_speed_pool is None:
        raise ValueError("Please provide --hunter_max_speed or --hunter_max_speed_choices")
    if any(float(x) <= 0.0 for x in hunter_max_speed_pool):
        raise ValueError("--hunter_max_speed must be > 0")

    target_max_speed_pool = None
    target_max_speed_ratio_pool = None
    if args.target_max_speed is not None:
        target_max_speed_pool = [float(args.target_max_speed)]
    if args.target_max_speed_ratio_choices is not None:
        target_max_speed_ratio_pool = _parse_float_choices(
            str(args.target_max_speed_ratio_choices),
            "--target_max_speed_ratio_choices",
        )
    if target_max_speed_pool is not None and target_max_speed_ratio_pool is not None:
        raise ValueError("Use only one of --target_max_speed and --target_max_speed_ratio_choices")
    if target_max_speed_pool is None and target_max_speed_ratio_pool is None:
        raise ValueError("Please provide --target_max_speed or --target_max_speed_ratio_choices")
    if target_max_speed_pool is not None and any(float(x) <= 0.0 for x in target_max_speed_pool):
        raise ValueError("--target_max_speed must be > 0")
    if target_max_speed_ratio_pool is not None and any(float(x) <= 0.0 for x in target_max_speed_ratio_pool):
        raise ValueError("--target_max_speed_ratio_choices values must be > 0")
    if float(args.target_hunter_zone_min_dis) < 0.0:
        raise ValueError("--target_hunter_zone_min_dis must be >= 0")

    # Step 2: 解析策略候选与巡逻路线候选
    policy_choices = [x.strip().lower() for x in str(args.target_policy_choices).split(",") if x.strip()]
    if len(policy_choices) == 0:
        raise ValueError("No valid policy in --target_policy_choices")
    patrol_route_pool = _resolve_patrol_route_pool(list(args.target_patrol_paths))
    if "patrol" in policy_choices and len(patrol_route_pool) == 0:
        raise ValueError("Patrol policy requested but no valid patrol route names found")
    hunters_in_zone_choices = _parse_bool_choices(str(args.hunters_in_zone_choices), "--hunters_in_zone_choices")
    target_avoid_hunter_zone = None
    if args.target_avoid_hunter_zone is not None:
        target_avoid_hunter_zone = _parse_bool_value(
            str(args.target_avoid_hunter_zone),
            "--target_avoid_hunter_zone",
        )

    # Step 3: 先采样基础环境，再与hunter数量集合做笛卡尔组合
    rng = random.Random(int(args.rand_seed))
    base_specs: list[dict] = []

    if bool(args.factorial):
        world_values = world_size_pool if world_size_pool is not None else [None]
        capture_values = capture_dis_pool if capture_dis_pool is not None else [None]
        capture_ratio_values = capture_dis_ratio_pool if capture_dis_ratio_pool is not None else [None]
        capture_step_values = capture_step_pool if capture_step_pool is not None else [None]
        target_speed_values = target_max_speed_pool if target_max_speed_pool is not None else [None]
        target_speed_ratio_values = (
            target_max_speed_ratio_pool if target_max_speed_ratio_pool is not None else [None]
        )
        factor_rows = list(
            itertools.product(
                world_values,
                capture_values,
                capture_ratio_values,
                capture_step_values,
                hunter_max_speed_pool,
                target_speed_values,
                target_speed_ratio_values,
                policy_choices,
                hunters_in_zone_choices,
            )
        )
    else:
        factor_rows = []
        for _ in range(int(num_base_envs)):
            sampled_world_size = None
            if world_size_range is not None:
                sampled_world_size = float(rng.uniform(float(world_size_range[0]), float(world_size_range[1])))
            elif world_size_pool is not None:
                sampled_world_size = float(rng.choice(world_size_pool))
            factor_rows.append(
                (
                    sampled_world_size,
                    None if capture_dis_pool is None else float(rng.choice(capture_dis_pool)),
                    None if capture_dis_ratio_pool is None else float(rng.choice(capture_dis_ratio_pool)),
                    None if capture_step_pool is None else int(rng.choice(capture_step_pool)),
                    float(rng.choice(hunter_max_speed_pool)),
                    None if target_max_speed_pool is None else float(rng.choice(target_max_speed_pool)),
                    None if target_max_speed_ratio_pool is None else float(rng.choice(target_max_speed_ratio_pool)),
                    str(rng.choice(policy_choices)),
                    bool(rng.choice(hunters_in_zone_choices)),
                )
            )

    for idx, row in enumerate(factor_rows):
        (
            world_size,
            capture_dis,
            capture_dis_ratio,
            capture_step,
            hunter_max_speed,
            target_max_speed,
            target_max_speed_ratio,
            policy,
            hunters_in_zone,
        ) = row
        route_path, route_name = rng.choice(patrol_route_pool)
        resolved_capture_dis = (
            float(capture_dis)
            if capture_dis is not None
            else float(float(hunter_max_speed) * float(capture_dis_ratio))
        )
        base_spec = {
            "seed": int(int(args.seed_start) + idx * int(args.seed_step)),
            "capture_dis": float(resolved_capture_dis),
            "collision_dis": float(args.collision_dis),
            "hunter_safe_dis": float(args.hunter_safe_dis),
            "hunter_max_speed": float(hunter_max_speed),
            "target_policy_source": str(policy),
            "target_patrol_path": str(route_path),
            "target_route_id": int(args.target_route_id),
            "target_patrol_names": [str(route_name)],
            "hunters_in_zone": bool(hunters_in_zone),
            "target_hunter_zone_min_dis": float(args.target_hunter_zone_min_dis),
        }
        if target_avoid_hunter_zone is not None:
            base_spec["target_avoid_hunter_zone"] = bool(target_avoid_hunter_zone)
        if capture_dis_ratio is not None:
            base_spec["capture_dis_ratio"] = float(capture_dis_ratio)
        if capture_step is not None:
            base_spec["capture_step"] = int(capture_step)
        if target_max_speed is not None:
            base_spec["target_max_speed"] = float(target_max_speed)
        if target_max_speed_ratio is not None:
            base_spec["target_max_speed_ratio"] = float(target_max_speed_ratio)
        if world_size is not None:
            base_spec["world_size"] = float(world_size)
        base_specs.append(base_spec)

    tasks: list[dict] = []
    for base_spec in base_specs:
        for num_hunters in hunter_count_choices:
            task = dict(base_spec)
            task["num_hunters"] = int(num_hunters)
            tasks.append(task)
    return tasks


def main() -> None:
    """
    功能:
        程序入口：生成任务并写入JSON文件。
    输入:
        无（从CLI读取）。
    输出:
        无。
    """
    # Step 1: 读取参数并生成任务列表
    args = parse_args()
    tasks = build_tasks(args)

    # Step 2: 写出JSON文件（兼容train.py中带tasks字段的格式）
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() in [".yaml", ".yml"]:
        _dump_grouped_yaml(args.output, tasks)
    else:
        payload = {"tasks": tasks}
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # Step 3: 打印生成摘要
    print(
        "Generated {} fixed eval tasks to {} (num_base_envs={}, hunter_count_choices={}, world_size={}, factorial={}).".format(
            int(len(tasks)),
            str(args.output),
            int(_resolve_num_base_envs(args)),
            _resolve_hunter_count_choices(args),
            (
                "disabled"
                if args.world_size is None and args.world_size_choices is None
                else (
                    str(args.world_size_choices)
                    if args.world_size_choices is not None
                    else f"[{float(args.world_size[0])}, {float(args.world_size[1])}]"
                )
            ),
            bool(args.factorial),
        )
    )


def _dump_grouped_yaml(output_path: Path, tasks: list[dict]) -> None:
    """
    功能:
        将任务按num_hunters分组并以紧凑对齐风格写入YAML，便于人工检查。
    输入:
        output_path (Path): 输出YAML路径。
        tasks (list[dict]): 任务列表。
    输出:
        无。
    """
    # Step 1: 按num_hunters分组并组内按seed排序
    grouped = defaultdict(list)
    for task in tasks:
        grouped[int(task["num_hunters"])].append(dict(task))
    for hunters in grouped:
        grouped[hunters] = sorted(grouped[hunters], key=lambda x: int(x.get("seed", 0)))

    # Step 2: 统一字段顺序，构建可读性更强的inline映射行
    preferred_order = [
        "num_hunters",
        "hunters_in_zone",
        "world_size",
        "seed",
        "capture_dis",
        "capture_dis_ratio",
        "capture_step",
        "collision_dis",
        "hunter_safe_dis",
        "hunter_max_speed",
        "target_max_speed",
        "target_max_speed_ratio",
        "target_avoid_hunter_zone",
        "target_hunter_zone_min_dis",
        "target_policy_source",
        "target_patrol_path",
        "target_patrol_names",
        "target_route_id",
    ]
    lines = ["tasks:"]
    for group_id, hunters in enumerate(sorted(grouped.keys())):
        if group_id > 0:
            lines.append("")
        lines.append(f"  # num_hunters = {int(hunters)}")
        for task in grouped[hunters]:
            kv_parts = []
            for key in preferred_order:
                if key not in task:
                    continue
                value = task[key]
                if isinstance(value, str):
                    value_repr = value
                else:
                    value_repr = json.dumps(value, ensure_ascii=False)
                kv_parts.append(f"{key}: {value_repr}")
            lines.append("  - {" + ", ".join(kv_parts) + "}")

    # Step 3: 写入文件
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
