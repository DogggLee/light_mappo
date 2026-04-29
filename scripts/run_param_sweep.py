"""
Run parameter sweep experiments for Multi-UAV Pursuit training.

Example sweep YAML:

base_config: config/v1/base.yaml
exp_name: base_ablation
max_parallel: 2
python: python
train_script: train/train.py
time_stat: false
final_test_eval: async        # Choices: off/sync/async
final_test_eval_model_glob: best_eval_capture_rate

parameters:
  env.action_frame:
    values: [global, local]
  reward.base_reward_mode:
    values: [legacy, delta_window]
  reward.spread_reward_enable:
    values: [false, true]
  optim.lr:
    values: [0.0005, 0.001]
  schedule.use_linear_lr_decay:
    values: [false, true]
  domain_randomization.train_split.enable:
    values: [false, true]

auto_reduce_redundant_by_base_reward_mode: true
# 或者使用通用条件规则（支持*前缀通配）
# conditional_ignore:
#   - when:
#       reward.base_reward_mode: legacy
#     ignore:
#       - reward.base_delta_*
#   - when:
#       reward.base_reward_mode: delta_window
#     ignore:
#       - reward.base_far_scale
#       - reward.base_near_scale
#
plot:
  enabled: true
  buckets: [fixed_zone_false, fixed_zone_true]
  metrics: [capture_rate, alive_rate, eval_reward]
"""

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ALIASES = {
    "reward.mi_deversity_enable": "reward.mi_diversity_enable",
    "reward.speed_panalty": "reward.speed_penalty",
}

NAME_ALIASES = {
    "env.action_frame": "acF",
    "env.obs_mode": "obs",
    "env.own_obs_items": "own_obs",
    "env.hunter_collision_state_mode": "colli",
    
    "reward.base_reward_mode": "base",
    "reward.capture_reward_allocation": "capAlloc",

    "reward.mi_diversity_enable": "mi",
    "reward.mi_diversity_coef": "miC",

    "reward.speed_penalty": "spdPe",

    "reward.spread_reward_enable": "spread",
    "reward.spread_reward_coef": "sprC",

    "optim.lr": "lr",

    "schedule.use_linear_lr_decay": "lrDec",
    
    "domain_randomization.train_split.enable": "dyEnv",
    "curriculum.enable": "curriculum"
}


def _load_yaml(path):
    """
    功能:
        读取YAML文件并返回普通dict。
    输入:
        path (str | Path): YAML文件路径。
    输出:
        dict: YAML内容；空文件返回空dict。
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dump_yaml(data, path):
    """
    功能:
        将dict写入YAML文件。
    输入:
        data (dict): 待写入配置。
        path (str | Path): 输出路径。
    输出:
        无。
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _canonical_param_path(path):
    """
    功能:
        将参数路径规范化，并兼容常见拼写别名。
    输入:
        path (str): 点号分隔参数路径。
    输出:
        str: 规范参数路径。
    """
    key = str(path).strip()
    return ALIASES.get(key, key)


def _set_by_path(data, path, value):
    """
    功能:
        按点号路径设置嵌套dict中的参数值。
    输入:
        data (dict): 待修改配置。
        path (str): 点号分隔路径，例如reward.speed_penalty。
        value (object): 新参数值。
    输出:
        无。
    """
    keys = str(path).split(".")
    cur = data
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            raise KeyError("Config path does not exist or is not a dict: {}".format(str(path)))
        cur = cur[key]
    last = keys[-1]
    if last not in cur:
        raise KeyError("Config key does not exist: {}".format(str(path)))
    cur[last] = value


def _set_by_path_if_exists(data, path, value):
    """
    功能:
        当点号路径存在时写入值；不存在时跳过。
    输入:
        data (dict): 待修改配置。
        path (str): 点号分隔路径。
        value (object): 新参数值。
    输出:
        bool: True表示成功写入，False表示路径不存在。
    """
    keys = str(path).split(".")
    cur = data
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            return False
        cur = cur[key]
    last = keys[-1]
    if last not in cur:
        return False
    cur[last] = value
    return True


def _expand_values(spec):
    """
    功能:
        将参数取值规格展开为列表。
    输入:
        spec (dict | list): 支持values/range/linspace或直接list。
    输出:
        list: 参数取值列表。
    """
    if isinstance(spec, list):
        return list(spec)
    if not isinstance(spec, dict):
        return [spec]
    if "values" in spec:
        return list(spec["values"])
    if "range" in spec:
        r = dict(spec["range"])
        start = float(r["start"])
        stop = float(r["stop"])
        step = float(r["step"])
        if step == 0.0:
            raise ValueError("range.step cannot be 0")
        vals = []
        cur = start
        if step > 0:
            while cur <= stop + 1e-12:
                vals.append(_maybe_int(cur))
                cur += step
        else:
            while cur >= stop - 1e-12:
                vals.append(_maybe_int(cur))
                cur += step
        return vals
    if "linspace" in spec:
        r = dict(spec["linspace"])
        start = float(r["start"])
        stop = float(r["stop"])
        num = int(r["num"])
        if num <= 1:
            return [_maybe_int(start)]
        step = (stop - start) / float(num - 1)
        return [_maybe_int(start + step * i) for i in range(num)]
    raise ValueError("Parameter spec must contain values, range, or linspace: {}".format(spec))


def _maybe_int(value):
    """
    功能:
        将数值中接近整数的float转为int，保持YAML更简洁。
    输入:
        value (float): 数值。
    输出:
        int | float: 规范化数值。
    """
    val = float(value)
    if abs(val - round(val)) <= 1e-12:
        return int(round(val))
    return val


def _value_to_name(value):
    """
    功能:
        将参数值转成适合作为实验名片段的字符串。
    输入:
        value (object): 参数值。
    输出:
        str: 参数值字符串。
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    text = str(value)
    text = text.replace("/", "_").replace("\\", "_").replace(" ", "")
    return text


def _sanitize_name_token(text):
    """
    功能:
        将实验名片段清洗为文件名安全字符串。
    输入:
        text (str): 原始文本。
    输出:
        str: 清洗后的字符串。
    """
    return re.sub(r"[^0-9A-Za-z_.+=,-]+", "_", str(text))


def _own_obs_items_alias(value):
    """
    功能:
        为env.own_obs_items提供更紧凑的实验名别名。
    输入:
        value (object): own_obs_items参数值。
    输出:
        str | None: 若命中特定组合则返回别名，否则返回None。
    """
    if not isinstance(value, list):
        return None
    if value == ["global_pos", "global_vel"]:
        return "g_pv"
    if value == ["speed_norm"]:
        return "spdN"
    if value == ["speed_norm", "boundary_dist_norm"]:
        return "spdN_bd"
    return None


def _format_param_name_fragment(path, value):
    """
    功能:
        将单个参数格式化为实验名片段；必要时返回None表示忽略。
    输入:
        path (str): 参数路径。
        value (object): 参数值。
    输出:
        str | None: 格式化后的片段。
    """
    if str(path) == "env.own_obs_items":
        own_alias = _own_obs_items_alias(value)
        if own_alias is not None:
            return own_alias

    safe_path = NAME_ALIASES.get(str(path), str(path).split(".")[-1])
    safe_path = _sanitize_name_token(str(safe_path).replace("/", "_").replace("\\", "_"))
    if isinstance(value, bool):
        return safe_path if bool(value) else None
    return "{}-{}".format(str(safe_path), _sanitize_name_token(_value_to_name(value)))


def _build_experiment_name(combo):
    """
    功能:
        根据参数组合构造experiment_name。
    输入:
        combo (list[tuple[str, object]]): 参数路径和值列表。
    输出:
        str: 形如param-value + param-value的实验名。
    """
    combo_map = {str(path): value for path, value in combo}
    skip_keys = set()
    parts = []

    if (
        bool(combo_map.get("reward.mi_diversity_enable", False))
        and "reward.mi_diversity_coef" in combo_map
    ):
        parts.append("mi-{}".format(_sanitize_name_token(_value_to_name(combo_map["reward.mi_diversity_coef"]))))
        skip_keys.add("reward.mi_diversity_enable")
        skip_keys.add("reward.mi_diversity_coef")
    if (
        bool(combo_map.get("reward.spread_reward_enable", False))
        and "reward.spread_reward_coef" in combo_map
    ):
        parts.append("spd-{}".format(_sanitize_name_token(_value_to_name(combo_map["reward.spread_reward_coef"]))))
        skip_keys.add("reward.spread_reward_enable")
        skip_keys.add("reward.spread_reward_coef")

    for path, value in combo:
        if str(path) in skip_keys:
            continue
        fragment = _format_param_name_fragment(path, value)
        if fragment is None or str(fragment) == "":
            continue
        parts.append(str(fragment))

    if len(parts) == 0:
        return "base"
    return " + ".join(parts)


def _build_combinations(parameters):
    """
    功能:
        将参数空间展开为笛卡尔积组合。
    输入:
        parameters (dict): 参数路径到取值规格的映射。
    输出:
        list[list[tuple[str, object]]]: 参数组合列表。
    """
    param_items = []
    for raw_path, spec in parameters.items():
        path = _canonical_param_path(raw_path)
        values = _expand_values(spec)
        if len(values) == 0:
            raise ValueError("Parameter has no values: {}".format(path))
        param_items.append((path, values))
    paths = [x[0] for x in param_items]
    value_lists = [x[1] for x in param_items]
    combos = []
    for values in itertools.product(*value_lists):
        combos.append(list(zip(paths, values)))
    return combos


def _normalize_conditional_ignore_rules(sweep_cfg):
    """
    功能:
        规范化条件忽略规则，并按配置拼接内置冗余规则。
    输入:
        sweep_cfg (dict): sweep配置。
    输出:
        list[dict]: 规则列表，每条含when/ignore。
    """
    rules = []
    user_rules = list(sweep_cfg.get("conditional_ignore", []))
    for rule in user_rules:
        if not isinstance(rule, dict):
            raise ValueError("Each conditional_ignore rule must be a dict.")
        when = dict(rule.get("when", {}))
        ignore = list(rule.get("ignore", []))
        if len(when) == 0 or len(ignore) == 0:
            raise ValueError("Each conditional_ignore rule requires non-empty 'when' and 'ignore'.")
        rules.append(
            {
                "when": { _canonical_param_path(k): v for k, v in when.items() },
                "ignore": [ _canonical_param_path(x) for x in ignore ],
            }
        )

    if bool(sweep_cfg.get("auto_reduce_redundant_by_base_reward_mode", False)):
        rules.extend(
            [
                {
                    "when": {"reward.base_reward_mode": "legacy"},
                    "ignore": ["reward.base_delta_*"],
                },
                {
                    "when": {"reward.base_reward_mode": "delta_window"},
                    "ignore": ["reward.base_far_scale", "reward.base_near_scale"],
                },
            ]
        )
    return rules


def _prune_redundant_combinations(combos, rules):
    """
    功能:
        根据条件忽略规则裁剪冗余参数组合，并去重。
    输入:
        combos (list[list[tuple[str, object]]]): 原始笛卡尔积参数组合。
        rules (list[dict]): 条件忽略规则。
    输出:
        list[list[tuple[str, object]]]: 裁剪去重后的组合。
    """
    if len(rules) == 0:
        return combos

    unique = {}
    for combo in combos:
        combo_map = {str(k): v for k, v in combo}
        effective = dict(combo_map)

        for rule in rules:
            cond = dict(rule["when"])
            matched = True
            for key, expected in cond.items():
                if key not in combo_map or combo_map[key] != expected:
                    matched = False
                    break
            if not matched:
                continue

            for ignore_key in list(rule["ignore"]):
                ig = str(ignore_key)
                if ig.endswith("*"):
                    prefix = ig[:-1]
                    for k in list(effective.keys()):
                        if str(k).startswith(prefix):
                            effective.pop(k, None)
                else:
                    effective.pop(ig, None)

        eff_items = sorted(effective.items(), key=lambda x: str(x[0]))
        key = tuple((k, json.dumps(v, ensure_ascii=False, sort_keys=True)) for k, v in eff_items)
        if key not in unique:
            unique[key] = [(k, v) for k, v in eff_items]

    return list(unique.values())


def _prepare_experiments(sweep_cfg, sweep_path):
    """
    功能:
        生成所有实验配置文件和manifest条目。
    输入:
        sweep_cfg (dict): sweep YAML内容。
        sweep_path (Path): sweep YAML路径。
    输出:
        tuple[Path, list[dict]]: sweep输出目录和实验manifest。
    """
    root = PROJECT_ROOT
    base_config_path = Path(str(sweep_cfg["base_config"]))
    if not base_config_path.is_absolute():
        base_config_path = root / base_config_path
    base_cfg = _load_yaml(base_config_path)

    output_dir = _resolve_output_dir(sweep_cfg)
    if output_dir.exists():
        raise FileExistsError(
            "Sweep output_dir already exists: {}. Please remove/rename it before running.".format(
                str(output_dir)
            )
        )
    configs_dir = output_dir / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    combos = _build_combinations(dict(sweep_cfg.get("parameters", {})))
    conditional_rules = _normalize_conditional_ignore_rules(sweep_cfg)
    combos = _prune_redundant_combinations(combos, conditional_rules)
    fixed_exp_prefix = sweep_cfg.get("env_name", None)
    manifest = []
    
    for idx, combo in enumerate(combos):
        cfg_i = json.loads(json.dumps(base_cfg))
        combo_name = _build_experiment_name(combo)
        exp_name = "exp_{:04d} + {}".format(int(idx), combo_name)
        _set_by_path(cfg_i, "exp.experiment_name", exp_name)
        if fixed_exp_prefix is not None:
            _set_by_path_if_exists(cfg_i, "env.env_name", str(fixed_exp_prefix))
        for path, value in combo:
            _set_by_path(cfg_i, path, value)
        cfg_path = configs_dir / "exp_{:04d}.yaml".format(idx)
        _dump_yaml(cfg_i, cfg_path)
        manifest.append(
            {
                "id": idx,
                "config_path": str(cfg_path),
                "experiment_name": exp_name,
                "params": {path: value for path, value in combo},
                "status": "pending",
                "returncode": None,
                "run_dir": "",
                "start_time": None,
                "end_time": None,
                "duration_sec": None,
            }
        )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sweep_file": str(sweep_path),
                "base_config": str(base_config_path),
                "experiments": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return output_dir, manifest


def _resolve_output_dir(sweep_cfg):
    """
    功能:
        从sweep配置解析输出目录。
    输入:
        sweep_cfg (dict): sweep YAML内容。
    输出:
        Path: 绝对输出目录。
    """
    root = PROJECT_ROOT
    base_config_path = Path(str(sweep_cfg["base_config"]))
    if not base_config_path.is_absolute():
        base_config_path = root / base_config_path
    base_cfg = _load_yaml(base_config_path)
    env_name = str(sweep_cfg.get("env_name", base_cfg["env"]["env_name"]))
    return root / "results" / str(env_name) / "sweeps"


def _expected_run_root(config_path):
    """
    功能:
        根据训练配置推断train.py输出的run根目录。
    输入:
        config_path (str | Path): 实验配置文件。
    输出:
        Path: results/env/algo/experiment_name目录。
    """
    from utils.util import load_config

    root = PROJECT_ROOT
    cfg = load_config(str(config_path))
    return root / "results" / str(cfg.env.env_name) / str(cfg.exp.algorithm_name) / str(cfg.exp.experiment_name)


def _latest_run_dir(config_path):
    """
    功能:
        查找某实验配置对应的最新run目录。
    输入:
        config_path (str | Path): 实验配置文件。
    输出:
        str: 最新run目录路径；不存在则为空字符串。
    """
    run_root = _expected_run_root(config_path)
    if not run_root.exists():
        return ""
    candidates = []
    for folder in run_root.iterdir():
        if not folder.is_dir() or not folder.name.startswith("run"):
            continue
        try:
            run_id = int(folder.name.replace("run", ""))
        except ValueError:
            continue
        candidates.append((run_id, folder))
    if len(candidates) == 0:
        return ""
    return str(sorted(candidates, key=lambda x: x[0])[-1][1])


def _write_leaderboard_file(output_dir, payload):
    """
    功能:
        将当前global best与topK结果写入输出目录。
    输入:
        output_dir (Path): sweep输出目录。
        payload (dict): 需写入的排行榜信息。
    输出:
        无。
    """
    out_path = Path(output_dir) / "leaderboard.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _run_experiments(sweep_cfg, output_dir, manifest):
    """
    功能:
        按max_parallel并行运行所有训练实验。
    输入:
        sweep_cfg (dict): sweep配置。
        output_dir (Path): sweep输出目录。
        manifest (list[dict]): 实验manifest条目。
    输出:
        list[dict]: 更新状态后的manifest。
    """
    root = PROJECT_ROOT
    max_parallel = int(sweep_cfg.get("max_parallel", 1))
    py_bin = str(sweep_cfg.get("python", "python"))
    train_script = str(sweep_cfg.get("train_script", "train/train.py"))
    time_stat = bool(sweep_cfg.get("time_stat", False))
    final_test_eval = str(sweep_cfg.get("final_test_eval", "off")).lower()
    final_test_eval_model_glob = str(
        sweep_cfg.get("final_test_eval_model_glob", "best_eval_capture_rate")
    )
    if final_test_eval not in ("off", "sync", "async"):
        raise ValueError("final_test_eval must be one of off/sync/async")
    progress_interval_sec = float(sweep_cfg.get("progress_interval_sec", 20.0))
    progress_topk = int(sweep_cfg.get("progress_topk", 5))
    logs_dir = output_dir / "launcher_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    def _tail_file_lines(path, n_lines):
        """
        功能:
            读取文件末尾若干行文本。
        输入:
            path (str): 文件路径。
            n_lines (int): 末尾行数。
        输出:
            list[str]: 末尾行列表；异常时返回空列表。
        """
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            return [str(x).rstrip("\n") for x in lines[-int(n_lines):]]
        except Exception:
            return []

    def _parse_float(row, key):
        """
        功能:
            从CSV行中解析指定字段为float。
        输入:
            row (dict): CSV行。
            key (str): 字段名。
        输出:
            float | None: 成功返回数值，失败返回None。
        """
        if key not in row:
            return None
        text = str(row.get(key, "")).strip()
        if text == "":
            return None
        try:
            return float(text)
        except Exception:
            return None

    def _read_run_progress(run_dir):
        """
        功能:
            读取单个run目录当前训练进度与最优捕获指标。
        输入:
            run_dir (str): run目录。
        输出:
            dict: 包含episode、当前指标与各指标最优值及其episode。
        """
        out = {
            "episode": None,
            "current_capture_rate": None,
            "current_capture_steps": None,
            "current_alive_rate": None,
            "best_capture_rate": None,
            "best_capture_rate_episode": None,
            "best_capture_steps": None,
            "best_capture_steps_episode": None,
            "best_alive_rate": None,
            "best_alive_rate_episode": None,
            "best_capture_steps_metric": None,
            "best_capture_steps_metric_episode": None,
            "best_alive_rate_metric": None,
            "best_alive_rate_metric_episode": None,
        }
        if not run_dir:
            return out

        run_path = Path(str(run_dir))
        eval_path = run_path / "eval.csv"
        log_path = run_path / "log.csv"

        # 优先从eval.csv读取指标，并按严格双桶口径聚合episode指标。
        if eval_path.exists():
            try:
                with open(eval_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    best_rate = None
                    best_rate_ep = None
                    best_steps = None
                    best_steps_ep = None
                    best_alive = None
                    best_alive_ep = None
                    best_steps_metric = None
                    best_steps_metric_ep = None
                    best_alive_metric = None
                    best_alive_metric_ep = None
                    ep_bucket_metrics = {}

                    def _ep_key(ep_val):
                        if ep_val is None:
                            return None
                        if abs(float(ep_val) - float(int(ep_val))) <= 1e-9:
                            return int(ep_val)
                        return float(ep_val)

                    def _mean_or_none(vals):
                        valid = [float(v) for v in vals if v is not None]
                        if len(valid) <= 0:
                            return None
                        return float(sum(valid) / float(len(valid)))

                    def _aggregate_ep_metrics(bucket_map):
                        # 严格口径：仅当fixed_zone_false与fixed_zone_true同时存在时才产出指标。
                        lower_map = {str(k).strip().lower(): v for k, v in bucket_map.items()}
                        dual_keys = ["fixed_zone_false", "fixed_zone_true"]
                        if not all(k in lower_map for k in dual_keys):
                            return None
                        rows = [lower_map[k] for k in dual_keys]
                        return {
                            "capture_rate": _mean_or_none([r.get("capture_rate", None) for r in rows]),
                            "capture_steps": _mean_or_none([r.get("capture_steps", None) for r in rows]),
                            "alive_rate": _mean_or_none([r.get("alive_rate", None) for r in rows]),
                        }

                    for row in reader:
                        ep = _parse_float(row, "episode")
                        bucket = str(row.get("bucket", ""))
                        cap_rate = _parse_float(row, "capture_rate")
                        cap_steps = _parse_float(row, "capture_steps")
                        alive_rate = _parse_float(row, "alive_rate")

                        if ep is not None:
                            ep_key = _ep_key(ep)
                            if ep_key not in ep_bucket_metrics:
                                ep_bucket_metrics[ep_key] = {}
                            ep_bucket_metrics[ep_key][str(bucket).strip().lower()] = {
                                "capture_rate": cap_rate,
                                "capture_steps": cap_steps,
                                "alive_rate": alive_rate,
                            }

                    # 按episode聚合fixed指标后，再计算当前与最优。
                    ep_keys = sorted(ep_bucket_metrics.keys(), key=lambda x: float(x))
                    agg_by_ep = []
                    for ep_key in ep_keys:
                        agg = _aggregate_ep_metrics(ep_bucket_metrics[ep_key])
                        if agg is None:
                            continue
                        agg_by_ep.append((ep_key, agg))

                    if len(agg_by_ep) > 0:
                        cur_ep, cur_metrics = agg_by_ep[-1]
                        out["episode"] = cur_ep
                        out["current_capture_rate"] = cur_metrics.get("capture_rate", None)
                        out["current_capture_steps"] = cur_metrics.get("capture_steps", None)
                        out["current_alive_rate"] = cur_metrics.get("alive_rate", None)

                    for ep_key, agg in agg_by_ep:
                        cap_rate = agg.get("capture_rate", None)
                        cap_steps = agg.get("capture_steps", None)
                        alive_rate = agg.get("alive_rate", None)

                        if cap_rate is not None:
                            if best_rate is None or float(cap_rate) > float(best_rate):
                                best_rate = float(cap_rate)
                                best_rate_ep = ep_key
                                best_steps = None if cap_steps is None else float(cap_steps)
                                best_steps_ep = ep_key
                                best_alive = None if alive_rate is None else float(alive_rate)
                                best_alive_ep = ep_key
                            elif float(cap_rate) == float(best_rate):
                                if cap_steps is not None and (best_steps is None or float(cap_steps) < float(best_steps)):
                                    best_steps = float(cap_steps)
                                    best_steps_ep = ep_key
                                    best_alive = None if alive_rate is None else float(alive_rate)
                                    best_alive_ep = ep_key
                                elif cap_steps is not None and best_steps is not None and float(cap_steps) == float(best_steps):
                                    if alive_rate is not None and (best_alive is None or float(alive_rate) > float(best_alive)):
                                        best_alive = float(alive_rate)
                                        best_alive_ep = ep_key

                        if cap_steps is not None:
                            if best_steps_metric is None or float(cap_steps) < float(best_steps_metric):
                                best_steps_metric = float(cap_steps)
                                best_steps_metric_ep = ep_key

                        if alive_rate is not None:
                            if best_alive_metric is None or float(alive_rate) > float(best_alive_metric):
                                best_alive_metric = float(alive_rate)
                                best_alive_metric_ep = ep_key

                    out["best_capture_rate"] = best_rate
                    out["best_capture_rate_episode"] = best_rate_ep
                    out["best_capture_steps"] = best_steps
                    out["best_capture_steps_episode"] = best_steps_ep
                    out["best_alive_rate"] = best_alive
                    out["best_alive_rate_episode"] = best_alive_ep
                    out["best_capture_steps_metric"] = best_steps_metric
                    out["best_capture_steps_metric_episode"] = best_steps_metric_ep
                    out["best_alive_rate_metric"] = best_alive_metric
                    out["best_alive_rate_metric_episode"] = best_alive_metric_ep
            except Exception:
                pass
        return out

    last_reported_episode_by_exp = {}

    def _has_new_strict_progress():
        """
        功能:
            检查运行中实验是否出现了新的“严格口径”评估episode。
        输入:
            无。
        输出:
            bool: True表示检测到新episode（且三项指标齐全）。
        """
        has_new = False
        for exp_item in manifest:
            if str(exp_item.get("status", "")) != "running":
                continue
            run_dir = str(exp_item.get("run_dir", ""))
            if run_dir == "":
                run_dir = _latest_run_dir(exp_item["config_path"])
                if run_dir != "":
                    exp_item["run_dir"] = str(run_dir)
            prog = _read_run_progress(run_dir)
            ep = prog.get("episode", None)
            cur_rate = prog.get("current_capture_rate", None)
            cur_steps = prog.get("current_capture_steps", None)
            cur_alive = prog.get("current_alive_rate", None)
            if ep is None or cur_rate is None or cur_steps is None or cur_alive is None:
                continue
            exp_id = int(exp_item.get("id", -1))
            ep_val = float(ep)
            prev_ep = last_reported_episode_by_exp.get(exp_id, None)
            if prev_ep is None or ep_val > float(prev_ep) + 1e-9:
                last_reported_episode_by_exp[exp_id] = ep_val
                has_new = True
        return bool(has_new)

    def _print_progress():
        """
        功能:
            打印当前sweep进度、失败数量与ETA估计。
        输入:
            无。
        输出:
            无。
        """
        total = len(manifest)
        done = 0
        failed = 0
        running_cnt = 0
        pending_cnt = 0
        durations = []
        for exp_item in manifest:
            st = str(exp_item.get("status", "pending"))
            if st == "done":
                done += 1
                dur = exp_item.get("duration_sec", None)
                if isinstance(dur, (int, float)):
                    durations.append(float(dur))
            elif st == "failed":
                failed += 1
                dur = exp_item.get("duration_sec", None)
                if isinstance(dur, (int, float)):
                    durations.append(float(dur))
            elif st == "running":
                running_cnt += 1
            else:
                pending_cnt += 1

        eta_text = "N/A"
        if len(durations) > 0 and max_parallel > 0:
            mean_dur = sum(durations) / float(len(durations))
            eta_sec = (pending_cnt / float(max_parallel)) * mean_dur
            eta_text = "{:.1f} min".format(eta_sec / 60.0)
        print(
            "[Sweep][Progress] total={}, done={}, failed={}, running={}, pending={}, eta={}".format(
                int(total), int(done), int(failed), int(running_cnt), int(pending_cnt), eta_text
            )
        )

        # 汇总当前实验进度，并构建全局TopK候选。
        scored_items = []
        for exp_item in manifest:
            st = str(exp_item.get("status", "pending"))
            if st not in ("running", "done", "failed"):
                continue
            run_dir = str(exp_item.get("run_dir", ""))
            if run_dir == "":
                run_dir = _latest_run_dir(exp_item["config_path"])
                if run_dir != "":
                    exp_item["run_dir"] = str(run_dir)
            prog = _read_run_progress(run_dir)
            rate = prog.get("best_capture_rate", None)
            steps = prog.get("best_capture_steps", None)
            alive = prog.get("best_alive_rate", None)
            if rate is None:
                continue
            steps_sort = float("inf") if steps is None else float(steps)
            alive_sort = float("-inf") if alive is None else float(alive)
            scored_items.append(
                {
                    "id": int(exp_item.get("id", -1)),
                    "experiment_name": str(exp_item.get("experiment_name", "")),
                    "capture_rate": float(rate),
                    "capture_steps": None if steps is None else float(steps),
                    "alive_rate": None if alive is None else float(alive),
                    "sort_key": (-float(rate), steps_sort, -alive_sort),
                }
            )

        scored_items = sorted(scored_items, key=lambda x: x["sort_key"])
        leaderboard_payload = {
            "progress": {
                "total": int(total),
                "done": int(done),
                "failed": int(failed),
                "running": int(running_cnt),
                "pending": int(pending_cnt),
                "eta": str(eta_text),
            },
            "global_best": None,
            "topk_best": [],
        }
        if len(scored_items) > 0:
            best = scored_items[0]
            best_steps_text = "NA" if best["capture_steps"] is None else "{:.2f}".format(float(best["capture_steps"]))
            best_alive_text = "NA" if best["alive_rate"] is None else "{:.4f}".format(float(best["alive_rate"]))
            leaderboard_payload["global_best"] = {
                "id": int(best["id"]),
                "experiment_name": str(best["experiment_name"]),
                "capture_rate": float(best["capture_rate"]),
                "capture_steps": None if best["capture_steps"] is None else float(best["capture_steps"]),
                "alive_rate": None if best["alive_rate"] is None else float(best["alive_rate"]),
            }
            print(
                "[Sweep][GlobalBest] exp_{:04d} name={} best_capture_rate={:.4f} best_capture_steps={} best_alive_rate={}".format(
                    int(best["id"]),
                    str(best["experiment_name"]),
                    float(best["capture_rate"]),
                    str(best_steps_text),
                    str(best_alive_text),
                )
            )
            topn = min(max(int(progress_topk), 1), len(scored_items))
            for rank, cand in enumerate(scored_items[:topn], start=1):
                cand_steps_text = "NA" if cand["capture_steps"] is None else "{:.2f}".format(float(cand["capture_steps"]))
                cand_alive_text = "NA" if cand["alive_rate"] is None else "{:.4f}".format(float(cand["alive_rate"]))
                leaderboard_payload["topk_best"].append(
                    {
                        "rank": int(rank),
                        "id": int(cand["id"]),
                        "experiment_name": str(cand["experiment_name"]),
                        "capture_rate": float(cand["capture_rate"]),
                        "capture_steps": None if cand["capture_steps"] is None else float(cand["capture_steps"]),
                        "alive_rate": None if cand["alive_rate"] is None else float(cand["alive_rate"]),
                    }
                )
                print(
                    "[Sweep][TopK] rank={} exp_{:04d} capture_rate={:.4f} capture_steps={} alive_rate={} name={}".format(
                        int(rank),
                        int(cand["id"]),
                        float(cand["capture_rate"]),
                        str(cand_steps_text),
                        str(cand_alive_text),
                        str(cand["experiment_name"]),
                    )
                )
        else:
            print("[Sweep][GlobalBest] exp_NA name=NA best_capture_rate=NA best_capture_steps=NA best_alive_rate=NA")
        _write_leaderboard_file(output_dir, leaderboard_payload)
        if failed > 0:
            failed_items = [x for x in manifest if str(x.get("status", "")) == "failed"]
            for exp_item in failed_items[:10]:
                print(
                    "[Sweep][Failed] exp_{:04d} returncode={} log={}".format(
                        int(exp_item["id"]),
                        int(exp_item.get("returncode", -1)),
                        str(exp_item.get("launcher_log", "")),
                    )
                )
        for exp_item in manifest:
            if str(exp_item.get("status", "")) != "running":
                continue
            run_dir = str(exp_item.get("run_dir", ""))
            if run_dir == "":
                run_dir = _latest_run_dir(exp_item["config_path"])
                if run_dir != "":
                    exp_item["run_dir"] = str(run_dir)
            prog = _read_run_progress(run_dir)
            ep = prog["episode"]
            cur_rate = prog.get("current_capture_rate", None)
            cur_steps = prog.get("current_capture_steps", None)
            cur_alive = prog.get("current_alive_rate", None)
            best_rate = prog["best_capture_rate"]
            best_rate_ep = prog.get("best_capture_rate_episode", None)
            best_steps = prog.get("best_capture_steps_metric", None)
            best_steps_ep = prog.get("best_capture_steps_metric_episode", None)
            best_alive = prog.get("best_alive_rate_metric", None)
            best_alive_ep = prog.get("best_alive_rate_metric_episode", None)
            ep_text = "NA" if ep is None else str(int(ep)) if abs(ep - int(ep)) <= 1e-9 else "{:.2f}".format(ep)
            rate_text = (
                "NA/NA(ep=NA)"
                if (cur_rate is None and best_rate is None)
                else "{}/{}(ep={})".format(
                    "NA" if cur_rate is None else "{:.4f}".format(float(cur_rate)),
                    "NA" if best_rate is None else "{:.4f}".format(float(best_rate)),
                    "NA" if best_rate_ep is None else str(int(best_rate_ep)),
                )
            )
            steps_text = (
                "NA/NA(ep=NA)"
                if (cur_steps is None and best_steps is None)
                else "{}/{}(ep={})".format(
                    "NA" if cur_steps is None else "{:.2f}".format(float(cur_steps)),
                    "NA" if best_steps is None else "{:.2f}".format(float(best_steps)),
                    "NA" if best_steps_ep is None else str(int(best_steps_ep)),
                )
            )
            alive_text = (
                "NA/NA(ep=NA)"
                if (cur_alive is None and best_alive is None)
                else "{}/{}(ep={})".format(
                    "NA" if cur_alive is None else "{:.4f}".format(float(cur_alive)),
                    "NA" if best_alive is None else "{:.4f}".format(float(best_alive)),
                    "NA" if best_alive_ep is None else str(int(best_alive_ep)),
                )
            )
            print(
                "[Sweep][Running] exp_{:04d} episode={} capture_rate(cur/best@ep)={} capture_steps(cur/best@ep)={} alive_rate(cur/best@ep)={}".format(
                    int(exp_item["id"]),
                    ep_text,
                    rate_text,
                    steps_text,
                    alive_text,
                )
            )

    pending = list(manifest)
    running = []
    last_scan_ts = 0.0
    while len(pending) > 0 or len(running) > 0:
        while len(pending) > 0 and len(running) < max_parallel:
            item = pending.pop(0)
            log_path = logs_dir / "exp_{:04d}.log".format(int(item["id"]))
            cmd = [py_bin, train_script, "--config_file", item["config_path"]]
            if time_stat:
                cmd.append("--time_stat")
            if final_test_eval != "off":
                cmd.extend(["--final_test_eval", final_test_eval])
                cmd.extend(["--final_test_eval_model_glob", final_test_eval_model_glob])
            log_f = open(log_path, "w", encoding="utf-8", buffering=1)
            proc = subprocess.Popen(
                cmd,
                cwd=str(root),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            item["status"] = "running"
            item["start_time"] = float(time.time())
            item["launcher_log"] = str(log_path)
            running.append({"item": item, "proc": proc, "log_f": log_f})
            print("[Sweep] started exp_{:04d}: {}".format(int(item["id"]), item["experiment_name"]))

        still_running = []
        for slot in running:
            proc = slot["proc"]
            ret = proc.poll()
            if ret is None:
                still_running.append(slot)
                continue
            slot["log_f"].close()
            item = slot["item"]
            item["end_time"] = float(time.time())
            if isinstance(item.get("start_time", None), (int, float)):
                item["duration_sec"] = float(item["end_time"] - item["start_time"])
            item["returncode"] = int(ret)
            item["status"] = "done" if int(ret) == 0 else "failed"
            item["run_dir"] = _latest_run_dir(item["config_path"])
            print(
                "[Sweep] finished exp_{:04d}: status={}, returncode={}, run_dir={}".format(
                    int(item["id"]),
                    item["status"],
                    int(ret),
                    str(item["run_dir"]),
                )
            )
            if item["status"] == "failed":
                tail_lines = _tail_file_lines(item.get("launcher_log", ""), 20)
                if len(tail_lines) > 0:
                    print("[Sweep][FailedSummary] exp_{:04d} log tail (last 20 lines):".format(int(item["id"])))
                    for line in tail_lines:
                        print("[Sweep][FailedSummary] {}".format(str(line)))
        running = still_running
        _write_manifest(output_dir, manifest)
        now_ts = float(time.time())
        if (now_ts - last_scan_ts) >= progress_interval_sec:
            last_scan_ts = now_ts
            if _has_new_strict_progress():
                _print_progress()
        if len(pending) > 0 or len(running) > 0:
            time.sleep(5.0)
    _print_progress()
    _write_manifest(output_dir, manifest)
    return manifest


def _write_manifest(output_dir, manifest):
    """
    功能:
        将当前manifest写回磁盘。
    输入:
        output_dir (Path): sweep输出目录。
        manifest (list[dict]): 实验manifest。
    输出:
        无。
    """
    with open(Path(output_dir) / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"experiments": manifest}, f, ensure_ascii=False, indent=2)


def _read_result_rows(run_dir, buckets, result_file):
    """
    功能:
        从run目录读取指定结果CSV中的bucket数据。
    输入:
        run_dir (str | Path): 训练run目录。
        buckets (list[str]): 需要读取的评估bucket。
        result_file (str): 结果CSV文件名。
    输出:
        list[dict]: 结果CSV行数据。
    """
    result_path = Path(run_dir) / str(result_file)
    if not result_path.exists():
        return []
    rows = []
    bucket_set = set(str(x) for x in buckets)
    with open(result_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("bucket", "")) not in bucket_set:
                continue
            rows.append(dict(row))
    return rows


def _write_summary_csv(output_dir, manifest, buckets, metrics, result_file, output_name):
    """
    功能:
        汇总所有实验结果到一个CSV。
    输入:
        output_dir (Path): sweep输出目录。
        manifest (list[dict]): 实验manifest。
        buckets (list[str]): 评估bucket列表。
        metrics (list[str]): 指标列表。
        result_file (str): 结果CSV文件名。
        output_name (str): 汇总CSV输出文件名。
    输出:
        Path: 汇总CSV路径。
    """
    out_path = Path(output_dir) / str(output_name)
    param_keys = sorted({k for item in manifest for k in item.get("params", {}).keys()})
    fieldnames = [
        "id",
        "experiment_name",
        "status",
        "run_dir",
        "model_name",
        "model_dir",
        "bucket",
        "episode",
        "total_num_steps",
    ] + param_keys + list(metrics)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in manifest:
            rows = _read_result_rows(
                item.get("run_dir", ""),
                buckets,
                result_file=result_file,
            )
            for row in rows:
                out = {
                    "id": item["id"],
                    "experiment_name": item["experiment_name"],
                    "status": item["status"],
                    "run_dir": item.get("run_dir", ""),
                    "model_name": row.get("model_name", ""),
                    "model_dir": row.get("model_dir", ""),
                    "bucket": row.get("bucket", ""),
                    "episode": row.get("episode", ""),
                    "total_num_steps": row.get("total_num_steps", ""),
                }
                for key in param_keys:
                    out[key] = item.get("params", {}).get(key, "")
                for metric in metrics:
                    out[metric] = row.get(metric, "")
                writer.writerow(out)
    return out_path


def _plot_results(output_dir, manifest, plot_cfg):
    """
    功能:
        绘制单参数变化的性能曲线；其他参数完全相同的实验分为同一组。
    输入:
        output_dir (Path): sweep输出目录。
        manifest (list[dict]): 实验manifest。
        plot_cfg (dict): 绘图配置。
    输出:
        无。
    """
    if not bool(plot_cfg.get("enabled", True)):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets = list(plot_cfg.get("buckets", ["fixed_zone_false", "fixed_zone_true", "fixed"]))
    metrics = list(plot_cfg.get("metrics", ["capture_rate", "alive_rate", "eval_reward"]))
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    param_keys = sorted({k for item in manifest for k in item.get("params", {}).keys()})
    result_file = str(plot_cfg.get("result_file", "eval.csv"))

    for bucket in buckets:
        for metric in metrics:
            for varied_param in param_keys:
                groups = {}
                for item in manifest:
                    params = dict(item.get("params", {}))
                    if varied_param not in params:
                        continue
                    other_key = tuple((k, _value_to_name(params.get(k, ""))) for k in param_keys if k != varied_param)
                    groups.setdefault(other_key, []).append(item)

                for group_idx, (_, items) in enumerate(sorted(groups.items(), key=lambda x: str(x[0]))):
                    if len(items) <= 1:
                        continue
                    plt.figure(figsize=(8.0, 4.8), dpi=130)
                    plotted = False
                    for item in sorted(items, key=lambda it: _value_to_name(it["params"][varied_param])):
                        rows = _read_result_rows(
                            item.get("run_dir", ""),
                            [bucket],
                            result_file=result_file,
                        )
                        xs = []
                        ys = []
                        for row in rows:
                            try:
                                xs.append(float(row["total_num_steps"]))
                                ys.append(float(row[metric]))
                            except Exception:
                                continue
                        if len(xs) == 0:
                            continue
                        label = "{}={}".format(varied_param, _value_to_name(item["params"][varied_param]))
                        plt.plot(xs, ys, marker="o", linewidth=1.6, markersize=3.5, label=label)
                        plotted = True
                    if not plotted:
                        plt.close()
                        continue
                    plt.title("{} | {} | vary {}".format(bucket, metric, varied_param))
                    plt.xlabel("total_num_steps")
                    plt.ylabel(metric)
                    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)
                    plt.legend(loc="best", fontsize=8)
                    plt.tight_layout()
                    file_name = "{}__{}__{}__group{:03d}.png".format(
                        _slug(bucket),
                        _slug(metric),
                        _slug(varied_param),
                        int(group_idx),
                    )
                    plt.savefig(str(plots_dir / file_name))
                    plt.close()


def _slug(text):
    """
    功能:
        将字符串转为适合文件名的slug。
    输入:
        text (str): 原字符串。
    输出:
        str: 文件名安全字符串。
    """
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text))


def main():
    """
    功能:
        解析命令行参数并执行sweep生成、训练和绘图。
    输入:
        无。
    输出:
        无。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_file", required=True, help="Path to sweep YAML.")
    parser.add_argument("--generate_only", action="store_true", help="Only generate configs and manifest.")
    parser.add_argument("--plot_only", action="store_true", help="Only plot from existing manifest/run outputs.")
    args = parser.parse_args()

    sweep_path = Path(args.sweep_file)
    sweep_cfg = _load_yaml(sweep_path)
    plot_cfg = dict(sweep_cfg.get("plot", {}))
    buckets = list(plot_cfg.get("buckets", ["fixed_zone_false", "fixed_zone_true", "fixed"]))
    metrics = list(plot_cfg.get("metrics", ["capture_rate", "alive_rate", "eval_reward"]))

    if args.plot_only:
        output_dir = _resolve_output_dir(sweep_cfg)
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("manifest.json not found: {}".format(str(manifest_path)))
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = list(data.get("experiments", []))
        _write_summary_csv(
            output_dir,
            manifest,
            buckets,
            metrics,
            result_file="eval.csv",
            output_name="summary.csv",
        )
        _write_summary_csv(
            output_dir,
            manifest,
            buckets,
            metrics,
            result_file="test_eval.csv",
            output_name="test_summary.csv",
        )
        _plot_results(output_dir, manifest, plot_cfg)
        print("[Sweep] plots written to {}".format(str(output_dir / "plots")))
        return

    output_dir, manifest = _prepare_experiments(sweep_cfg, sweep_path)

    if args.generate_only:
        print("[Sweep] generated {} configs under {}".format(len(manifest), str(output_dir)))
        return

    manifest = _run_experiments(sweep_cfg, output_dir, manifest)
    summary_path = _write_summary_csv(
        output_dir,
        manifest,
        buckets,
        metrics,
        result_file="eval.csv",
        output_name="summary.csv",
    )
    test_summary_path = _write_summary_csv(
        output_dir,
        manifest,
        buckets,
        metrics,
        result_file="test_eval.csv",
        output_name="test_summary.csv",
    )
    _plot_results(output_dir, manifest, plot_cfg)
    print("[Sweep] summary written to {}".format(str(summary_path)))
    print("[Sweep] test summary written to {}".format(str(test_summary_path)))
    print("[Sweep] plots written to {}".format(str(output_dir / "plots")))


if __name__ == "__main__":
    main()
