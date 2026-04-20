"""
Run parameter sweep experiments for Multi-UAV Pursuit training.

Example sweep YAML:

base_config: config/v1/base.yaml
output_dir: results/sweeps/base_ablation
max_parallel: 2
python: python
train_script: train/train.py
time_stat: false

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
import time
from pathlib import Path

import yaml


ALIASES = {
    "reward.mi_deversity_enable": "reward.mi_diversity_enable",
    "reward.speed_panalty": "reward.speed_penalty",
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


def _build_experiment_name(combo):
    """
    功能:
        根据参数组合构造experiment_name。
    输入:
        combo (list[tuple[str, object]]): 参数路径和值列表。
    输出:
        str: 形如param-value + param-value的实验名。
    """
    parts = []
    for path, value in combo:
        safe_path = str(path).replace("/", "_").replace("\\", "_")
        parts.append("{}-{}".format(safe_path, _value_to_name(value)))
    name = " + ".join(parts)
    return re.sub(r"[^0-9A-Za-z_.+= -]+", "_", name)


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
    root = Path(__file__).resolve().parents[1]
    base_config_path = Path(str(sweep_cfg["base_config"]))
    if not base_config_path.is_absolute():
        base_config_path = root / base_config_path
    base_cfg = _load_yaml(base_config_path)

    output_dir = Path(str(sweep_cfg.get("output_dir", "results/sweeps/default")))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    configs_dir = output_dir / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    combos = _build_combinations(dict(sweep_cfg.get("parameters", {})))
    manifest = []
    for idx, combo in enumerate(combos):
        cfg_i = json.loads(json.dumps(base_cfg))
        exp_name = _build_experiment_name(combo)
        _set_by_path(cfg_i, "exp.experiment_name", exp_name)
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
    root = Path(__file__).resolve().parents[1]
    output_dir = Path(str(sweep_cfg.get("output_dir", "results/sweeps/default")))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    return output_dir


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

    root = Path(__file__).resolve().parents[1]
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
    root = Path(__file__).resolve().parents[1]
    max_parallel = int(sweep_cfg.get("max_parallel", 1))
    py_bin = str(sweep_cfg.get("python", "python"))
    train_script = str(sweep_cfg.get("train_script", "train/train.py"))
    time_stat = bool(sweep_cfg.get("time_stat", False))
    logs_dir = output_dir / "launcher_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    pending = list(manifest)
    running = []
    while len(pending) > 0 or len(running) > 0:
        while len(pending) > 0 and len(running) < max_parallel:
            item = pending.pop(0)
            log_path = logs_dir / "exp_{:04d}.log".format(int(item["id"]))
            cmd = [py_bin, train_script, "--config_file", item["config_path"]]
            if time_stat:
                cmd.append("--time_stat")
            log_f = open(log_path, "w", encoding="utf-8", buffering=1)
            proc = subprocess.Popen(
                cmd,
                cwd=str(root),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            item["status"] = "running"
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
        running = still_running
        _write_manifest(output_dir, manifest)
        if len(pending) > 0 or len(running) > 0:
            time.sleep(5.0)
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


def _read_eval_rows(run_dir, buckets):
    """
    功能:
        从run目录读取eval.csv指定bucket的曲线数据。
    输入:
        run_dir (str | Path): 训练run目录。
        buckets (list[str]): 需要读取的评估bucket。
    输出:
        list[dict]: eval.csv行数据。
    """
    eval_path = Path(run_dir) / "eval.csv"
    if not eval_path.exists():
        return []
    rows = []
    bucket_set = set(str(x) for x in buckets)
    with open(eval_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("bucket", "")) not in bucket_set:
                continue
            rows.append(dict(row))
    return rows


def _write_summary_csv(output_dir, manifest, buckets, metrics):
    """
    功能:
        汇总所有实验eval结果到一个CSV。
    输入:
        output_dir (Path): sweep输出目录。
        manifest (list[dict]): 实验manifest。
        buckets (list[str]): 评估bucket列表。
        metrics (list[str]): 指标列表。
    输出:
        Path: 汇总CSV路径。
    """
    out_path = Path(output_dir) / "summary.csv"
    param_keys = sorted({k for item in manifest for k in item.get("params", {}).keys()})
    fieldnames = ["id", "experiment_name", "status", "run_dir", "bucket", "episode", "total_num_steps"] + param_keys + list(metrics)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in manifest:
            rows = _read_eval_rows(item.get("run_dir", ""), buckets)
            for row in rows:
                out = {
                    "id": item["id"],
                    "experiment_name": item["experiment_name"],
                    "status": item["status"],
                    "run_dir": item.get("run_dir", ""),
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
                        rows = _read_eval_rows(item.get("run_dir", ""), [bucket])
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
        _write_summary_csv(output_dir, manifest, buckets, metrics)
        _plot_results(output_dir, manifest, plot_cfg)
        print("[Sweep] plots written to {}".format(str(output_dir / "plots")))
        return

    output_dir, manifest = _prepare_experiments(sweep_cfg, sweep_path)

    if args.generate_only:
        print("[Sweep] generated {} configs under {}".format(len(manifest), str(output_dir)))
        return

    manifest = _run_experiments(sweep_cfg, output_dir, manifest)
    summary_path = _write_summary_csv(output_dir, manifest, buckets, metrics)
    _plot_results(output_dir, manifest, plot_cfg)
    print("[Sweep] summary written to {}".format(str(summary_path)))
    print("[Sweep] plots written to {}".format(str(output_dir / "plots")))


if __name__ == "__main__":
    main()
