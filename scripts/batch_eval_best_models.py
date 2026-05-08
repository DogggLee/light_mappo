"""
Batch offline evaluation for saved best-model directories.

设计目标:
1) 扫描给定根目录下所有包含指定模型目录的 run_dir。
2) 对每个 run_dir 的目标模型执行一次离线整体评估。
3) 将结果按任务文件名写回每个 run_dir，并在根目录汇总总表。
"""

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.eval import main_one_glob


def _slugify_filename_part(name: str) -> str:
    """
    功能:
        将任意文件名片段转换为适合落盘的安全名称。
    输入:
        name (str): 原始文件名片段。
    输出:
        str: 仅包含字母、数字、点、横线和下划线的名称。
    """
    safe_chars = []
    for ch in str(name):
        if ch.isalnum() or ch in (".", "-", "_"):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("_.")
    return safe or "tasks"


def _read_csv_rows(csv_path: Path):
    """
    功能:
        读取CSV文件全部行。
    输入:
        csv_path (Path): CSV文件路径。
    输出:
        list[dict]: 按表头解析后的行列表。
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv_rows(csv_path: Path, fieldnames, rows):
    """
    功能:
        将行数据写入CSV文件。
    输入:
        csv_path (Path): 输出CSV路径。
        fieldnames (list[str]): 表头字段顺序。
        rows (list[dict]): 待写入的记录列表。
    输出:
        无。
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _extract_experiment_alias(run_dir_rel: str):
    """
    功能:
        从相对run目录路径中提取实验目录别名；优先使用首层目录开头的[alias]。
    输入:
        run_dir_rel (str): 相对root_dir的run目录路径。
    输出:
        tuple[str, str]: (实验目录名, 实验别名)。
    """
    rel_parts = Path(str(run_dir_rel)).parts
    if len(rel_parts) == 0:
        return "", ""
    experiment_dir_name = str(rel_parts[0])
    match = re.match(r"^\[([^\]]+)\]", experiment_dir_name)
    if match is not None:
        return experiment_dir_name, str(match.group(1)).strip()
    return experiment_dir_name, experiment_dir_name


def _build_model_result_dir(root_dir: Path, task_name: str, experiment_alias: str, model_glob: str):
    """
    功能:
        构造root级模型结果归档目录。
    输入:
        root_dir (Path): 批量评估根目录。
        task_name (str): 当前任务文件名。
        experiment_alias (str): 实验别名。
        model_glob (str): 当前模型glob名称。
    输出:
        Path: root/results/<task_name>/<experiment_alias>_<model_glob> 目录。
    """
    task_slug = _slugify_filename_part(task_name)
    alias_slug = _slugify_filename_part(experiment_alias)
    model_slug = _slugify_filename_part(model_glob)
    return root_dir / "results" / task_slug / f"{alias_slug}_{model_slug}"


def _copy_model_bucket_artifacts(root_dir: Path, run_dir: Path, run_dir_rel: str, task_name: str, model_glob: str):
    """
    功能:
        将模型目录下的bucket绘图与json复制到root级结果归档目录。
    输入:
        root_dir (Path): 批量评估根目录。
        run_dir_rel (str): 相对run目录路径。
        task_name (str): 当前任务文件名。
        model_glob (str): 当前模型glob名称。
    输出:
        Path | None: 实际写入的归档目录；若无源文件则返回None。
    """
    experiment_dir_name, experiment_alias = _extract_experiment_alias(run_dir_rel)
    _ = experiment_dir_name
    src_dir = run_dir / "models" / model_glob / "res"
    if not src_dir.exists():
        return None

    dst_dir = _build_model_result_dir(root_dir, task_name, experiment_alias, model_glob)
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied_any = False
    for pattern in [
        "eval_hunter_bucket_metrics_ep_*.png",
        "eval_hunter_bucket_metrics_ep_*.json",
        "eval_bucket_metrics_by_*_ep_*.png",
        "eval_bucket_metrics_by_*_ep_*.json",
    ]:
        for src in sorted(src_dir.glob(pattern)):
            shutil.copy2(str(src), str(dst_dir / src.name))
            copied_any = True
    return dst_dir if copied_any else None


def _build_comparison_summary_rows(rows):
    """
    功能:
        将明细评估记录聚合为“每个模型一行”的横向对比表。
    输入:
        rows (list[dict]): 批量评估明细记录。
    输出:
        tuple[list[str], list[dict]]: 汇总表字段名与记录列表。
    """
    metric_fields = [
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
    lead_fields = [
        "experiment_alias",
        "task_name",
        "model_name",
    ]
    tail_fields = [
        "experiment_dir_name",
        "task_file",
        "run_dir_rel",
        "run_dir",
        "model_dir",
        "episode",
        "total_num_steps",
    ]

    grouped = {}
    bucket_names = set()
    for row in rows:
        key = (
            str(row.get("task_file", "")),
            str(row.get("run_dir", "")),
            str(row.get("model_name", "")),
            str(row.get("model_dir", "")),
        )
        if key not in grouped:
            grouped[key] = {}
            for field in lead_fields + tail_fields:
                grouped[key][field] = row.get(field, "")
        bucket = str(row.get("bucket", ""))
        bucket_names.add(bucket)
        for metric in metric_fields:
            grouped[key][f"{bucket}__{metric}"] = row.get(metric, "")

    bucket_priority = [
        "fixed_zone_true",
        "fixed_zone_false",
        "fixed",
        "learn_zone_true",
        "learn_zone_false",
        "learn",
    ]
    ordered_buckets = [b for b in bucket_priority if b in bucket_names]
    ordered_buckets.extend(sorted([b for b in bucket_names if b not in set(bucket_priority)]))
    fieldnames = list(lead_fields)
    for bucket in ordered_buckets:
        for metric in metric_fields:
            fieldnames.append(f"{bucket}__{metric}")
    fieldnames.extend(tail_fields)

    summary_rows = sorted(
        [grouped[key] for key in grouped.keys()],
        key=lambda row: (
            str(row.get("experiment_alias", "")),
            str(row.get("run_dir_rel", "")),
            str(row.get("model_name", "")),
            str(row.get("model_dir", "")),
        ),
    )
    return fieldnames, summary_rows


def _merge_summary_rows(existing_rows, new_rows):
    """
    功能:
        合并历史summary与本次summary；相同(task, run, model)主键使用新结果覆盖旧结果。
    输入:
        existing_rows (list[dict]): 已存在的summary记录。
        new_rows (list[dict]): 本次新生成的summary记录。
    输出:
        list[dict]: 合并去重后的summary记录。
    """
    def _row_key(row):
        return (
            str(row.get("task_file", "")),
            str(row.get("run_dir", "")),
            str(row.get("model_name", "")),
            str(row.get("model_dir", "")),
        )

    merged = {}
    for row in existing_rows:
        merged[_row_key(row)] = dict(row)
    for row in new_rows:
        merged[_row_key(row)] = dict(row)

    return sorted(
        list(merged.values()),
        key=lambda row: (
            str(row.get("task_name", "")),
            str(row.get("experiment_alias", "")),
            str(row.get("run_dir_rel", "")),
            str(row.get("model_name", "")),
            str(row.get("model_dir", "")),
        ),
    )


def _merge_summary_fieldnames(existing_fieldnames, new_fieldnames):
    """
    功能:
        合并summary表头，保持已有顺序并追加新增列。
    输入:
        existing_fieldnames (list[str]): 历史summary表头。
        new_fieldnames (list[str]): 本次summary表头。
    输出:
        list[str]: 合并后的表头顺序。
    """
    merged = []
    seen = set()
    for name in list(existing_fieldnames) + list(new_fieldnames):
        key = str(name)
        if key not in seen:
            merged.append(key)
            seen.add(key)
    return merged


def _discover_run_dirs(root_dir: Path, model_glob: str):
    """
    功能:
        扫描根目录，返回包含目标模型目录的全部 run_dir。
    输入:
        root_dir (Path): 扫描根目录。
        model_glob (str): 相对 run_dir/models 的模型目录匹配模式。
    输出:
        list[Path]: 满足条件的 run_dir 路径列表。
    """
    run_dirs = []
    for cfg_path in sorted(root_dir.rglob("train_cfg.yaml")):
        run_dir = cfg_path.parent
        models_dir = run_dir / "models"
        if not models_dir.exists():
            continue
        matched = sorted(models_dir.glob(model_glob))
        if len(matched) > 0:
            run_dirs.append(run_dir)
    return run_dirs


def _build_eval_args(args, run_dir: Path):
    """
    功能:
        构造与 train.eval.main_one_glob 兼容的参数对象。
    输入:
        args (argparse.Namespace): 批量评估命令行参数。
        run_dir (Path): 当前待评估实验目录。
    输出:
        argparse.Namespace: 单次离线评估参数。
    """
    return argparse.Namespace(
        config_file=args.config_file,
        run_dir=str(run_dir),
        cuda=bool(args.cuda),
        total_num_steps=args.total_num_steps,
        episode=args.episode,
        task_file=str(args.task_file),
        render=bool(args.render),
        plot=bool(args.plot),
        model_glob=args.model_glob,
    )


def _augment_rows(rows, root_dir: Path, run_dir: Path, task_file: Path):
    """
    功能:
        为原始评估结果补充任务文件与run目录标识字段。
    输入:
        rows (list[dict]): 原始 test_eval.csv 记录。
        root_dir (Path): 批量扫描根目录。
        run_dir (Path): 当前实验目录。
        task_file (Path): 本次评估使用的任务文件。
    输出:
        list[dict]: 增补字段后的记录列表。
    """
    task_name = task_file.name
    try:
        run_dir_rel = str(run_dir.relative_to(root_dir))
    except ValueError:
        run_dir_rel = str(run_dir)
    experiment_dir_name, experiment_alias = _extract_experiment_alias(run_dir_rel)

    out = []
    for row in rows:
        enriched = {
            "task_file": str(task_file),
            "task_name": str(task_name),
            "experiment_alias": str(experiment_alias),
            "experiment_dir_name": str(experiment_dir_name),
            "run_dir": str(run_dir),
            "run_dir_rel": str(run_dir_rel),
        }
        enriched.update(dict(row))
        out.append(enriched)
    return out


def main():
    """
    功能:
        批量扫描指定目录下的best模型，并使用给定固定任务做离线整体评估。
    输入:
        无，参数来自命令行。
    输出:
        无。
    """
    parser = argparse.ArgumentParser(
        description="Batch offline evaluation for saved best-model directories",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory that contains multiple experiment run dirs",
    )
    parser.add_argument(
        "--task_file",
        type=str,
        required=True,
        help="Override eval fixed task file used for offline benchmark",
    )
    parser.add_argument(
        "--model_glob",
        type=str,
        default="best_eval_capture_rate",
        help="Only evaluate model dirs matching this glob under run_dir/models",
    )
    parser.add_argument("--config_file", type=str, default=None, help="Optional YAML config override")
    parser.add_argument("--cuda", action="store_true", help="Use GPU if available")
    parser.add_argument("--total_num_steps", type=int, default=None, help="Override total_num_steps for eval logging")
    parser.add_argument("--episode", type=int, default=None, help="Override episode id used in eval/GIF naming")
    parser.add_argument("--render", action="store_true", help="Render eval env at every step during evaluation")
    parser.add_argument("--plot", action="store_true", help="Plot per-step debug curves during evaluation")
    parser.add_argument(
        "--aggregate_out",
        type=str,
        default=None,
        help="Optional path for aggregate CSV. Default: <root_dir>/batch_test_eval__<task_name>.csv",
    )
    cli_args = parser.parse_args()

    root_dir = Path(str(cli_args.root_dir)).resolve()
    task_file = Path(str(cli_args.task_file)).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")
    if not task_file.exists():
        raise FileNotFoundError(f"task_file not found: {task_file}")

    run_dirs = _discover_run_dirs(root_dir, str(cli_args.model_glob))
    if len(run_dirs) == 0:
        raise RuntimeError(
            f"No run_dir with models/{cli_args.model_glob} found under {root_dir}"
        )

    task_slug = _slugify_filename_part(task_file.name)
    aggregate_out = (
        Path(str(cli_args.aggregate_out)).resolve()
        if cli_args.aggregate_out is not None
        else root_dir / f"batch_test_eval__{task_slug}.csv"
    )
    summary_out = root_dir / "batch_summary.csv"

    print(f"[BatchEval] root_dir={root_dir}")
    print(f"[BatchEval] task_file={task_file}")
    print(f"[BatchEval] model_glob={cli_args.model_glob}")
    print(f"[BatchEval] matched run_dirs={len(run_dirs)}")

    aggregate_rows = []
    failures = []

    for idx, run_dir in enumerate(run_dirs, start=1):
        print(f"[BatchEval] ({idx}/{len(run_dirs)}) evaluating {run_dir}")
        try:
            eval_args = _build_eval_args(cli_args, run_dir)
            main_one_glob(eval_args, str(cli_args.model_glob))

            raw_csv_path = run_dir / "test_eval.csv"
            if not raw_csv_path.exists():
                raise FileNotFoundError(f"eval result csv not found: {raw_csv_path}")

            raw_rows = _read_csv_rows(raw_csv_path)
            augmented_rows = _augment_rows(raw_rows, root_dir, run_dir, task_file)
            per_run_out = run_dir / f"test_eval__{task_slug}.csv"
            fieldnames = list(augmented_rows[0].keys()) if len(augmented_rows) > 0 else [
                "task_name",
                "model_name",
                "task_file",
                "capture_rate",
                "capture_steps",
                "capture_steps_objective",
                "alive_rate",
                "max_escape_gap_angle",
                "capture_spread_reward",
                "captured_episodes",
                "total_eval_episodes",
                "run_dir",
                "run_dir_rel",
                "model_dir",
                "episode",
                "total_num_steps",
                "bucket",
                "eval_reward"
            ]
            _write_csv_rows(per_run_out, fieldnames, augmented_rows)
            aggregate_rows.extend(augmented_rows)
            archive_dir = _copy_model_bucket_artifacts(
                root_dir=root_dir,
                run_dir=run_dir,
                run_dir_rel=str(augmented_rows[0]["run_dir_rel"]) if len(augmented_rows) > 0 else str(run_dir),
                task_name=str(task_file.name),
                model_glob=str(cli_args.model_glob),
            )
            if archive_dir is not None:
                print(f"[BatchEval] archived bucket artifacts to {archive_dir}")
            print(f"[BatchEval] wrote {per_run_out}")
        except Exception as exc:
            failures.append({"run_dir": str(run_dir), "error": str(exc)})
            print(f"[BatchEval][ERROR] {run_dir}: {exc}")

    if len(aggregate_rows) > 0:
        fieldnames = list(aggregate_rows[0].keys())
        _write_csv_rows(aggregate_out, fieldnames, aggregate_rows)
        print(f"[BatchEval] aggregate csv written to {aggregate_out}")
        summary_fieldnames, summary_rows = _build_comparison_summary_rows(aggregate_rows)
        existing_summary_rows = []
        existing_summary_fieldnames = []
        if summary_out.exists():
            existing_summary_rows = _read_csv_rows(summary_out)
            if len(existing_summary_rows) > 0:
                existing_summary_fieldnames = list(existing_summary_rows[0].keys())
        merged_summary_rows = _merge_summary_rows(existing_summary_rows, summary_rows)
        merged_summary_fieldnames = _merge_summary_fieldnames(existing_summary_fieldnames, summary_fieldnames)
        _write_csv_rows(summary_out, merged_summary_fieldnames, merged_summary_rows)
        print(f"[BatchEval] comparison summary csv written to {summary_out}")

    if len(failures) > 0:
        error_out = aggregate_out.with_name(aggregate_out.stem + "__failures.csv")
        _write_csv_rows(error_out, ["run_dir", "error"], failures)
        print(f"[BatchEval] failures csv written to {error_out}")
        raise RuntimeError(f"{len(failures)} run(s) failed during batch evaluation")


if __name__ == "__main__":
    main()
