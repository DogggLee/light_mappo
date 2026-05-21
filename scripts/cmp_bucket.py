#!/usr/bin/env python3
"""
对比多个 eval_hunter_bucket_metrics_ep_*.json 的关键性能曲线。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """
    功能:
        解析命令行参数。
    输入:
        无。
    输出:
        argparse.Namespace: 解析后的参数对象。
    """
    # Step 1: 构建参数解析器并注册必需参数
    parser = argparse.ArgumentParser(
        description="Compare bucket metrics from multiple eval_hunter_bucket_metrics json files."
    )
    parser.add_argument(
        "--jsons",
        type=str,
        nargs="+",
        required=True,
        help="多个 eval_hunter_bucket_metrics_ep_*.json 文件路径。",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        required=True,
        help="与 --jsons 一一对应的曲线名称。",
    )
    parser.add_argument(
        "--ls",
        action="store_true",
        help="为 True 时用线型区分；不传则为 False，使用颜色区分。",
    )
    parser.add_argument(
        "--marker",
        action="store_true",
        help="为 True 时用 marker 区分曲线；不传则不使用 marker。",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="可选：指定使用的 bucket 名称（默认自动选择第一个）。",
    )
    parser.add_argument(
        "--x_values",
        type=float,
        nargs="+",
        default=None,
        help="可选：只绘制并显示指定的x轴数值，例如 --x_values 1 4 6 8 10 15。",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="可选：输出图片路径；不填则直接弹窗展示。",
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=16,
        help="标题、坐标轴名称和坐标刻度字号，默认 16。",
    )
    return parser.parse_args()


def _load_metric_xy(
    json_path: Path,
    bucket_name: str | None,
    metric_name: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    功能:
        从单个bucket指标JSON中读取指定metric的x/y序列。
    输入:
        json_path (Path): 指标JSON文件路径。
        bucket_name (str | None): 指定bucket名；为None时自动选择第一个。
        metric_name (str): 指标名（如eval_reward/capture_rate/capture_steps）。
    输出:
        Tuple[np.ndarray, np.ndarray, str]:
            - x坐标数组；
            - y坐标数组；
            - 实际使用的bucket名称。
    """
    # Step 1: 读取并检查JSON结构
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid json object: {json_path}")
    buckets = data.get("buckets", {})
    if not isinstance(buckets, dict) or len(buckets) == 0:
        raise ValueError(f"Missing non-empty 'buckets' in: {json_path}")

    # Step 2: 解析bucket并读取目标metric
    used_bucket = str(bucket_name) if bucket_name is not None else str(next(iter(buckets.keys())))
    if used_bucket not in buckets:
        raise ValueError(f"Bucket '{used_bucket}' not found in: {json_path}")
    bucket_metrics = buckets.get(used_bucket, {})
    if not isinstance(bucket_metrics, dict):
        raise ValueError(f"Bucket '{used_bucket}' is not a dict in: {json_path}")
    metric_obj = bucket_metrics.get(metric_name)
    if not isinstance(metric_obj, dict):
        raise ValueError(f"Metric '{metric_name}' missing in {json_path} bucket='{used_bucket}'")

    x_raw = metric_obj.get("x", [])
    y_raw = metric_obj.get("y", [])
    x = np.asarray(x_raw, dtype=np.float32)
    y = np.asarray(y_raw, dtype=np.float32)
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Metric '{metric_name}' x/y length mismatch in {json_path}: {x.shape[0]} vs {y.shape[0]}"
        )
    return x, y, used_bucket


def _build_styles(
    num_curves: int,
    use_line_style: bool,
    use_marker: bool,
) -> List[Dict[str, str | None]]:
    """
    功能:
        生成每条曲线的绘图风格（颜色/线型/marker）。
    输入:
        num_curves (int): 曲线数量。
        use_line_style (bool): 是否启用线型区分模式。
        use_marker (bool): 是否启用 marker 区分模式。
    输出:
        List[Dict[str, str | None]]: 每条曲线的style字典。
    """
    # Step 1: 准备颜色与线型候选
    color_cycle = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    line_cycle = ["-", "--", "-.", ":"]
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    styles: List[Dict[str, str | None]] = []

    # Step 2: 根据参数选择区分策略
    for idx in range(num_curves):
        color = "black" if (use_line_style or use_marker) else color_cycle[idx % len(color_cycle)]
        linestyle = line_cycle[idx % len(line_cycle)] if use_line_style else "-"
        marker = marker_cycle[idx % len(marker_cycle)] if use_marker else 'o'
        styles.append({"color": color, "linestyle": linestyle, "marker": marker})
    return styles


def _plot_one_metric(
    ax,
    json_paths: List[Path],
    names: List[str],
    styles: List[Dict[str, str | None]],
    bucket_name: str | None,
    x_values: List[float] | None,
    metric_name: str,
    title: str,
    y_label: str,
    font_size: int,
) -> str:
    """
    功能:
        在单个子图中绘制同一指标的多条对比曲线。
    输入:
        ax: matplotlib子图对象。
        json_paths (List[Path]): 多个指标JSON路径。
        names (List[str]): 对应曲线名称列表。
        styles (List[Dict[str, str | None]]): 对应曲线样式列表。
        bucket_name (str | None): 指定bucket名；None时自动选第一个。
        x_values (List[float] | None): 指定要绘制和显示的x轴取值；None表示不筛选。
        metric_name (str): 指标名。
        title (str): 子图标题。
        y_label (str): y轴标签。
        font_size (int): 标题、轴标签与刻度字号。
    输出:
        str: 实际使用的bucket名称（用于总标题展示）。
    """
    # Step 1: 循环加载并绘制每条曲线，同时收集真实x坐标
    used_bucket_final = ""
    x_ticks: List[float] = []
    for idx, (path, name) in enumerate(zip(json_paths, names)):
        x, y, used_bucket = _load_metric_xy(path, bucket_name, metric_name)
        used_bucket_final = used_bucket
        if x_values is not None:
            keep_mask = np.zeros_like(x, dtype=bool)
            for x_value in x_values:
                keep_mask = np.logical_or(keep_mask, np.isclose(x, float(x_value)))
            x = x[keep_mask]
            y = y[keep_mask]
        y = np.where(np.isfinite(y), y, np.nan)
        x_ticks.extend([float(v) for v in x[np.isfinite(x)]])
        ax.plot(
            x,
            y,
            label=str(name),
            color=styles[idx]["color"],
            linestyle=styles[idx]["linestyle"],
            linewidth=2.0,
            marker=styles[idx]["marker"],
            markersize=6.0,
        )

    # Step 2: 设置坐标轴样式
    if len(x_ticks) > 0:
        sorted_x_ticks = sorted(set(x_ticks))
        ax.set_xticks(sorted_x_ticks)
        if all(float(v).is_integer() for v in sorted_x_ticks):
            ax.set_xticklabels([str(int(v)) for v in sorted_x_ticks])
    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel("Number of Pursuers", fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.tick_params(axis="both", labelsize=font_size)
    ax.grid(True, linestyle="--", alpha=0.3)
    return used_bucket_final


def main() -> None:
    """
    功能:
        程序入口：读取多组bucket指标JSON并绘制4个核心指标对比图。
    输入:
        无（从CLI读取）。
    输出:
        无。
    """
    # Step 1: 参数校验与路径准备
    args = parse_args()
    if len(args.jsons) != len(args.names):
        raise ValueError(
            f"--jsons count ({len(args.jsons)}) must equal --names count ({len(args.names)})"
        )
    json_paths = [Path(p) for p in args.jsons]
    for path in json_paths:
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

    # Step 2: 创建画布并绘制4个核心指标（2x2）
    styles = _build_styles(
        num_curves=len(json_paths),
        use_line_style=bool(args.ls),
        use_marker=bool(args.marker),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=False)
    axes = np.asarray(axes).reshape(-1)
    used_bucket = _plot_one_metric(
        ax=axes[0],
        json_paths=json_paths,
        names=list(args.names),
        styles=styles,
        bucket_name=args.bucket,
        x_values=args.x_values,
        metric_name="capture_rate",
        title="",
        y_label="Capture Rate (%)",
        font_size=int(args.font_size),
    )
    axes[0].set_ylim(0.0, 1.0)
    # axes[0].set_yticks(np.linspace(0.3, 1.0, 7))
    axes[0].set_yticks(np.linspace(0.0, 1.0, 11))
    axes[0].set_yticklabels(
        [f"{int(v * 1)}%" for v in range(0, 110, 10)],
        fontsize=int(args.font_size),
    )

    _plot_one_metric(
        ax=axes[3],
        json_paths=json_paths,
        names=list(args.names),
        styles=styles,
        bucket_name=used_bucket,
        x_values=args.x_values,
        metric_name="capture_steps",
        title="Capture Steps",
        y_label="Steps",
        font_size=int(args.font_size),
    )
    axes[3].set_ylim(100.0, 200.0)

    _plot_one_metric(
        ax=axes[2],
        json_paths=json_paths,
        names=list(args.names),
        styles=styles,
        bucket_name=used_bucket,
        x_values=args.x_values,
        metric_name="alive_rate",
        title="Alive Rate",
        y_label="Rate",
        font_size=int(args.font_size),
    )
    # axes[2].set_ylim(0.4, 1.0)
    # axes[2].set_yticks(np.linspace(0.4, 1.0, 7))
    # axes[2].set_yticklabels(
    #     [f"{int(v * 100)}%" for v in np.linspace(0.4, 1.0, 7)],
    #     fontsize=int(args.font_size),
    # )
    axes[2].set_ylim(0.3, 1.0)
    axes[2].set_yticks(np.linspace(0.3, 1.0, 8))
    axes[2].set_yticklabels(
        [f"{int(v * 1)}%" for v in range(30, 110, 10)],
        fontsize=int(args.font_size),)

    _plot_one_metric(
        ax=axes[1],
        json_paths=json_paths,
        names=list(args.names),
        styles=styles,
        bucket_name=used_bucket,
        x_values=args.x_values,
        metric_name="capture_spread_reward",
        title="",
        y_label="Spread Reward",
        font_size=int(args.font_size),
    )

    # Step 3: 设置统一标题与图外底部图例，避免遮挡曲线
    mode_parts: List[str] = []
    mode_parts.append("LineStyle" if bool(args.ls) else "SolidLine")
    mode_parts.append("Marker" if bool(args.marker) else "NoMarker")
    mode_parts.append("Black" if (bool(args.ls) or bool(args.marker)) else "Color")
    fig.suptitle(
        f"Bucket Compare ({used_bucket}) | Style={' + '.join(mode_parts)}",
        fontsize=int(args.font_size),
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=max(1, min(len(labels), 6)),
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        fontsize=int(args.font_size),
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])

    fig.subplots_adjust(0.15, 0.15, 0.95, 0.9, 0.17, 0.35)

    # Step 4: 输出图片或直接展示
    if args.out is not None and len(str(args.out).strip()) > 0:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
        print(f"[Saved] {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
