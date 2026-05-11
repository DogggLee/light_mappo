#!/usr/bin/env python3
"""
可视化 spread reward 的几何含义。

脚本功能：
1) 随机生成 N 个 hunter 与 1 个 target 的二维场景。
2) target 固定在中心，hunter 在其周围按极坐标随机采样。
3) 在同一张图中绘制 5x5 地图、target、hunter、单位圆投影点与几何中心。
4) 按空格键重新随机采样并重绘。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """
    功能:
        解析命令行参数。
    输入:
        无。
    输出:
        argparse.Namespace: 参数对象。
    """
    parser = argparse.ArgumentParser(
        description="Visualize spread reward geometry for random hunter-target scenes."
    )
    parser.add_argument(
        "--num_hunters",
        type=int,
        default=6,
        help="Hunter 数量。",
    )
    parser.add_argument(
        "--map_size",
        type=float,
        default=5.0,
        help="地图边长，target 固定在中心，默认 5x5。",
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=1.0,
        help="Hunter 与 target 的最小距离。",
    )
    parser.add_argument(
        "--max_dist",
        type=float,
        default=2.0,
        help="Hunter 与 target 的最大距离。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="target_to_hunter",
        choices=["hunter_to_target", "target_to_hunter"],
        help="单位方向定义。默认 target_to_hunter，更直观；与当前代码实现相比仅差整体取反，spread score 不变。",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="可选：输出图片路径；不填则直接弹窗显示。",
    )
    return parser.parse_args()


def sample_scene(
    num_hunters: int,
    map_size: float,
    min_dist: float,
    max_dist: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    功能:
        随机生成 target 与多个 hunter 的二维位置。
    输入:
        num_hunters (int): Hunter 数量。
        map_size (float): 地图边长。
        min_dist (float): Hunter 与 target 的最小距离。
        max_dist (float): Hunter 与 target 的最大距离。
        seed (int): 随机种子。
    输出:
        tuple[np.ndarray, np.ndarray]:
            - target_pos: shape=(2,)
            - hunter_pos: shape=(num_hunters, 2)
    """
    rng = np.random.default_rng(int(seed))
    if float(min_dist) <= 0.0:
        raise ValueError("--min_dist must be > 0")
    if float(max_dist) < float(min_dist):
        raise ValueError("--max_dist must be >= --min_dist")
    half_size = float(map_size) * 0.5
    target_pos = np.asarray([0.0, 0.0], dtype=np.float32)

    radii = rng.uniform(float(min_dist), float(max_dist), size=(int(num_hunters),))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(int(num_hunters),))
    hunter_pos = np.stack(
        [radii * np.cos(angles), radii * np.sin(angles)],
        axis=1,
    ).astype(np.float32)

    if bool(np.any(np.abs(hunter_pos) > half_size + 1e-6)):
        raise ValueError("Sampled hunters exceed map bounds. Increase --map_size or reduce --max_dist.")
    return target_pos, hunter_pos


def compute_spread_geometry(
    target_pos: np.ndarray,
    hunter_pos: np.ndarray,
    direction: str,
) -> dict:
    """
    功能:
        计算单位圆投影点、几何中心与 spread score。
    输入:
        target_pos (np.ndarray): Target 位置，shape=(2,)。
        hunter_pos (np.ndarray): Hunter 位置，shape=(N,2)。
        direction (str): 单位方向定义，取 hunter_to_target 或 target_to_hunter。
    输出:
        dict: 几何计算结果字典。
    """
    if str(direction) == "hunter_to_target":
        rel = np.asarray(target_pos[None, :] - hunter_pos, dtype=np.float32)
    else:
        rel = np.asarray(hunter_pos - target_pos[None, :], dtype=np.float32)

    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    if bool(np.any(dist <= 1e-8)):
        raise ValueError("Encountered degenerate hunter-target distance <= 1e-8.")

    unit_points = rel / dist
    centroid = np.mean(unit_points, axis=0)
    centroid_dist = float(np.linalg.norm(centroid))
    centroid_dist = float(np.clip(centroid_dist, 0.0, 1.0))
    spread_score = float(np.clip(1.0 - centroid_dist, 0.0, 1.0))

    return {
        "rel": rel,
        "dist": dist.reshape(-1),
        "unit_points": unit_points,
        "centroid": centroid,
        "centroid_dist": centroid_dist,
        "spread_score": spread_score,
    }


def _draw_scene(
    ax,
    target_pos: np.ndarray,
    hunter_pos: np.ndarray,
    unit_points: np.ndarray,
    centroid: np.ndarray,
    spread_score: float,
    map_size: float,
    direction: str,
) -> None:
    """
    功能:
        在同一张图中绘制地图、单位圆投影、hunter 与几何中心。
    输入:
        ax: matplotlib 坐标轴对象。
        target_pos (np.ndarray): Target 位置，shape=(2,)。
        hunter_pos (np.ndarray): Hunter 位置，shape=(N,2)。
        unit_points (np.ndarray): 单位方向向量点，shape=(N,2)。
        centroid (np.ndarray): 几何中心，shape=(2,)。
        spread_score (float): spread score。
        map_size (float): 地图边长。
        direction (str): 单位方向定义。
    输出:
        无。
    """
    ax.clear()
    half_size = float(map_size) * 0.5
    map_rect = plt.Rectangle(
        (-half_size, -half_size),
        float(map_size),
        float(map_size),
        fill=False,
        edgecolor="#777777",
        linewidth=1.2,
        linestyle="-",
        label=f"Map {float(map_size):g}x{float(map_size):g}",
        zorder=0,
    )
    ax.add_patch(map_rect)

    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        color="#444444",
        linewidth=1.4,
        label="Unit Circle",
        zorder=1,
    )
    ax.axhline(0.0, color="#dddddd", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="#dddddd", linewidth=0.8, zorder=0)

    ax.scatter(
        hunter_pos[:, 0],
        hunter_pos[:, 1],
        s=55,
        c="#1f77b4",
        label="Hunters",
        zorder=3,
    )
    ax.scatter(
        [float(target_pos[0])],
        [float(target_pos[1])],
        s=90,
        c="#d62728",
        marker="s",
        label="Target",
        zorder=4,
    )

    ax.scatter(
        unit_points[:, 0],
        unit_points[:, 1],
        s=34,
        marker="o",
        c="#2ca02c",
        label=f"Normalized Bearings ({direction})",
        zorder=4,
    )
    if int(unit_points.shape[0]) >= 3:
        poly_angles = np.arctan2(unit_points[:, 1], unit_points[:, 0])
        poly_order = np.argsort(poly_angles)
        polygon_points = unit_points[poly_order]
        ax.fill(
            polygon_points[:, 0],
            polygon_points[:, 1],
            color="#2ca02c",
            alpha=0.18,
            label="Bearing Polygon",
            zorder=2,
        )

    ax.scatter(
        [float(centroid[0])],
        [float(centroid[1])],
        s=34,
        c="#ff7f0e",
        marker="o",
        label="Centroid",
        zorder=5,
    )
    ax.plot(
        [0.0, float(centroid[0])],
        [0.0, float(centroid[1])],
        color="#ff7f0e",
        linewidth=1.2,
        linestyle="--",
        zorder=2,
    )

    for idx, hunter in enumerate(hunter_pos):
        ax.plot(
            [float(hunter[0]), float(target_pos[0])],
            [float(hunter[1]), float(target_pos[1])],
            color="#b0b0b0",
            linewidth=0.8,
            alpha=0.8,
            zorder=1,
        )
        ax.annotate(
            f"H{int(idx)}",
            xy=(float(hunter[0]), float(hunter[1])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#1f77b4",
        )
        ax.annotate(
            f"P{int(idx)}",
            xy=(float(unit_points[idx, 0]), float(unit_points[idx, 1])),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=8,
            color="#2ca02c",
        )

    text = (
        f"spread_score = {float(spread_score):.4f}\n"
        f"centroid = ({float(centroid[0]):.4f}, {float(centroid[1]):.4f})\n"
        f"||centroid|| = {float(np.linalg.norm(centroid)):.4f}"
    )
    ax.text(
        0.98,
        0.02,
        text,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "#f8f8f8", "edgecolor": "#dddddd"},
    )

    ax.set_title("Spread Reward Geometry")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    margin = max(0.3, 0.12 * float(map_size))
    ax.set_xlim(-half_size - margin, half_size + margin)
    ax.set_ylim(-half_size - margin, half_size + margin)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        fontsize=9,
    )


def _render_scene(ax, fig, args, seed):
    """
    功能:
        基于指定随机种子重采样并重绘整张图。
    输入:
        ax: matplotlib 坐标轴对象。
        fig: matplotlib 图对象。
        args (argparse.Namespace): 命令行参数。
        seed (int): 本次绘图使用的随机种子。
    输出:
        int: 下一次可继续使用的随机种子。
    """
    target_pos, hunter_pos = sample_scene(
        num_hunters=int(args.num_hunters),
        map_size=float(args.map_size),
        min_dist=float(args.min_dist),
        max_dist=float(args.max_dist),
        seed=int(seed),
    )
    geom = compute_spread_geometry(
        target_pos=target_pos,
        hunter_pos=hunter_pos,
        direction=str(args.direction),
    )
    _draw_scene(
        ax=ax,
        target_pos=target_pos,
        hunter_pos=hunter_pos,
        unit_points=geom["unit_points"],
        centroid=geom["centroid"],
        spread_score=float(geom["spread_score"]),
        map_size=float(args.map_size),
        direction=str(args.direction),
    )
    fig.suptitle(
        "Spread Reward Visualization | num_hunters={} | seed={} | Press Space to Resample".format(
            int(args.num_hunters),
            int(seed),
        )
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    return int(seed) + 1


def main() -> None:
    """
    功能:
        程序入口：随机生成场景并输出 spread reward 可视化图。
    输入:
        无（从命令行读取）。
    输出:
        无。
    """
    args = parse_args()
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 8.0), dpi=130)
    current_seed = _render_scene(ax=ax, fig=fig, args=args, seed=int(args.seed))

    if args.out is not None:
        out_path = Path(str(args.out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), bbox_inches="tight")
        print(f"Saved figure to {out_path}")
        plt.close(fig)
        return

    state = {"seed": int(current_seed)}

    def _on_key(event):
        """
        功能:
            键盘交互回调；按空格键后重采样并重绘图像。
        输入:
            event: matplotlib 键盘事件对象。
        输出:
            无。
        """
        if str(event.key) != " ":
            return
        state["seed"] = _render_scene(
            ax=ax,
            fig=fig,
            args=args,
            seed=int(state["seed"]),
        )
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.show()


if __name__ == "__main__":
    main()
