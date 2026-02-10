import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HELP_TEXT = """
Mouse controls:
  Left click  : add point
  Right click : remove last point
Keyboard:
  n : name and save current route
  e : edit existing route
  c : clear current points
  s : save all routes to file
  h : show help
  q : quit
""".strip()


def _load_routes(path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    routes = []
    if isinstance(data, dict) and "routes" in data:
        routes_data = data["routes"]
        if isinstance(routes_data, dict):
            for name, payload in routes_data.items():
                waypoints = payload.get("waypoints", payload)
                routes.append({"name": str(name), "waypoints": waypoints})
        elif isinstance(routes_data, list):
            for item in routes_data:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("alias")
                waypoints = item.get("waypoints")
                if name and waypoints:
                    routes.append({"name": str(name), "waypoints": waypoints})
    return routes


def _save_routes(path, routes):
    payload = {
        "meta": {"version": 1, "coords": "normalized_0_1"},
        "routes": routes,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clamp_point(x, y):
    return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))


def _maybe_auto_close(points, auto_close):
    if not auto_close or len(points) < 2:
        return points
    first = points[0]
    last = points[-1]
    if float(first[0]) == float(last[0]) and float(first[1]) == float(last[1]):
        return points
    return points + [first]


def main():
    parser = argparse.ArgumentParser(description="Draw and save target patrol routes.")
    parser.add_argument("--out", type=str, default="config/target_patrol_routes.json")
    parser.add_argument("--show-existing", action="store_true", help="Show existing routes in the plot")
    parser.add_argument("--auto-close", action="store_true", help="Append the first point to close the route when saving")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    routes = _load_routes(out_path)
    current = []
    editing_name = None

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("Target Patrol Route Editor")

    existing_lines = []
    if args.show_existing and routes:
        for item in routes:
            pts = np.asarray(item["waypoints"], dtype=float)
            if pts.ndim == 2 and pts.shape[1] == 2 and len(pts) >= 2:
                line, = ax.plot(pts[:, 0], pts[:, 1], color="#999999", linewidth=1.0, alpha=0.6)
                existing_lines.append(line)

    (line,) = ax.plot([], [], color="#d62728", linewidth=2.0)
    scatter = ax.scatter([], [], color="#d62728", s=45)

    def refresh():
        if current:
            pts = np.asarray(current, dtype=float)
            pts_line = pts
            if args.auto_close and len(pts) >= 2:
                pts_line = np.vstack([pts, pts[0]])
            line.set_data(pts_line[:, 0], pts_line[:, 1])
            scatter.set_offsets(pts)
        else:
            line.set_data([], [])
            scatter.set_offsets(np.empty((0, 2)))
        route_names = ", ".join([r["name"] for r in routes]) or "(none)"
        ax.set_xlabel(f"Routes: {route_names}")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            x, y = _clamp_point(event.xdata, event.ydata)
            current.append([x, y])
            refresh()
        elif event.button == 3:
            if current:
                current.pop()
                refresh()

    def on_key(event):
        if event.key == "h":
            print(HELP_TEXT)
        elif event.key == "c":
            current.clear()
            refresh()
        elif event.key == "n":
            if len(current) < 2:
                print("当前路径至少需要2个点。")
                return
            name = input("请输入该路线名称: ").strip()
            if not name:
                print("路线名称不能为空。")
                return
            waypoints = _maybe_auto_close(current.copy(), args.auto_close)
            replaced = False
            for item in routes:
                if item["name"] == name:
                    item["waypoints"] = waypoints
                    replaced = True
                    break
            if not replaced:
                routes.append({"name": name, "waypoints": waypoints})
            editing_name = None
            current.clear()
            refresh()
            action = "已更新路线" if replaced else "已保存路线"
            print(f"{action}: {name}")
        elif event.key == "e":
            if not routes:
                print("当前没有可编辑的路线。")
                return
            print("已有路线:")
            for idx, item in enumerate(routes):
                print(f"  {idx + 1}. {item['name']}")
            raw = input("请输入要编辑的路线名称或编号: ").strip()
            if not raw:
                return
            target = None
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(routes):
                    target = routes[idx]
            else:
                for item in routes:
                    if item["name"] == raw:
                        target = item
                        break
            if target is None:
                print("未找到该路线。")
                return
            current.clear()
            current.extend([list(pt) for pt in target["waypoints"]])
            editing_name = target["name"]
            print(f"已载入路线: {target['name']}，编辑后按 n 可覆盖同名或另存新名。")
            refresh()
        elif event.key == "s":
            _save_routes(out_path, routes)
            print(f"已保存到: {out_path}")
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print(HELP_TEXT)
    refresh()
    plt.show()


if __name__ == "__main__":
    main()
