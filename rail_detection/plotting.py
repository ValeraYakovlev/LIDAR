"""Графики для результатов детекции желоба и рельс."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .detector import classify_points
from .tracking import robust_centerline
from .curvature import describe_path

COLORS = {"groove": "green", "rail": "orange", "rest": "red"}


def plot_bag_overlay(ax, bag_name, records, n_skipped, x_lim=(-2, 2), z_lim=(-2.2, -0.8)):
    """Рисует на ax наложенные друг на друга (выровненные по центру рельс) срезы
    одного бэга. Возвращает (rail_gauge_mean, rail_gauge_std)."""
    gauges = [r["rail_gauge"] for r in records]
    for r in records:
        cx = r["rail_center"]
        x_shift = r["x_raw"] - cx
        z_raw = r["z_raw"]
        is_groove, is_rail, is_rest = classify_points(r)
        ax.scatter(x_shift[is_rest], z_raw[is_rest], s=1.5, c=COLORS["rest"], alpha=0.12)
        ax.scatter(x_shift[is_groove], z_raw[is_groove], s=1.5, c=COLORS["groove"], alpha=0.3)
        ax.scatter(x_shift[is_rail], z_raw[is_rail], s=2.5, c=COLORS["rail"], alpha=0.6)
    g_mean = float(np.mean(gauges)) if gauges else float("nan")
    g_std = float(np.std(gauges)) if gauges else float("nan")
    ax.set_xlim(*x_lim); ax.set_ylim(*z_lim)
    ax.set_title(
        f"{bag_name}\nвыровнено: {len(records)}, пропущено: {n_skipped}\n"
        f"рельс-рельс: {g_mean:.2f}±{g_std:.2f}м",
        fontsize=10,
    )
    ax.set_xlabel("X относительно центра рельс, м")
    ax.set_ylabel("Z, м")
    ax.grid(True, alpha=0.3)
    return g_mean, g_std


def plot_all_bags_grid(results_by_bag, out_path, x_lim=(-2, 2), z_lim=(-2.2, -0.8)):
    """results_by_bag: {bag_name: (records, n_skipped)}. Сохраняет PNG,
    возвращает сводку {bag_name: {...}} с числами по каждому бэгу."""
    n = len(results_by_bag)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.3 * ncols, 5.5 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    summary = {}
    for ax, (bag_name, (records, n_skip)) in zip(axes, results_by_bag.items()):
        g_mean, g_std = plot_bag_overlay(ax, bag_name, records, n_skip, x_lim, z_lim)
        summary[bag_name] = {
            "n_aligned": len(records),
            "n_skipped": n_skip,
            "rail_gauge_mean": g_mean,
            "rail_gauge_std": g_std,
        }
    for ax in axes[len(results_by_bag):]:
        ax.axis("off")

    fig.suptitle(
        "Выравнивание срезов по рельсам (оранжевый); желоб — зелёный, остальное — красный",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return summary


def plot_combined_overlay(results_by_bag, out_path, x_lim=(-2, 2), z_lim=(-2.2, -0.8)):
    """Все бэги вместе на одном графике, выровнено по рельсам."""
    fig, ax = plt.subplots(figsize=(9, 7))
    buckets_x = {"groove": [], "rail": [], "rest": []}
    buckets_z = {"groove": [], "rail": [], "rest": []}

    for records, _ in results_by_bag.values():
        for r in records:
            cx = r["rail_center"]
            x_shift = r["x_raw"] - cx
            z_raw = r["z_raw"]
            is_groove, is_rail, is_rest = classify_points(r)
            buckets_x["rest"].append(x_shift[is_rest]); buckets_z["rest"].append(z_raw[is_rest])
            buckets_x["groove"].append(x_shift[is_groove]); buckets_z["groove"].append(z_raw[is_groove])
            buckets_x["rail"].append(x_shift[is_rail]); buckets_z["rail"].append(z_raw[is_rail])

    counts = {}
    for key, s, alpha in [("rest", 1, 0.06), ("groove", 1, 0.15), ("rail", 2, 0.4)]:
        xs = np.concatenate(buckets_x[key]) if buckets_x[key] else np.array([])
        zs = np.concatenate(buckets_z[key]) if buckets_z[key] else np.array([])
        counts[key] = len(xs)
        ax.scatter(xs, zs, s=s, c=COLORS[key], alpha=alpha)

    ax.set_xlim(*x_lim); ax.set_ylim(*z_lim)
    ax.set_xlabel("X относительно центра рельс, м")
    ax.set_ylabel("Z, м")
    ax.set_title(
        f"Все бэги вместе, выровнено по рельсам\n"
        f"желоб: {counts['groove']}, рельсы: {counts['rail']}, остальное: {counts['rest']}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_centerline_grid(results_by_bag, out_path):
    """Извлечённая centerline (X центра рельс по глубине) — по одному графику на бэг."""
    n = len(results_by_bag)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (bag_name, (records, _)) in zip(axes, results_by_bag.items()):
        depths = [(r["depth_lo"] + r["depth_hi"]) / 2 for r in records]
        centers = [r["rail_center"] for r in records]
        ax.plot(depths, centers, marker="o", color="darkgreen")
        ax.set_title(bag_name, fontsize=9)
        ax.set_xlabel("глубина, м")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("X центра рельс (centerline), м")

    fig.suptitle("Извлечённая centerline пути по рельсам (по одному кадру на бэг)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_topdown(ax, points, records, bag_name, depth_max=60, x_lim=(-6, 6), robust=True,
                  wall_records=None, wall_turn_desc=None, thresh_deg=1.5):
    """Вид сверху (план): X — вбок, глубина (-Y) — вверх по графику (ось Y на
    графике = ось глубины тоннеля, перпендикулярно оси Z — высоту не показываем).
    Серым — сырое облако точек (контекст: стены), красным — извлечённая
    centerline по рельсам, пунктиром — прямая линия от первой точки для
    сравнения "путь идёт прямо / путь поворачивает".

    robust=True: одиночные выбросы centerline (промахи детектора) отбрасываются
    и заменяются значением по устойчивой (робастной) сглаженной кривой —
    см. tracking.robust_centerline. Отброшенные точки помечаются крестиком.

    wall_records: список из walls.analyze_walls (опц.) — если задан, поверх
    рисуются найденные позиции левой/правой стены (независимое от рельс
    свидетельство поворота).
    wall_turn_desc: готовая строка описания курса ПО СТЕНАМ (опц., считается
    вызывающим кодом через curvature.describe_path на подогнанной по стенам
    кривой) — выносится в заголовок как основной источник "как определили
    поворот"; описание по рельсам показывается отдельной строкой ниже."""
    x, y = points['x'], points['y']
    depth = -y
    mask = (depth > 0) & (depth < depth_max) & (np.abs(x) < x_lim[1] + 1)
    ax.scatter(x[mask], depth[mask], s=0.3, c='gray', alpha=0.35)

    turn_desc = ""
    if records:
        if robust:
            records_sorted, is_outlier, centers, coeffs = robust_centerline(records)
            depths = np.array([(r["depth_lo"] + r["depth_hi"]) / 2 for r in records_sorted])
        else:
            depths = np.array([(r["depth_lo"] + r["depth_hi"]) / 2 for r in records])
            centers = np.array([r["rail_center"] for r in records])
            order = np.argsort(depths)
            depths, centers = depths[order], centers[order]
            is_outlier = np.zeros(len(depths), dtype=bool)
            coeffs = None

        ax.plot(centers, depths, color='crimson', lw=2, marker='o', ms=4,
                label='centerline (по рельсам)')
        if is_outlier.any():
            raw_centers = np.array([r["rail_center"] for r in records_sorted])
            ax.scatter(raw_centers[is_outlier], depths[is_outlier], marker='x',
                       s=60, c='black', zorder=5, label='отброшенный выброс')

        if len(depths) >= 2:
            # прямая линия от первой точки в направлении начального курса
            dx0 = centers[1] - centers[0]
            dz0 = depths[1] - depths[0]
            heading = dx0 / dz0 if dz0 != 0 else 0.0
            straight_x = centers[0] + heading * (depths - depths[0])
            ax.plot(straight_x, depths, color='steelblue', lw=1.3, ls='--',
                     label='прямая (по начальному курсу)')

        if coeffs is not None:
            turn_desc = "по рельсам: " + describe_path(coeffs, list(depths), thresh_deg)

    if wall_turn_desc:
        turn_desc = f"по стенам: {wall_turn_desc}" + (f"\n{turn_desc}" if turn_desc else "")

    if wall_records:
        wd = np.array([(w["depth_lo"] + w["depth_hi"]) / 2 for w in wall_records])
        lw = np.array([w["left_wall_x"] for w in wall_records])
        rw = np.array([w["right_wall_x"] for w in wall_records])
        order = np.argsort(wd)
        ax.plot(lw[order], wd[order], color='darkblue', lw=1.6, marker='s', ms=3,
                label='левая стена')
        ax.plot(rw[order], wd[order], color='purple', lw=1.6, marker='s', ms=3,
                label='правая стена')

    ax.set_xlim(*x_lim)
    ax.set_ylim(0, depth_max)
    ax.set_xlabel("X, м (вбок)")
    ax.set_ylabel("глубина, м (вперёд)")
    title = bag_name if not turn_desc else f"{bag_name}\n{turn_desc}"
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6.5, loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_topdown_grid(points_by_bag, results_by_bag, out_path, depth_max=60, x_lim=(-6, 6),
                       wall_records_by_bag=None, wall_turn_desc_by_bag=None):
    """points_by_bag: {bag_name: points}. results_by_bag: {bag_name: (records, n_skipped)}.
    wall_records_by_bag, wall_turn_desc_by_bag: {bag_name: ...} (опц.) — см. plot_topdown."""
    n = len(points_by_bag)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 6.8 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, bag_name in zip(axes, points_by_bag):
        records, _ = results_by_bag[bag_name]
        wall_records = (wall_records_by_bag or {}).get(bag_name)
        wall_turn_desc = (wall_turn_desc_by_bag or {}).get(bag_name)
        plot_topdown(ax, points_by_bag[bag_name], records, bag_name, depth_max, x_lim,
                     wall_records=wall_records, wall_turn_desc=wall_turn_desc)
    for ax in axes[len(points_by_bag):]:
        ax.axis("off")

    fig.suptitle(
        "Тоннель сверху (план): серое — сырые точки, красное — centerline по рельсам,\n"
        "синий/фиолетовый — левая/правая стена, курс определён по стенам (независимо от рельс)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_centerline_before_after(results_by_bag, out_path):
    """До/после устойчивой подгонки: сырая centerline (с выбросами) слева,
    очищенная (робастная) справа — по каждому бэгу."""
    n = len(results_by_bag)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8), sharex=True)
    axes = np.atleast_2d(axes)
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (bag_name, (records, _)) in enumerate(results_by_bag.items()):
        depths_raw = np.array([(r["depth_lo"] + r["depth_hi"]) / 2 for r in records])
        centers_raw = np.array([r["rail_center"] for r in records])
        order = np.argsort(depths_raw)
        depths_raw, centers_raw = depths_raw[order], centers_raw[order]

        ax0 = axes[0, col]
        ax0.plot(depths_raw, centers_raw, marker='o', color='crimson')
        ax0.set_title(f"{bag_name}\nдо (сырая детекция)", fontsize=9)
        ax0.grid(True, alpha=0.3)

        records_sorted, is_outlier, corrected, coeffs = robust_centerline(records)
        depths_sorted = np.array([(r["depth_lo"] + r["depth_hi"]) / 2 for r in records_sorted])
        raw_sorted = np.array([r["rail_center"] for r in records_sorted])

        ax1 = axes[1, col]
        ax1.plot(depths_sorted, corrected, marker='o', color='darkgreen', label='устойчивая centerline')
        if is_outlier.any():
            ax1.scatter(depths_sorted[is_outlier], raw_sorted[is_outlier], marker='x',
                        s=70, c='black', zorder=5, label=f'выбросы ({is_outlier.sum()})')
        ax1.set_title("после (робастная подгонка)", fontsize=9)
        ax1.set_xlabel("глубина, м")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("X центра рельс, м")
    axes[1, 0].set_ylabel("X центра рельс, м")
    fig.suptitle("Centerline до/после отбраковки выбросов (робастная полиномиальная подгонка)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
