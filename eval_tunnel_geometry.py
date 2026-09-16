#!/usr/bin/env python3
"""Оценка rail-guided геометрии тоннеля (rail_detection.tunnel_frame) на
ЗАФИКСИРОВАННОЙ тестовой выборке, с прямым сравнением со старыми методами.

Про метрику. Прежняя метрика (validate_geometry.py) считала кадр пройденным,
если подгонка стены получилась ГЛАДКОЙ. Гладкость не отличает верную стену от
неверной: прямая, проведённая через кабельный лоток в метре от центра, ровно так
же гладкая, и именно поэтому прежние 90% уживались с картинками, где линия
идёт заметно внутри тоннеля. Здесь метрика привязана к точкам:

  coverage — доля срезов, где стена реально наблюдается (рядом с предсказанным
             положением есть точки). Ловит кривую, проведённую через пустоту.
  leak     — доля точек ЗА стеной дальше чем на 30 см, посчитанная внутри
             каждого среза и усреднённая по срезам (плотность точек падает как
             1/r², поэтому доля по всем точкам сразу мерила бы только ближние
             метры). Настоящая стена непрозрачна, за ней точек почти нет.

Обе считаются одинаково для всех сравниваемых методов, поэтому сравнение
честное. Порог по leak намеренно строгий: именно его старые методы и не проходят.

Запуск:
    python eval_tunnel_geometry.py --test-set test_set.json
    python eval_tunnel_geometry.py --test-set hidden_test_set.json --compare
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from rail_detection import (
    DEFAULT_DEPTH_BINS,
    WALL_DEPTH_BINS,
    analyze_walls,
    bag_path,
    combined_wall_fit,
    eval_fit,
    fit_straight_or_arc,
    fit_tunnel_geometry,
    iter_selected_frames,
    slope_from_fit,
    to_track_coords,
)
from rail_detection.tunnel_frame import V_HI, V_LO, wall_metrics

MIN_COVERAGE = 0.70
MAX_LEAK = 0.15
OFFSET_RANGE = (1.2, 6.5)


def side_ok(m):
    return (
        m is not None
        and m["coverage"] >= MIN_COVERAGE
        and m["leak"] <= MAX_LEAK
        and OFFSET_RANGE[0] <= m["offset"] <= OFFSET_RANGE[1]
    )


def _band(points, res):
    """Точки клиренс-полосы в координатах пути — общая система отсчёта, в
    которой меряются все методы."""
    x = points['x'].astype(float)
    y = points['y'].astype(float)
    z = points['z'].astype(float)
    d, u, v = to_track_coords(x, y, z, res["frame"])
    band = (v >= V_LO) & (v <= V_HI) & (d > 2) & (d < 45)
    return d[band], u[band]


def measure_new(points, res):
    out = {}
    db, ub = _band(points, res)
    for side, sign in (("left", -1.0), ("right", +1.0)):
        s = res[side]
        if s is None:
            out[side] = None
            continue
        out[side] = {"coverage": s["coverage"], "leak": s["leak"], "offset": s["offset"]}
    return out


def measure_sensor_fits(points, res, left_fit, right_fit):
    """Меряет теми же метриками метод, который выдаёт стену в координатах
    СЕНСОРА (старые combined_wall_fit / analyze_walls): его кривая переводится
    в координаты пути и проверяется по тем же точкам."""
    db, ub = _band(points, res)
    frame = res["frame"]
    out = {}
    for side, sign, fit in (("left", -1.0, left_fit), ("right", +1.0, right_fit)):
        if fit is None or fit.get("kind") is None:
            out[side] = None
            continue

        def predict(dd, _f=fit, _s=sign):
            xc = eval_fit(frame["axis_fit"], dd)
            heading = np.arctan(slope_from_fit(frame["axis_fit"], dd))
            return _s * (eval_fit(_f, dd) - xc) * np.cos(heading)

        m = wall_metrics(db, ub, sign, predict, WALL_DEPTH_BINS)
        m["offset"] = float(np.nanmedian(predict(np.linspace(4, 40, 20))))
        out[side] = m
    return out


def old_combined(points, res):
    rail_records = res["frame"]["rail_records"]
    floor_z = float(np.mean([r["shoulder_z"] for r in rail_records]))
    r = combined_wall_fit(points, floor_z, DEFAULT_DEPTH_BINS)
    if r is None:
        return None, None
    return r["left_fit"], r["right_fit"]


def old_density(points, res):
    rail_records = res["frame"]["rail_records"]
    by_depth = {(r["depth_lo"], r["depth_hi"]): r for r in rail_records}
    wr = analyze_walls(points, by_depth, DEFAULT_DEPTH_BINS)
    if len(wr) < 4:
        return None, None
    d = [(w["depth_lo"] + w["depth_hi"]) / 2 for w in wr]
    lf = fit_straight_or_arc(d, [w["left_wall_x"] for w in wr])
    rf = fit_straight_or_arc(d, [w["right_wall_x"] for w in wr])
    return lf, rf


def iter_test_frames(dataset, test_set):
    """Проход по выборке с группировкой по bag: один проход по файлу на bag
    вместо одного полного чтения на каждый кадр."""
    by_bag = defaultdict(list)
    for item in test_set:
        by_bag[item["bag"]].append(item["frame"])
    for bag, frames in by_bag.items():
        for idx, points in iter_selected_frames(bag_path(dataset, bag), frames):
            yield bag, idx, points


def evaluate(dataset, test_set_path, compare=False, verbose=True, collect=None):
    with open(test_set_path) as f:
        test_set = json.load(f)["test_set"]

    stats = {k: {"both": 0, "any": 0} for k in ("new", "combined", "density")}
    n_frames, n_no_frame = 0, 0
    fails = []

    for bag, idx, points in iter_test_frames(dataset, test_set):
        n_frames += 1
        res = fit_tunnel_geometry(points, WALL_DEPTH_BINS)
        if res is None:
            n_no_frame += 1
            fails.append((bag, idx, "не удалось построить опору по рельсам"))
            continue

        m_new = measure_new(points, res)
        graded = {"new": m_new}
        if compare:
            graded["combined"] = measure_sensor_fits(points, res, *old_combined(points, res))
            graded["density"] = measure_sensor_fits(points, res, *old_density(points, res))

        for k, m in graded.items():
            l_ok, r_ok = side_ok(m.get("left")), side_ok(m.get("right"))
            stats[k]["both"] += int(l_ok and r_ok)
            stats[k]["any"] += int(l_ok or r_ok)

        if not (side_ok(m_new.get("left")) and side_ok(m_new.get("right"))):
            def d(s):
                return "нет" if s is None else f"cov={s['coverage']:.2f} leak={s['leak']:.3f} u={s['offset']:.2f}"
            fails.append((bag, idx, f"L: {d(m_new.get('left'))} | R: {d(m_new.get('right'))}"))

        if collect is not None:
            collect.append((bag, idx, points, res, m_new))

    if verbose:
        print(f"\nВыборка {test_set_path}: {n_frames} кадров "
              f"(порог: coverage>={MIN_COVERAGE}, leak<={MAX_LEAK})")
        names = {"new": "rail-guided (новый)", "combined": "combined_wall_fit (старый)",
                 "density": "плотность+подгонка (старый)"}
        for k in ("new", "combined", "density") if compare else ("new",):
            s = stats[k]
            print(f"  {names[k]:34s} обе стены: {s['both']:3d}/{n_frames} = {100*s['both']/n_frames:5.1f}%"
                  f"   хотя бы одна: {s['any']:3d}/{n_frames} = {100*s['any']/n_frames:5.1f}%")
        if n_no_frame:
            print(f"  (в {n_no_frame} кадрах не построилась опора по рельсам)")
        print("\nКадры, где новый метод не дал обе стены:")
        for b, i, r in fails[:25]:
            print(f"  {b} #{i}: {r}")
        if len(fails) > 25:
            print(f"  ... ещё {len(fails) - 25}")
    return stats, n_frames, fails


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--test-set", default="test_set.json")
    p.add_argument("--compare", action="store_true", help="мерить теми же метриками старые методы")
    a = p.parse_args()
    evaluate(a.dataset, a.test_set, compare=a.compare)
