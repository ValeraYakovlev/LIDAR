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
    WALL_DEPTH_BINS_DEEP,
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
from rail_detection.tracker import TunnelTracker
from rail_detection.tunnel_frame import (RAIL_FIT_DEPTH, V_HI, V_LO,
                                         _axis_offset_coeffs, tracked_depth,
                                         tunnel_center_coeffs, wall_metrics)

MIN_COVERAGE = 0.70
# Полуширина меньше этого физически невозможна: габарит подвижного состава шире.
# Служит только диагностикой в отчёте; решение о запрете — эксперимент 7.
MIN_HALF_WIDTH = 1.60
MAX_LEAK = 0.15
OFFSET_RANGE = (1.2, 6.5)


def side_ok(m):
    return (
        m is not None
        and m["coverage"] >= MIN_COVERAGE
        and m["leak"] <= MAX_LEAK
        and OFFSET_RANGE[0] <= m["offset"] <= OFFSET_RANGE[1]
    )


def axis_holdout_error(res):
    """Ошибка оси на ОТЛОЖЕННЫХ рельсах — тех, что глубже RAIL_FIT_DEPTH и в
    подгонке не участвовали. Отвечает на вопрос, ради которого рельсы и
    увязывались с изгибом тоннеля: предсказывает ли найденная по стенам
    кривизна, куда на самом деле уходит путь. Метрика стен на это ответить не
    может — она про границы, а не про ось.

    Возвращает медиану |ошибки| по отложенным срезам либо None.
    """
    rd, ru = res.get("rails_all", (np.zeros(0), np.zeros(0)))
    far = rd > RAIL_FIT_DEPTH
    if far.sum() < 2:
        return None
    pred = np.polyval(_axis_offset_coeffs(res["shape"]), rd[far])
    err = np.abs(ru[far] - pred)
    # Одиночные грубые промахи детектора рельс (> 1 м) — не ошибка оси
    err = err[err < 1.0]
    return float(np.median(err)) if len(err) else None


def axis_slope(res, depth=20.0):
    """Наклон ИТОГОВОЙ оси пути на фиксированной глубине — то, что метод
    утверждает про направление движения. Берётся на 20 м, а не у самого поезда:
    вблизи наклон почти целиком определяется опорой по рельсам, а спорная часть
    (добавка по стенам) проявляется дальше."""
    return float(np.polyval(np.polyder(tunnel_center_coeffs(res)), depth))


def segment_signature(res):
    """Сколько участков полуширины на каждой стороне.

    Считается ЧИСЛО участков, а не положения границ. Первая версия сравнивала
    округлённые до метра границы и оказалась бракованной: разрыв ширины — это
    место в тоннеле, и при движении вперёд оно ЗАКОННО приближается на Δs
    (полтора метра за кадр), так что его округлённое положение меняется каждый
    кадр. Метрика засчитывала это как перестройку и показывала 32% churn там,
    где ничего не перестраивалось.

    Число участков такой подделке не подвержено: оно меняется, только когда
    модель реально решила, что ступеней стало больше или меньше, — а вот это
    за 0.2 с действительно произойти не может.
    """
    return tuple(0 if res[side] is None else len(res[side]["widths"])
                 for side in ("left", "right"))


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
    вместо одного полного чтения на каждый кадр.

    Отдаёт ещё и флаг начала записи со её шагом — временному трекингу нужно
    знать, где сбросить состояние и сколько кадров прошло между вызовами.
    """
    by_bag = defaultdict(list)
    for item in test_set:
        by_bag[item["bag"]].append(item["frame"])
    for bag, frames in by_bag.items():
        step = (frames[1] - frames[0]) if len(frames) > 1 else 1
        first = True
        for idx, points in iter_selected_frames(bag_path(dataset, bag), frames):
            yield bag, idx, points, first, step
            first = False


def evaluate(dataset, test_set_path, compare=False, verbose=True, collect=None,
             use_rails=True, rail_roles=None, track=False, surface_tol=None,
             accumulate=None, deep=False, beta_prior=None, sticky=None,
             min_half_width=None):
    with open(test_set_path) as f:
        test_set = json.load(f)["test_set"]
    bins = WALL_DEPTH_BINS_DEEP if deep else WALL_DEPTH_BINS

    stats = {k: {"both": 0, "any": 0} for k in ("new", "combined", "density")}
    per_bag = defaultdict(lambda: {"n": 0, "both": 0})
    n_frames, n_no_frame = 0, 0
    fails = []
    axis_errors = []
    reaches = []
    jumps = []        # покадровый скачок наклона оси
    restructures = [] # сменилась ли структура разрывов ширины между кадрами
    too_narrow = 0    # сегментов уже физически возможного
    prev = None

    tracker = TunnelTracker(**(accumulate or {})) if track else None
    for bag, idx, points, first, step in iter_test_frames(dataset, test_set):
        n_frames += 1
        per_bag[bag]["n"] += 1
        # Переопределения передаются, только если заданы: иначе CLI со своим
        # default=0 молча подменял бы умолчания модуля, и выбранная в
        # эксперименте настройка не доезжала бы до подгонки.
        kw = dict(use_rails=use_rails, surface_tol=surface_tol,
                  **({} if rail_roles is None else {'rail_roles': rail_roles}),
                  **{k: v for k, v in (("beta_prior", beta_prior), ("sticky", sticky),
                                       ("min_half_width", min_half_width))
                     if v is not None})
        if tracker is not None:
            if first:
                tracker.reset()
            res = tracker.update(points, steps=step, depth_bins=bins, **kw)
        else:
            res = fit_tunnel_geometry(points, bins, **kw)
        if res is None:
            n_no_frame += 1
            fails.append((bag, idx, "не удалось построить опору"))
            prev = None
            continue

        # Дрожание геометрии между кадрами. Метрика стен к нему почти слепа
        # (§18: она не заметила опоры, уехавшей на 2.57 м), а именно оно и
        # видно на GIF как перекос кадра и перестройка границы.
        cur = {"slope": axis_slope(res), "segs": segment_signature(res)}
        if prev is not None and not first:
            jumps.append(abs(cur["slope"] - prev["slope"]))
            restructures.append(int(cur["segs"] != prev["segs"]))
        prev = cur
        for side in ("left", "right"):
            if res[side] is not None:
                too_narrow += sum(1 for w in res[side]["widths"] if w < MIN_HALF_WIDTH)

        m_new = measure_new(points, res)
        graded = {"new": m_new}
        if compare:
            graded["combined"] = measure_sensor_fits(points, res, *old_combined(points, res))
            graded["density"] = measure_sensor_fits(points, res, *old_density(points, res))

        for k, m in graded.items():
            l_ok, r_ok = side_ok(m.get("left")), side_ok(m.get("right"))
            stats[k]["both"] += int(l_ok and r_ok)
            stats[k]["any"] += int(l_ok or r_ok)
            if k == "new":
                per_bag[bag]["both"] += int(l_ok and r_ok)

        seen = [tracked_depth(res, s) for s in ("left", "right")]
        seen = [s for s in seen if s]
        if seen:
            reaches.append(float(np.mean(seen)))

        if not (side_ok(m_new.get("left")) and side_ok(m_new.get("right"))):
            def d(s):
                return "нет" if s is None else f"cov={s['coverage']:.2f} leak={s['leak']:.3f} u={s['offset']:.2f}"
            fails.append((bag, idx, f"L: {d(m_new.get('left'))} | R: {d(m_new.get('right'))}"))

        ae = axis_holdout_error(res)
        if ae is not None:
            axis_errors.append(ae)

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
        if axis_errors:
            a = np.array(axis_errors)
            print(f"  ось на отложенных рельсах (>{RAIL_FIT_DEPTH:.0f} м): медиана "
                  f"{np.median(a):.3f} м, 90-й перцентиль {np.percentile(a, 90):.3f} м "
                  f"({len(a)} кадров)")
        if jumps:
            j = np.array(jumps)
            print(f"  скачок наклона оси между кадрами: медиана {np.median(j):.5f}, "
                  f"90-й перцентиль {np.percentile(j, 90):.5f}")
        if restructures:
            print(f"  число участков ширины меняется в "
                  f"{100 * np.mean(restructures):.1f}% переходов"
                  + (f"; сегментов уже {MIN_HALF_WIDTH} м: {too_narrow}" if too_narrow else ""))
        if reaches:
            rr = np.array(reaches)
            print(f"  дальность наблюдения границы: медиана {np.median(rr):.1f} м, "
                  f"90-й перцентиль {np.percentile(rr, 90):.1f} м")
        if len(per_bag) > 1:
            print("  по прогонам:")
            for b, s in sorted(per_bag.items()):
                print(f"    {b:38s} {s['both']:3d}/{s['n']:3d} = {100*s['both']/s['n']:5.1f}%")
        if n_no_frame:
            print(f"  (в {n_no_frame} кадрах не построилась опора)")
        print("\nКадры, где новый метод не дал обе стены:")
        for b, i, r in fails[:25]:
            print(f"  {b} #{i}: {r}")
        if len(fails) > 25:
            print(f"  ... ещё {len(fails) - 25}")
    return {"stats": stats, "n_frames": n_frames, "fails": fails, "per_bag": dict(per_bag),
            "axis": axis_errors, "reach": reaches}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--test-set", default="test_set.json")
    p.add_argument("--compare", action="store_true", help="мерить теми же метриками старые методы")
    p.add_argument("--track", action="store_true",
                   help="связывать кадры между собой (rail_detection.tracker)")
    p.add_argument("--rail-roles", nargs="*", default=None,
                   help="аблация: какие роли играют рельсы (floor axis rows)")
    p.add_argument("--no-rails", action="store_true",
                   help="стресс-проверка: отключить рельсы и мерить безрельсовый путь")
    p.add_argument("--surface-tol", type=float, default=None,
                   help="эксп.2: брать в оценку границы только точки сплошной "
                        "поверхности (порог деприцированного скачка, м)")
    p.add_argument("--accumulate", type=int, default=None,
                   help="сколько кадров складывать со сдвигом на Δs "
                        "(по умолчанию как в rail_detection.tracker; 1 — без накопления)")
    p.add_argument("--beta-prior", type=float, default=None,
                   help="эксп.6: притяжение курса к курсу опоры по рельсам (0 — как было)")
    p.add_argument("--sticky", type=float, default=None,
                   help="эксп.7: удержание структуры разрывов ширины (0 — выключено)")
    p.add_argument("--min-half-width", type=float, default=None,
                   help="эксп.7: физический предел полуширины снизу, м (0 — выключен)")
    p.add_argument("--deep", action="store_true",
                   help="эксп.3: продлить срезы подгонки с 42 до 62 м")
    a = p.parse_args()
    evaluate(a.dataset, a.test_set, compare=a.compare, use_rails=not a.no_rails,
             rail_roles=a.rail_roles, track=a.track, surface_tol=a.surface_tol,
             accumulate=({"accumulate": a.accumulate} if a.accumulate else None),
             deep=a.deep, beta_prior=a.beta_prior, sticky=a.sticky,
             min_half_width=a.min_half_width)
