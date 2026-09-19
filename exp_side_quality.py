#!/usr/bin/env python3
"""Эксперимент 12, шаг 1: чем НЕВЕРНАЯ сторона отличается от верной — ОТСЕЧКА.

Замер при проектировании плана показал: в 67% кадров одна из стен по отдельности
объясняет ось лучше, чем обе вместе, но оракул, который каждый раз выбирает
правильную, выигрывает всего 6% медианы. Слепое же отключение стороны хуже пары
в полтора раза. Значит весь эксперимент упирается в один вопрос: **можно ли
узнать неверную сторону, не подглядывая в ответ.**

Если нельзя — плана 12 нет, и это результат. Отсечка проверяется здесь и только
здесь, до всякой реализации переключателя.

## Почему прежний замер ничего не показал

Медианы coverage, leak, числа сегментов и доли инлайеров совпали у верной и
неверной стороны до третьего знака. Причина методическая: они считались по ВСЕМ
кадрам, а в большинстве кадров обе стороны в порядке, признаки насыщены
(coverage = 1, leak = 0), и различать там нечего.

Здесь выборка сужена до кадров, где отключение стороны ДЕЙСТВИТЕЛЬНО помогает
больше чем на MARGIN, и сравниваются стороны внутри таких кадров. Плюс добавлены
два признака, которых раньше не было:

  несогласие сторон — насколько расходятся формы, подогнанные по левой стене и по
      правой по отдельности. Именно эта величина обязана выдавать асимметрию:
      когда обе стены на своих местах, формы совпадают; когда одна уехала на
      соседний путь, они расходятся, и величина расхождения говорит «здесь
      проблема» (но ещё не говорит, в какой из сторон);

  скачок полуширины сверх физически возможного — Δs известен (§20), тоннель
      неподвижен, поэтому граница может приблизиться ровно на Δs. Всё, что
      сверх, — движение самой оценки, а не тоннеля.

Насыщенные признаки берутся как СОБЫТИЯ («coverage упал ниже 1»), а не как
числа: у насыщенной величины медиана неинформативна, а доля срабатываний — нет.

Запуск:
    python exp_side_quality.py --stride 6
"""

import argparse

import numpy as np

from rail_detection import DEFAULT_BAGS, bag_path, fit_tunnel_geometry, iter_frames
from rail_detection.accumulate import track_band
from rail_detection.shift import estimate_shift
from rail_detection.tunnel_frame import (DEPTH_SCALE, RAIL_FIT_DEPTH,
                                         _axis_offset_coeffs, side_offset)

MARGIN = 0.20     # насколько отключение должно улучшить ось, чтобы считаться полезным
OTHER = {"left": "right", "right": "left"}


def holdout(res):
    """Ошибка оси на ОТЛОЖЕННЫХ рельсах — единственная честная мера здесь:
    рельсы глубже RAIL_FIT_DEPTH в подгонке не участвуют."""
    rd, ru = res.get("rails_all", (np.zeros(0), np.zeros(0)))
    far = rd > RAIL_FIT_DEPTH
    if far.sum() < 2:
        return None
    err = np.abs(ru[far] - np.polyval(_axis_offset_coeffs(res["shape"]), rd[far]))
    err = err[err < 1.0]
    return float(np.median(err)) if len(err) else None


def disagreement(one):
    """Насколько расходятся формы, подогнанные по одной и по другой стене:
    разница предсказанного увода оси вбок на DEPTH_SCALE метрах, в метрах.

    Это признак КАДРА, а не стороны: он говорит, что одна из сторон неправа, но
    не какая. Без него отключать нечего — при согласии сторон отключение вредит.
    """
    if any(one[s] is None for s in ("left", "right")):
        return None
    a = {s: _axis_offset_coeffs(one[s]["shape"]) for s in ("left", "right")}
    return float(abs(np.polyval(a["left"], DEPTH_SCALE) - np.polyval(a["right"], DEPTH_SCALE)))


def side_features(res, side, prev_width, shift):
    """Признаки одной стороны на одном кадре."""
    s = res[side]
    if s is None:
        return None
    d, inl = np.asarray(s["depths"], dtype=float), np.asarray(s["inliers"], dtype=bool)
    o = np.asarray(s["offsets"], dtype=float)
    pred = side_offset(res["shape"], side, d)
    resid = float(np.median(np.abs(o[inl] - pred[inl]))) if inl.any() else np.nan
    # Скачок полуширины сверх пройденного пути: тоннель неподвижен, граница может
    # приблизиться ровно на Δs, всё сверх — движение самой оценки.
    jump = np.nan
    if prev_width is not None and shift is not None and np.isfinite(shift):
        jump = abs(s["offset"] - prev_width)
    return {
        "cov_below_1": float(s["coverage"] < 1.0),
        "leak_nonzero": float(s["leak"] > 1e-6),
        "n_seg": float(len(s["widths"])),
        "offset": float(s["offset"]),
        "inlier_frac": float(inl.mean()) if len(inl) else 0.0,
        "resid": resid,
        "depth_span": float(np.ptp(d[inl])) if inl.sum() > 1 else 0.0,
        "width_jump": jump,
    }


def collect(dataset, bags, stride):
    rows, frames = [], []
    for bag in bags:
        prev_band, prev_w = None, {"left": None, "right": None}
        for idx, pts, _ in iter_frames(bag_path(dataset, bag), stride=stride):
            both = fit_tunnel_geometry(pts)
            if both is None:
                prev_band = None
                continue
            band = track_band(pts, both["frame"])
            shift = None
            if prev_band is not None:
                est = estimate_shift(prev_band, band)
                shift = est["shift"] if est["ok"] else None
            prev_band = band

            one = {s: fit_tunnel_geometry(pts, sides={s}) for s in ("left", "right")}
            e_both = holdout(both)
            e_one = {s: (holdout(one[s]) if one[s] is not None else None)
                     for s in ("left", "right")}
            if e_both is None or any(v is None for v in e_one.values()):
                continue

            dis = disagreement(one)
            frames.append({"bag": bag, "idx": idx, "e_both": e_both, "dis": dis,
                           **{f"e_{s}": e_one[s] for s in ("left", "right")}})
            for s in ("left", "right"):
                f = side_features(both, s, prev_w[s], shift)
                if f is None:
                    continue
                # «Отключить сторону s» = подгонка по ДРУГОЙ стороне
                gain = (e_both - e_one[OTHER[s]]) / max(e_both, 1e-9)
                rows.append({"bag": bag, "idx": idx, "side": s, "gain": gain,
                             "drop_helps": float(gain > MARGIN), "dis": dis, **f})
            for s in ("left", "right"):
                prev_w[s] = both[s]["offset"] if both[s] is not None else None
    return rows, frames


def auc(a, b, n=200000, seed=0):
    """Вероятность, что признак у «вредной» стороны больше, чем у полезной.
    Совпадения делятся пополам — иначе у насыщенных величин мера вырождается."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    rng = np.random.default_rng(seed)
    x, y = rng.choice(a, n, replace=True), rng.choice(b, n, replace=True)
    return float(np.mean(x > y) + 0.5 * np.mean(x == y))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bags", nargs="*", default=list(DEFAULT_BAGS))
    p.add_argument("--stride", type=int, default=6)
    a = p.parse_args()

    rows, frames = collect(a.dataset, a.bags, a.stride)
    if not rows:
        print("нет данных")
        return
    keys = ("cov_below_1", "leak_nonzero", "n_seg", "offset", "inlier_frac",
            "resid", "depth_span", "width_jump")
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    helps = np.array([r["drop_helps"] for r in rows], dtype=bool)
    gain = np.array([r["gain"] for r in rows], dtype=float)

    print(f"кадров {len(frames)}, сторон {len(rows)}")
    print(f"отключение помогает больше {MARGIN:.0%}: {helps.sum()} сторон "
          f"({100 * helps.mean():.1f}%)\n")

    print("=== Признак стороны: отличается ли ВРЕДНАЯ сторона от полезной ===")
    print("(вредная = её отключение улучшает ось; выборка сужена до кадров,")
    print(" где хоть одно отключение помогает — иначе различать нечего)")
    # Внутри кадра сравниваем сторону, которую полезно убрать, с её соседкой
    by_frame = {}
    for r in rows:
        by_frame.setdefault((r["bag"], r["idx"]), []).append(r)
    bad, good = {k: [] for k in keys}, {k: [] for k in keys}
    n_useful = 0
    for pair in by_frame.values():
        if len(pair) != 2 or not any(x["drop_helps"] for x in pair):
            continue
        n_useful += 1
        worst = max(pair, key=lambda x: x["gain"])
        best = min(pair, key=lambda x: x["gain"])
        for k in keys:
            bad[k].append(worst[k])
            good[k].append(best[k])
    print(f"таких кадров: {n_useful}\n")
    print(f"{'признак':16s} {'вредная':>10s} {'полезная':>10s} {'AUC':>7s}")
    best_auc = 0.5
    for k in keys:
        b, g = np.array(bad[k], dtype=float), np.array(good[k], dtype=float)
        v = auc(b, g)
        if np.isfinite(v):
            best_auc = max(best_auc, max(v, 1 - v))
        bm = np.nanmedian(b) if len(b) else np.nan
        gm = np.nanmedian(g) if len(g) else np.nan
        print(f"{k:16s} {bm:10.3f} {gm:10.3f} {v:7.3f}")

    print("\n=== Признак КАДРА: предсказывает ли несогласие сторон пользу от отключения ===")
    dis = np.array([f["dis"] for f in frames], dtype=float)
    win = np.array([max(f["e_both"] - min(f["e_left"], f["e_right"]), 0.0) / max(f["e_both"], 1e-9)
                    for f in frames])
    ok = np.isfinite(dis)
    if ok.sum() > 20:
        q = np.percentile(dis[ok], [25, 50, 75])
        print(f"несогласие сторон, м на 40 м: квартили {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}")
        for lo, hi, name in ((0, q[0], "малое"), (q[0], q[1], "ниже среднего"),
                             (q[1], q[2], "выше среднего"), (q[2], np.inf, "большое")):
            m = ok & (dis >= lo) & (dis < hi)
            if m.sum() < 5:
                continue
            print(f"  {name:14s} (n={m.sum():3d}): выигрыш от оракула "
                  f"{100 * np.median(win[m]):5.1f}%, доля кадров с пользой >{MARGIN:.0%}: "
                  f"{100 * np.mean(win[m] > MARGIN):4.0f}%")
        c = np.corrcoef(dis[ok], win[ok])[0, 1]
        print(f"  корреляция несогласия с выигрышем: {c:+.3f}")

    print(f"\n=== ОТСЕЧКА ===")
    print(f"лучший признак стороны: AUC {best_auc:.3f} (0.5 — не различает)")
    if best_auc < 0.65:
        print("НЕ ПРОЙДЕНА: выбрать неверную сторону нечем. Отключение стороны")
        print("без детектора хуже пары в полтора раза — эксперимент 12 закрывается.")
    else:
        print("пройдена: признак есть, можно строить переключатель (шаг 2)")


if __name__ == "__main__":
    main()
