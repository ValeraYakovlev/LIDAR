#!/usr/bin/env python3
"""Эксперимент 20: почему Δs на синтетике ноль — пробник на профилях плотности.

Трекер (`ParallelGauge`) проходит запись как есть; перехватываются полосы кадра
(`band`: d, u, v), по которым измеритель §20 считает Δs. На них сравниваются:

- `base`  — корреляция профилей плотности, как в `shift.estimate_shift_density`;
- `sub`   — то же после вычета составляющей, которая стоит на месте относительно
            лидара: среднего профиля прошлых K кадров в координатах сенсора. Рисунок
            колец на гладкой обделке одинаков на каждом кадре и в среднем остаётся,
            а текстура тоннеля едет на Δs за кадр и в среднем размывается.

Правда — Δs по /tf (`exp20_truth.py`) на синтетике; на реальных записях правды
нет, там смотрится согласие с базой (где база работает, её ответ верный, §20).

    python exp20_ds_probe.py --dataset output/last_synth --bags conv_r300_a30 --max-frames 80
    python exp20_ds_probe.py --dataset /Volumes/T7/Dataset --bags doubleT_platform --max-frames 80
"""

import argparse
from pathlib import Path

import numpy as np

from exp20_split import guard
from rail_detection import bag_path, iter_frames
from rail_detection import shift as sh
from rail_detection.parallel_path import ParallelGauge

EDGES = np.arange(5.0, 60.0 + sh.D_BIN, sh.D_BIN)   # постоянная сетка в координатах сенсора


def profile(d):
    h = np.histogram(d, bins=EDGES)[0].astype(float)
    base = np.convolve(h, np.ones(sh.DENSITY_SMOOTH) / sh.DENSITY_SMOOTH, mode="same")
    return h / np.maximum(base, 1.0) - 1.0


def corr_shift(a, b, max_shift=sh.MAX_SHIFT):
    n_steps = int(round(max_shift / sh.D_BIN)) + 1
    score = np.zeros(n_steps)
    for i in range(n_steps):
        aa, bb = a[i:], b[:len(b) - i]
        n = min(len(aa), len(bb))
        aa, bb = aa[:n], bb[:n]
        sa, sb = aa.std(), bb.std()
        score[i] = float(np.dot(aa - aa.mean(), bb - bb.mean()) / (n * sa * sb)) if sa * sb > 0 else 0.0
    return sh._peak(np.maximum(score, 0.0), 0.0, sh.D_BIN), score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--bags", nargs="+", required=True)
    p.add_argument("--max-frames", type=int, default=80)
    p.add_argument("--K", type=int, nargs="+", default=[5, 10, 20])
    a = p.parse_args()
    guard(a.bags)
    for bag in a.bags:
        truth = None
        tp = Path(f"output/exp20_truth/{bag}.npz")
        if "last_synth" in a.dataset and tp.exists():
            truth = np.load(tp)["ds"]
        bands = []
        orig = sh.estimate_shift

        def spy(prev, cur, **kw):
            out = orig(prev, cur, **kw)
            bands.append((len(bands), prev, cur, out))
            return out

        import rail_detection.parallel_path as pp
        pg = ParallelGauge()
        rows = []
        import rail_detection.shift as shmod
        shmod.estimate_shift = spy
        try:
            for idx, points, n_total in iter_frames(bag_path(a.dataset, bag), max_frames=a.max_frames):
                n0 = len(bands)
                pg.update(points, steps=1)
                cur_band = pg.prev_band
                rows.append((idx, cur_band, bands[-1][3] if len(bands) > n0 else None))
                print(f"\r  {bag}: кадр {idx + 1}/{min(n_total, a.max_frames)}", end="", flush=True)
        finally:
            shmod.estimate_shift = orig
        print()
        profs = [profile(b["d"]) if b is not None else None for _, b, _ in rows]
        print(f"{bag}: кадр | правда | база (контраст) | " +
              " | ".join(f"вычет K={K} (контраст)" for K in a.K))
        errs = {"base": []}
        errs.update({K: [] for K in a.K})
        for k in range(1, len(rows)):
            if profs[k] is None or profs[k - 1] is None:
                continue
            t = truth[k] if truth is not None and k < len(truth) else np.nan
            (s0, c0), _ = corr_shift(profs[k - 1], profs[k])
            cells = [f"{s0:5.2f} ({c0:4.1f})"]
            errs["base"].append((s0, c0, t))
            for K in a.K:
                past = [q for q in profs[max(0, k - 1 - K):k - 1] if q is not None]
                if len(past) < 3:
                    cells.append("   —       ")
                    continue
                m = np.mean(past + [profs[k - 1]], axis=0)
                (s1, c1), _ = corr_shift(profs[k - 1] - m, profs[k] - m)
                cells.append(f"{s1:5.2f} ({c1:4.1f})")
                errs[K].append((s1, c1, t))
            if k % 5 == 0:
                print(f"  {rows[k][0]:4d} | {t:5.2f} | " + " | ".join(cells))
        for key, v in errs.items():
            v = np.array(v)
            if not len(v):
                continue
            ok = v[:, 1] >= sh.MIN_CONTRAST
            line = f"  {str(key):5s}: принято {ok.mean():.2f}"
            if truth is not None:
                e = np.abs(v[:, 0] - v[:, 2])
                line += (f", |ошибка| медиана {np.nanmedian(e):.2f}, среди принятых — "
                         f"{np.nanmedian(e[ok]) if ok.any() else np.nan:.2f}, "
                         f"доля принятых с ошибкой > 0.2 м: {np.mean(e[ok] > 0.2) if ok.any() else np.nan:.2f}")
            else:
                line += f", медиана сдвига {np.median(v[ok, 0]) if ok.any() else np.nan:.2f}"
            print(line)


if __name__ == "__main__":
    main()
