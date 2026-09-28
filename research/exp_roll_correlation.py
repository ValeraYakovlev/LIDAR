#!/usr/bin/env python3
"""Корреляция крена рельсов и крена свода.

Вопрос: наклонён ли свод вместе с путём? Если да — тоннель строился под
возвышение наружного рельса, и крен можно брать с любого из двух. Если нет —
путь накренён внутри ровного тоннеля, и габарит вагона надо наклонять по
рельсам, а не по своду.

Три меры связи, и у каждой свой смысл:

  по срезам — пары (крен рельсов, крен свода) в одном и том же срезе одного
      кадра. Самая прямая, но и самая шумная: по одному срезу крен рельсов
      меряется с ошибкой порядка градуса.

  по кадрам — медианы по срезам кадра. Шум срезов гасится, остаётся крен
      участка пути.

  по пикам — взаимная корреляция двух рядов вдоль записи со сдвигом. Если крен
      свода и крен рельсов действительно одно и то же, их пики на кривых
      совпадают по времени и максимум взаимной корреляции стоит на нулевом
      сдвиге. Сдвинутый максимум означал бы, что свод и рельсы видятся на
      разных глубинах или что это вообще разные величины.

Запуск:
    python exp_roll_correlation.py
"""

import argparse
from pathlib import Path

import numpy as np

from rail_detection import DEFAULT_BAGS, bag_path, iter_frames
from rail_detection.detector import analyze_frame
from rail_detection.roll import frame_roll

BAGS = list(DEFAULT_BAGS) + ["doubleT_obstacle"]
STRIDES = {"doubleT_platform": 2, "roundT_doubleT": 2, "roundT_pressureGate_roundT": 2,
           "roundT_squareT_pressureGate_squareT": 4, "squareT_platform_squareT_switch": 6,
           "doubleT_obstacle": 2}
MAX_LAG = 15


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def cross_corr(a, b, max_lag=MAX_LAG):
    """Взаимная корреляция рядов вдоль записи. Пропуски (NaN) исключаются
    попарно для каждого сдвига. Положительный сдвиг — свод отстаёт от рельсов."""
    out = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[:len(a) - lag], b[lag:]
        else:
            x, y = a[-lag:], b[:len(b) + lag]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            out.append(np.nan)
            continue
        out.append(float(np.corrcoef(x[ok], y[ok])[0, 1]))
    return np.arange(-max_lag, max_lag + 1), np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bags", nargs="*", default=BAGS)
    p.add_argument("--plot", default="output/roll_correlation.png")
    a = p.parse_args()

    runs = {}
    for bag in a.bags:
        stride = STRIDES.get(bag, 2)
        frames, pairs = [], []
        for idx, pts, _ in iter_frames(bag_path(a.dataset, bag), stride=stride):
            recs, _ = analyze_frame(pts)
            fr = frame_roll(pts, recs)
            frames.append((idx, fr["rail_med"], fr["ceil_med"],
                           [ro for _, ro in fr["rail"]]))
            pairs += fr["pairs"]
            print(f"\r  {bag}: кадр {idx}", end="", flush=True)
        print()
        runs[bag] = {"frames": frames, "pairs": pairs, "stride": stride}

    deg = np.degrees
    print("\n=== Величина крена по рельсам, градусы (медиана по кадру) ===")
    print(f"{'прогон':38s} {'кадров':>7s} {'медиана':>8s} {'5%':>7s} {'95%':>7s} "
          f"{'шум среза':>10s}")
    for bag, r in runs.items():
        rail = np.array([f[1] for f in r["frames"] if f[1] is not None])
        if not len(rail):
            print(f"{bag:38s} нет рельсов")
            continue
        # Шум одного среза — отклонение от медианы СВОЕГО кадра: внутри кадра
        # крен пути почти постоянен, так что разброс там и есть ошибка замера.
        dev = [ro - f[1] for f in r["frames"] if f[1] is not None for ro in f[3]]
        noise = 1.4826 * np.median(np.abs(dev)) if dev else np.nan
        print(f"{bag:38s} {len(rail):7d} {deg(np.median(rail)):8.2f} "
              f"{deg(np.percentile(rail, 5)):7.2f} {deg(np.percentile(rail, 95)):7.2f} "
              f"{deg(noise):10.2f}")

    print("\n=== Связь крена рельсов и крена свода ===")
    print(f"{'прогон':38s} {'пар срез':>9s} {'r срез':>7s} {'пар кадр':>9s} {'r кадр':>7s} "
          f"{'ρ кадр':>7s} {'наклон':>7s} {'пик ВКФ':>8s} {'сдвиг':>6s}")
    pooled_r, pooled_c = [], []
    for bag, r in runs.items():
        P = np.array(r["pairs"]) if r["pairs"] else np.zeros((0, 3))
        r_slice = float(np.corrcoef(P[:, 1], P[:, 2])[0, 1]) if len(P) > 5 else np.nan
        F = [(f[1], f[2]) for f in r["frames"] if f[1] is not None and f[2] is not None]
        F = np.array(F) if F else np.zeros((0, 2))
        if len(F) > 5:
            r_fr = float(np.corrcoef(F[:, 0], F[:, 1])[0, 1])
            rho = spearman(F[:, 0], F[:, 1])
            slope = float(np.polyfit(F[:, 0], F[:, 1], 1)[0])
            pooled_r += list(F[:, 0])
            pooled_c += list(F[:, 1])
        else:
            r_fr = rho = slope = np.nan
        ra = np.array([np.nan if f[1] is None else f[1] for f in r["frames"]])
        ca = np.array([np.nan if f[2] is None else f[2] for f in r["frames"]])
        lags, cc = cross_corr(ra, ca)
        if np.isfinite(cc).any():
            # Пик по МОДУЛЮ: связь может быть и обратной, и тогда максимум ВКФ
            # ничего не значит, а минимум — главное, что в ней есть.
            k = int(np.nanargmax(np.abs(cc)))
            peak, lag = cc[k], int(lags[k])
        else:
            peak, lag = np.nan, 0
        r["cc"] = (lags, cc)
        print(f"{bag:38s} {len(P):9d} {r_slice:7.2f} {len(F):9d} {r_fr:7.2f} {rho:7.2f} "
              f"{slope:7.2f} {peak:8.2f} {lag:+6d}")
    pr, pc = np.array(pooled_r), np.array(pooled_c)
    if len(pr) > 5:
        print(f"{'ВСЕ ВМЕСТЕ (по кадрам)':38s} {'':9s} {'':7s} {len(pr):9d} "
              f"{np.corrcoef(pr, pc)[0, 1]:7.2f} {spearman(pr, pc):7.2f} "
              f"{np.polyfit(pr, pc, 1)[0]:7.2f}")
    print("\nr — Пирсон, ρ — Спирмен, наклон — МНК крена свода по крену рельсов;")
    print("пик ВКФ — экстремум взаимной корреляции по модулю, со знаком; сдвиг в кадрах выборки")

    _plot(a.plot, runs)


def _plot(path, runs):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    n = len(runs)
    fig = Figure(figsize=(12, 2.3 * n + 0.5), dpi=120)
    FigureCanvasAgg(fig)
    for i, (bag, r) in enumerate(runs.items()):
        ax = fig.add_subplot(n, 3, 3 * i + 1)
        idx = [f[0] for f in r["frames"]]
        ax.plot(idx, [np.degrees(f[1]) if f[1] is not None else np.nan for f in r["frames"]],
                c="#c8a45a", lw=1.2, label="рельсы")
        ax.plot(idx, [np.degrees(f[2]) if f[2] is not None else np.nan for f in r["frames"]],
                c="#5a8fc8", lw=1.0, label="свод")
        ax.axhline(0, c="gray", lw=0.6)
        ax.set_title(bag, fontsize=8)
        ax.set_ylabel("крен, °", fontsize=7)
        ax.tick_params(labelsize=6.5)
        if i == 0:
            ax.legend(fontsize=7)

        ax2 = fig.add_subplot(n, 3, 3 * i + 2)
        F = [(f[1], f[2]) for f in r["frames"] if f[1] is not None and f[2] is not None]
        if F:
            F = np.degrees(np.array(F))
            ax2.scatter(F[:, 0], F[:, 1], s=6, c="#444", alpha=0.6)
            lim = max(1.0, np.abs(F).max() * 1.1)
            ax2.plot([-lim, lim], [-lim, lim], c="gray", lw=0.6, ls="--")
            ax2.set_xlim(-lim, lim)
            ax2.set_ylim(-lim, lim)
        ax2.set_xlabel("крен рельсов, °", fontsize=7)
        ax2.set_ylabel("крен свода, °", fontsize=7)
        ax2.tick_params(labelsize=6.5)

        ax3 = fig.add_subplot(n, 3, 3 * i + 3)
        lags, cc = r.get("cc", (np.zeros(0), np.zeros(0)))
        ax3.plot(lags, cc, c="#d1495b", lw=1.2)
        ax3.axvline(0, c="gray", lw=0.6)
        ax3.set_ylim(-1, 1)
        ax3.set_xlabel("сдвиг, кадров выборки", fontsize=7)
        ax3.set_ylabel("ВКФ", fontsize=7)
        ax3.tick_params(labelsize=6.5)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"\nкартинка: {path}")


if __name__ == "__main__":
    main()
