#!/usr/bin/env python3
"""Эксперимент 3, шаг 4: обязательная проверка накопления на препятствии.

Накопление складывает кадры, сдвигая их на пройденный путь. Для неподвижной
геометрии тоннеля это законно, но объект, который движется ОТНОСИТЕЛЬНО тоннеля,
сдвигается не так, и накопление его размажет. Это неустранимое свойство метода,
а не дефект реализации, поэтому вопрос не «размазывает ли», а «насколько и что
из этого следует».

Проверка идёт по `doubleT_obstacle` — единственной записи с реальным
препятствием (стоящий поезд примерно в 3.3 м, knowledge.md §6). Меряется то, что
нужно детектору препятствий, а не геометрии: расстояние до ближайшей точки в
коридоре по курсу и продольная протяжённость её скопления. Если накопление
препятствие растягивает, протяжённость вырастет, а расстояние поедет.

Отдельно печатается аналитическая оценка размазывания для объекта, движущегося
относительно тоннеля: данных с таким объектом в датасете нет, но величина
считается точно и без эксперимента.

Запуск:
    python exp_obstacle_check.py
"""

import argparse

import numpy as np

from rail_detection import bag_path, fit_tunnel_geometry, iter_frames
from rail_detection.accumulate import FrameAccumulator, N_FRAMES

CORRIDOR_X = 1.0     # м: полуширина коридора прямо по курсу
CORRIDOR_Z = 2.0     # м: полувысота того же коридора
SELF_MASK = 2.0      # м: ближе этого — собственная конструкция поезда (§12: x≈0.95, range≈0.96)


def corridor(points):
    """Точки в узком коридоре по курсу — там, где препятствие и ищут."""
    x = points['x'].astype(float)
    z = points['z'].astype(float)
    d = -points['y'].astype(float)
    sel = (np.abs(x) < CORRIDOR_X) & (np.abs(z) < CORRIDOR_Z) & (d > SELF_MASK)
    return d[sel]


def describe(ds):
    """Расстояние до препятствия и продольная протяжённость его скопления."""
    if len(ds) < 20:
        return None
    near = float(np.percentile(ds, 1))
    close = ds[ds < near + 1.0]
    return {"n": len(ds), "dist": near, "extent": float(np.ptp(close)) if len(close) else 0.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bag", default="doubleT_obstacle")
    p.add_argument("--n-frames", type=int, default=N_FRAMES)
    p.add_argument("--max-frames", type=int, default=120)
    a = p.parse_args()

    acc = FrameAccumulator(n_frames=a.n_frames)
    solo_rows, merged_rows, shifts, sizes = [], [], [], []
    for idx, points, _ in iter_frames(bag_path(a.dataset, a.bag), stride=1,
                                      max_frames=a.max_frames):
        res = fit_tunnel_geometry(points)
        if res is None:
            continue
        merged = acc.push(points, res["frame"], steps=1)
        s = describe(corridor(points))
        m = describe(corridor(merged))
        if s and m:
            solo_rows.append(s)
            merged_rows.append(m)
            sizes.append(len(merged) / len(points))
        if np.isfinite(acc.last.get("shift", np.nan)):
            shifts.append(acc.last["shift"])

    if not solo_rows:
        print("препятствие не найдено ни в одном кадре — проверять нечего")
        return

    def med(rows, key):
        return float(np.median([r[key] for r in rows]))

    print(f"=== {a.bag}: {len(solo_rows)} кадров, окно накопления {a.n_frames} ===")
    print(f"измеренное Δs: медиана {np.median(shifts):.3f} м "
          f"(поезд на этой записи стоит, см. knowledge.md §6)")
    print(f"облако после слияния крупнее в {np.median(sizes):.1f} раза\n")
    print(f"{'величина':38s} {'один кадр':>12s} {'накопление':>12s}")
    print(f"{'расстояние до препятствия, м':38s} {med(solo_rows, 'dist'):12.3f} "
          f"{med(merged_rows, 'dist'):12.3f}")
    print(f"{'протяжённость скопления, м':38s} {med(solo_rows, 'extent'):12.3f} "
          f"{med(merged_rows, 'extent'):12.3f}")
    print(f"{'точек в коридоре':38s} {med(solo_rows, 'n'):12.0f} "
          f"{med(merged_rows, 'n'):12.0f}")

    dt = 0.1
    print(f"\nРазмазывание объекта, движущегося относительно тоннеля, — величина "
          f"аналитическая:\n  v_отн · (N−1) · Δt = v_отн · {(a.n_frames - 1) * dt:.1f} с")
    for v in (1.0, 5.0, 15.0):
        print(f"    при {v:4.1f} м/с ({v * 3.6:4.0f} км/ч) — {v * (a.n_frames - 1) * dt:4.1f} м")
    print("\nВывод по применимости: накопленное облако — для ГЕОМЕТРИИ тоннеля.\n"
          "Препятствие ищется в ТЕКУЩЕМ кадре, сравнением с габаритом, который\n"
          "снят по накопленному. Неподвижное относительно тоннеля препятствие\n"
          "накопление сохраняет; движущееся — размажет на величину выше.")


if __name__ == "__main__":
    main()
