#!/usr/bin/env python3
"""Эксперимент 19: страховка ускорения — эталон, сверка, замер скорости.

Ускорение меняет только реализацию. Ответ после каждой правки обязан совпасть
с эталоном `speed-ref` кадр в кадр (план `experiments/19-speed.md`).

    python exp_speed.py golden --dataset /Volumes/T7/Dataset --bags doubleT_platform ...
    python exp_speed.py check  --dataset /Volumes/T7/Dataset --bags doubleT_platform ...
    python exp_speed.py bench  --label ref
    python exp_speed.py plot   --label ref,seq,par --names "до,шаги 1–4,параллельно"

golden — покадровые выходы всего конвейера (путь §31 + детектор §34, варианты
         final и final_b2) в output/exp19_golden/<набор>/<запись>.npz;
check  — тот же прогон текущим кодом и сравнение с эталоном: дискретное —
         точно, числа — до 1e-6; печатает первое расхождение;
bench  — время кадра по стадиям на кадрах, заранее прочитанных в память
         (чтение с диска не входит), в results/exp19/bench_<label>.json.

New_synth_data в разработке ускорения не используется: её эталон снят с
`speed-ref` и заморожен, сверка на ней — один раз в конце.
"""

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

GOLDEN = Path("output/exp19_golden")
VARIANTS = ("final", "final_b2")
TOL = 1e-6
FROZEN = "cloud_with_fake_obj"      # New_synth — только финальная сверка


# ---------------------------------------------------------------- прогон

def _clusters(cl):
    return [[c["dist"], c["n"], c["u"], c["v"], *c["size"]] for c in cl]


def run_bag(dataset, bag, max_frames=None):
    """Весь конвейер по записи; на кадр — словарь чисел и дискретных полей."""
    from rail_detection import bag_path, iter_frames
    from rail_detection import far_detect as fd
    from rail_detection.contrast_gauge import PATH_GRID
    from rail_detection.parallel_path import ParallelGauge, to_path_dict

    pg = ParallelGauge()
    dets = {v: fd.FarDetector(fd.VARIANTS[v]) for v in VARIANTS}
    rows = []
    for idx, points, _ in iter_frames(bag_path(dataset, bag), max_frames=max_frames):
        res = pg.update(points, steps=1)
        row = {"ok": res is not None}
        if res is not None:
            tr = res["track"]
            pgrid, th, uc, vc = res["pose_curves"]
            alt = np.full(len(PATH_GRID), np.nan)
            if tr.get("alt") is not None:
                alt = to_path_dict(tr["alt"]["curve"], PATH_GRID, "")["x"]
            row.update({
                "path_x": np.asarray(res["path"]["x"], float), "alt_x": np.asarray(alt, float),
                "theta": th, "uc": uc, "vc": vc, "floor": np.asarray(res["floor"], float),
                "num": np.array([tr["ds"], np.nan if res["shift"] is None else res["shift"],
                                 res["reach"], res["limit"], tr["cost"],
                                 tr["curve"]["w0"][0], tr["curve"]["w0"][1],
                                 float(res["inside"].sum()), float(len(res["s"]))]),
                "mode": res["path"]["mode"], "origin": tr["origin"],
                "ds_measured": bool(tr["ds_measured"]), "n_rails": tr["n_rails"],
                "steps": [[list(map(float, st)) for st in side] for side in tr["curve"]["steps"]],
                "base_cl": _clusters(res["clusters"]),
                "base_conf": res["confirmed"],
            })
        for v, det in dets.items():
            fr = det.update(res)
            if fr is None:
                row[v] = None
                continue
            gp = fr["gauge_pts"]
            row[v] = {"cl": _clusters(fr["clusters"]), "conf": list(fr["confirmed"]),
                      "num": [fr["limit"], fr["vinfo"]["a"], fr["vinfo"]["b"],
                              np.nan if fr["vinfo"]["h_ceil"] is None else fr["vinfo"]["h_ceil"],
                              float(len(gp)), float(gp.sum()) if len(gp) else 0.0]}
        rows.append(row)
    return rows


ARRAYS = ("path_x", "alt_x", "theta", "uc", "vc", "floor", "num")


def save(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    meta = []
    for k, r in enumerate(rows):
        m = {key: val for key, val in r.items() if key not in ARRAYS}
        meta.append(m)
        for key in ARRAYS:
            if key in r:
                arrays[f"{key}_{k}"] = r[key]
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, default=float)


def load(path):
    z = np.load(path.with_suffix(".npz"))
    meta = json.load(open(path.with_suffix(".json")))
    for k, m in enumerate(meta):
        for key in ARRAYS:
            if f"{key}_{k}" in z.files:
                m[key] = z[f"{key}_{k}"]
    return meta


# ---------------------------------------------------------------- сверка

def _close(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return False
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(np.all(both_nan | (np.abs(a - b) <= TOL)))


def compare(ref, new):
    """Первое расхождение: (кадр, поле, эталон, сейчас) или None."""
    if len(ref) != len(new):
        return (-1, "число кадров", len(ref), len(new))
    for k, (a, b) in enumerate(zip(ref, new)):
        for key in sorted(set(a) | set(b)):
            x, y = a.get(key), b.get(key)
            if key in ARRAYS:
                if (x is None) != (y is None) or (x is not None and not _close(x, y)):
                    return (k, key, x, y)
            elif key in VARIANTS:
                if (x is None) != (y is None):
                    return (k, key, x, y)
                if x is None:
                    continue
                if x["conf"] and not _close(x["conf"], y["conf"]) or len(x["conf"]) != len(y["conf"]):
                    return (k, key + ".conf", x["conf"], y["conf"])
                if len(x["cl"]) != len(y["cl"]) or (x["cl"] and not _close(x["cl"], y["cl"])):
                    return (k, key + ".cl", x["cl"], y["cl"])
                if not _close(x["num"], y["num"]):
                    return (k, key + ".num", x["num"], y["num"])
            elif key in ("base_cl",):
                if len(x) != len(y) or (x and not _close(x, y)):
                    return (k, key, x, y)
            elif key == "base_conf":
                if (x is None) != (y is None) or (x is not None and abs(x - y) > TOL):
                    return (k, key, x, y)
            elif key == "steps":
                fx = [v for side in x for st in side for v in st]
                fy = [v for side in y for st in side for v in st]
                if [len(s) for s in x] != [len(s) for s in y] or not _close(fx, fy):
                    return (k, key, x, y)
            elif x != y:
                return (k, key, x, y)
    return None


def _job(args):
    global GOLDEN, VARIANTS
    mode, dataset, bag, max_frames, workers, process, golden, variants = args
    GOLDEN, VARIANTS = Path(golden), tuple(variants)
    from rail_detection import parallel as par
    par.set_workers(workers)
    par.set_process(process)
    tag = Path(dataset).name
    t0 = time.time()
    rows = run_bag(dataset, bag, max_frames)
    dt = time.time() - t0
    path = GOLDEN / tag / bag
    if mode == "golden":
        save(rows, path)
        return f"{tag}/{bag}: эталон, {len(rows)} кадров, {dt:.0f} с"
    ref = load(path)
    if max_frames is not None:
        ref = ref[:len(rows)]
    # сравнение через ту же запись/чтение, что и эталон: одинаковые типы
    tmp = Path("/tmp") / f"{GOLDEN.name}_check_{tag}_{bag}_w{workers}{'p' if process else ''}"
    save(rows, tmp)
    diff = compare(ref, load(tmp))
    if diff is None:
        return f"{tag}/{bag}: СОВПАЛО, {len(rows)} кадров, {dt:.0f} с"
    k, key, a, b = diff
    return f"{tag}/{bag}: РАСХОЖДЕНИЕ на кадре {k}, поле {key}\n    эталон: {a}\n    сейчас: {b}"


# ---------------------------------------------------------------- замер

STAGES = [
    ("views", "rasterize", "растр вида сверху"),
    ("views", "run_silhouette", "стены вида сверху"),
    ("tunnel_frame", "rail_samples", "рельсы"),
    ("contrast_gauge", "build_path", "путь без памяти"),
    ("views", "floor_level", "пол"),
    ("contrast_gauge", "to_path_coords", "координаты пути"),
    ("shift", "estimate_shift", "Δs"),
    ("roll", "rail_pose_track", "крен и головки"),
]


def _instrument(log):
    """Обёртки со временем на функции стадий — только внутри замера, код не меняется.
    Учитывается только верхний уровень: вложенный вызов стадии в стадию не
    считается дважды."""
    import importlib

    from rail_detection import far_detect, parallel_path

    import threading

    depth = [0]
    main = threading.main_thread()

    def wrap(fn, name):
        def inner(*a, **kw):
            # только основной поток и только верхний уровень: при распараллеливании
            # куски и задачи пула в учёт стадий не идут (их время — в ожидании)
            if depth[0] or threading.current_thread() is not main:
                return fn(*a, **kw)
            depth[0] += 1
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                log[name] = log.get(name, 0.0) + time.perf_counter() - t
                depth[0] -= 1
        return inner

    for mod, fn, name in STAGES:
        m = importlib.import_module(f"rail_detection.{mod}")
        setattr(m, fn, wrap(getattr(m, fn), name))
    from rail_detection import parallel
    parallel.chunked = wrap(parallel.chunked, "по точкам кусками")
    parallel.submit = wrap(parallel.submit, "постановка в пул")
    cls = parallel_path.WallParallelTracker
    cls.step = wrap(cls.step, "трекер пути")
    far_detect.FarDetector.update = wrap(far_detect.FarDetector.update, "детектор 18")


def bench(label, bags, n_frames, workers=1, process=False):
    from rail_detection import bag_path, iter_frames
    from rail_detection import parallel as par
    par.set_workers(workers)
    par.set_process(process)
    from rail_detection import far_detect as fd
    from rail_detection.parallel_path import ParallelGauge

    log = {}
    _instrument(log)
    import platform
    import subprocess
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                             text=True).stdout.strip()
    except OSError:
        cpu = ""
    if not cpu and Path("/proc/cpuinfo").exists():          # Linux (ВМ, Docker)
        cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo")
                    if ln.startswith("model name")), "")
    cpu = cpu or platform.processor()
    out = {"label": label, "cpu": cpu, "workers": workers, "process": process, "bags": {}}
    for dataset, bag in bags:
        frames = [p.copy() for _, p, _ in iter_frames(bag_path(dataset, bag), max_frames=n_frames)]
        pg = ParallelGauge()
        det = fd.FarDetector(fd.VARIANTS["final"])
        per = []
        for k, p in enumerate(frames):
            log.clear()
            t = time.perf_counter()
            det.update(pg.update(p, steps=1))
            total = time.perf_counter() - t
            if k >= 5:          # первые кадры — прогрев
                per.append({"total": total, **dict(log)})
        tot = np.array([r["total"] for r in per]) * 1000
        out["bags"][bag] = per
        print(f"  {bag}: {len(per)} кадров, медиана {np.median(tot):.0f} мс, "
              f"p95 {np.percentile(tot, 95):.0f}, макс {tot.max():.0f}")
        names = [n for _, _, n in STAGES] + ["трекер пути", "детектор 18",
                                             "по точкам кусками"]
        for n in names:
            v = np.array([r.get(n, 0.0) for r in per]) * 1000
            print(f"      {n:22s} {np.median(v):6.1f} мс")
        rest = tot - np.array([sum(r.get(n, 0.0) for n in names) for r in per]) * 1000
        print(f"      {'прочее':22s} {np.median(rest):6.1f} мс")
        del frames
    Path("results/exp19").mkdir(parents=True, exist_ok=True)
    with open(f"results/exp19/bench_{label}.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)


BENCH_BAGS = [("/Volumes/T7/Dataset", "doubleT_obstacle"),
              ("/Volumes/T7/Dataset", "squareT_platform_squareT_switch"),
              ("/Volumes/T7/Synthetic_data", "human_smashed")]

DEV = ([("/Volumes/T7/Dataset", b) for b in
        ("doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT",
         "roundT_squareT_pressureGate_squareT", "squareT_platform_squareT_switch",
         "doubleT_obstacle")]
       + [("/Volumes/T7/reversed", b) for b in
          ("doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT",
           "roundT_squareT_pressureGate_squareT", "squareT_platform_squareT_switch",
           "doubleT_obstacle")]
       + [("/Volumes/T7/Synthetic_data", b) for b in
          ("box", "human_smashed", "human_smashed_diff_tunnels")])

# Эксперимент 22: итоговая сверка — записи, не участвующие в разработке ускорения
HOLDOUT22 = ([("output/last_synth", b) for b in
              ("conv_r300_a30", "conv_r300_a45", "conv_r450_a15", "conv_r450_a30",
               "conv_r450_a45", "conv_r600_a15", "conv_r600_a30", "conv_r600_a45")]
             + [("output/new_synth", FROZEN)])
# Эксперимент 23: итоговая сверка — зеркала Last_synth (в сверках скорости не были)
HOLDOUT23 = [("/Volumes/T7/reversed/Last_synth_data", b) for _, b in HOLDOUT22[:8]]


def plot(labels, names, out):
    """Boxplot времени кадра: по записи — ящик на каждый замер."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [json.load(open(f"results/exp19/bench_{lb}.json")) for lb in labels]
    bags = list(runs[0]["bags"])
    titles = {"doubleT_obstacle": "doubleT_obstacle\n(921 тыс. точек в кадре)",
              "squareT_platform_squareT_switch": "squareT_platform_squareT_switch\n(307 тыс.)",
              "human_smashed": "human_smashed, синтетика\n(307 тыс.)"}
    colors = ["#adb5bd", "#74c0fc", "#40c057", "#f59f00"]
    fig, axs = plt.subplots(1, len(bags), figsize=(4.2 * len(bags), 5.2), sharey=True)
    for ax, bag in zip(np.atleast_1d(axs), bags):
        data = [np.array([r["total"] for r in run["bags"][bag]]) * 1000 for run in runs]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                        medianprops={"color": "black", "lw": 1.6},
                        flierprops={"marker": ".", "ms": 3, "alpha": 0.5})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        for i, d in enumerate(data, 1):
            ax.text(i + 0.34, np.median(d), f"{np.median(d):.0f} мс", va="center", fontsize=9,
                    fontweight="bold")
        ax.axhline(100, color="#e03131", ls="--", lw=1.2)
        ax.set_ylim(0, None)
        ax.set_xlim(0.5, len(data) + 0.8)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, fontsize=8.5)
        ax.set_title(titles.get(bag, bag), fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(axs)[0].set_ylabel("время обработки кадра, мс")
    np.atleast_1d(axs)[-1].text(0.55, 104, "бюджет 100 мс (лидар 10 Гц)", color="#e03131",
                                fontsize=8.5)
    cpu = runs[0].get("cpu", "")
    fig.suptitle(f"Эксперимент 19: время обработки кадра до и после ускорения — ответ тот же "
                 f"(сверка с эталоном на 15 записях и на замороженной New_synth)\n"
                 f"{cpu}, кадры в памяти, по 100 кадров на запись",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(out)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("mode", choices=["golden", "check", "bench", "plot"])
    a.add_argument("--dataset", default=None)
    a.add_argument("--bags", nargs="*", default=None)
    a.add_argument("--all", action="store_true", help="все записи разработки ускорения (15)")
    a.add_argument("--frozen", action="store_true", help="New_synth — финальная сверка")
    a.add_argument("--holdout22", action="store_true",
                   help="экспер. 22: Last_synth + New_synth — итоговая сверка, один раз")
    a.add_argument("--holdout23", action="store_true",
                   help="экспер. 23: зеркала Last_synth — итоговая сверка, один раз")
    a.add_argument("--golden", default=str(GOLDEN), help="папка эталона")
    a.add_argument("--variants", nargs="*", default=list(VARIANTS),
                   help="варианты детектора в эталоне")
    a.add_argument("--max-frames", type=int, default=None)
    a.add_argument("--jobs", type=int, default=5)
    a.add_argument("--label", default="ref")
    a.add_argument("--bench-frames", type=int, default=105)
    a.add_argument("--bench-bags", nargs="*", default=None,
                   help="bench: записи как «папка:запись» (по умолчанию — три записи §35 на T7)")
    a.add_argument("--workers", type=int, default=1, help="потоков на кадр (1 — последовательно)")
    a.add_argument("--process", action="store_true",
                   help="рельсы и «память» трекера — в отдельном процессе (при --workers > 1)")
    a.add_argument("--names", default=None, help="plot: подписи замеров через запятую")
    a.add_argument("--out", default="results/exp19/speed_boxplot.png")
    args = a.parse_args()
    if args.mode == "bench":
        bags = [tuple(x.rsplit(":", 1)) for x in args.bench_bags] if args.bench_bags else BENCH_BAGS
        bench(args.label, bags, args.bench_frames, args.workers, args.process)
        return
    if args.mode == "plot":
        labels = args.label.split(",")
        plot(labels, args.names.split(",") if args.names else labels, args.out)
        return
    if args.all:
        jobs = DEV
    elif args.holdout22:
        jobs = HOLDOUT22
    elif args.holdout23:
        jobs = HOLDOUT23
    elif args.frozen:
        jobs = [("output/new_synth", FROZEN)]
    else:
        jobs = [(args.dataset, b) for b in args.bags]
    if any(b == FROZEN for _, b in jobs) and not (args.frozen or args.holdout22):
        raise SystemExit("New_synth заморожена: сверка на ней — только с --frozen, в конце")
    with ProcessPoolExecutor(args.jobs) as ex:
        for line in ex.map(_job, [(args.mode, d, b, args.max_frames, args.workers, args.process,
                                   args.golden, args.variants) for d, b in jobs]):
            print(line, flush=True)


if __name__ == "__main__":
    main()
