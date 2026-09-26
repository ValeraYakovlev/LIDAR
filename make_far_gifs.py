#!/usr/bin/env python3
"""GIF эксперимента 18: габарит поезда, дальние находки, профиль пути по высоте.

Трекер пути (§31, без изменений) и детектор эксперимента 18 (`far_detect`)
обрабатывают КАЖДЫЙ кадр записи; в GIF попадает каждый k-й.

Панели:
  слева         — вид сверху: путь, стены, края коридора габарита поезда на
                  середине высоты (сужаются вдали на запас по погрешности пути);
                  красные кольца — подтверждённые находки, оранжевые × — точки в
                  габарите;
  справа вверху — срез перпендикулярно пути: габарит поезда (кузов 2.67 × 3.65 м,
                  снизу уже — контактный рельс), пунктир — он же, сжатый на запас;
  справа 2      — вид сбоку (полоса |u| <= 1 м над путём): пол и свод по полосам,
                  профиль пути по высоте (жёлтый), низ и верх габарита вдоль пути;
  справа внизу  — дальность находок по всему прогону; серым — где по разметке
                  стоят предметы (New_synth 1–5, синтетика — по /lidar_points_labeled).

На New_synth_data (`--holdout-cut`) всё, что по длине пути дальше S_B, в рисунок
не попадает: это отложенная часть (план 18).

    python make_far_gifs.py --dataset output/new_synth --bags cloud_with_fake_obj --holdout-cut
    python make_far_gifs.py --dataset /Volumes/T7/Dataset --bags roundT_doubleT
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from PIL import Image

from exp20_split import guard
from exp_far_truth import track_length
from make_gauge_gifs import X_LIM, _crop, _u8
from rail_detection import bag_path, frame_count, iter_frames
from rail_detection import far_detect as fd
from rail_detection.contrast_gauge import path_at
from rail_detection.parallel_path import ParallelGauge, offset_curve

warnings.filterwarnings("ignore")

FOLDER = "Opus 5.5/far_detection"
D_SHOW = 150.0
N_SLICE = 6000
DEFAULT_SLICE = 60.0


def gauge_poly(p, mg=0.0):
    """Контур габарита в (u, v) — с вырезом под контактный рельс и запасом mg."""
    h, top, b = p["half"] - mg, p["top"] - mg, p["bottom"] + mg
    if p.get("low_half") is None:
        return np.array([(-h, b), (h, b), (h, top), (-h, top)])
    lh, vs = p["low_half"] - mg, p["v_step"]
    return np.array([(-lh, b), (lh, b), (lh, vs), (h, vs), (h, top), (-h, top),
                     (-h, vs), (-lh, vs)])


def collect(dataset, bag, every, p, max_frames=None):
    pg = ParallelGauge()
    det = fd.FarDetector(p)
    rng = np.random.default_rng(0)
    recs, stats = [], []
    for idx, points, n_total in iter_frames(bag_path(dataset, bag), stride=1,
                                           max_frames=max_frames):
        res = pg.update(points, steps=1)
        fr = det.update(res)
        st = {"idx": idx, "ok": fr is not None}
        if fr is not None:
            st.update({"raw": [c["dist"] for c in fr["clusters"]], "conf": fr["confirmed"],
                       "ds": res["track"]["ds"], "limit": fr["limit"],
                       "clusters": fr["clusters"],
                       "vp": [fr["vinfo"]["a"], fr["vinfo"]["b"]]})
        stats.append(st)
        if idx % every == 0:
            recs.append(_render_data(idx, n_total, res, fr, p, rng))
        print(f"\r  {bag}: кадр {idx}/{n_total}", end="", flush=True)
    print()
    return recs, stats


def _render_data(idx, n_total, res, fr, p, rng):
    rec = {"idx": idx, "n_total": n_total, "ok": fr is not None}
    if fr is None:
        return rec
    tr = res["track"]
    c = tr["curve"]
    dd = np.linspace(2, D_SHOW, 300)
    x_p, psi, arc = path_at(res["path"], dd)
    pg_, _, uc, _ = res["pose_curves"]
    centre = x_p + np.interp(arc, pg_, uc) / np.cos(psi)
    half = p["half"] - fd.margin(arc, p) if p.get("m_slope") else np.full_like(arc, p["half"])
    half = np.maximum(half, 0.0)
    corridor = np.column_stack([centre - half / np.cos(psi), centre, centre + half / np.cos(psi), dd])
    conf = fr["confirmed"]
    cl = fr["clusters"]
    dist = conf[0] if conf else (cl[0]["dist"] if cl else min(DEFAULT_SLICE, 0.8 * fr["limit"]))
    ht = fd_half_thick(dist)
    s, ug, vg = res["s"], res["ug"], res["vg"]
    dv = fr["delta"](s) if fr["delta"] is not None else 0.0
    vv = vg - dv
    m = (np.abs(s - dist) < ht) & (np.abs(ug) < 5.0) & (vv > -1.2) & (vv < 5.5)
    sel = np.where(m)[0]
    if len(sel) > N_SLICE:
        sel = rng.choice(sel, N_SLICE, replace=False)
    gp = fr["gauge_pts"]
    x, y, _ = res["xyz"]
    # точки в габарите — на вид сверху по их (s, u) через путь
    pd = res["path"]["d"]
    xpd, psid, arcd = path_at(res["path"], pd)
    hits = np.zeros((0, 2), np.float32)
    if len(gp):
        j = np.clip(np.searchsorted(arcd, gp[:, 0]), 0, len(pd) - 1)
        hits = np.column_stack([xpd[j] + gp[:, 1] / np.cos(psid[j]), pd[j]]).astype(np.float32)
        if len(hits) > 3000:
            hits = hits[rng.choice(len(hits), 3000, replace=False)]
    marks = []
    for q in cl:
        j = int(np.argmin(np.abs(arcd - q["dist"])))
        marks.append((xpd[j] + q["u"] / np.cos(psid[j]), pd[j],
                      any(abs(q["dist"] - d) < 0.5 for d in conf)))
    walls = {}
    for side, w in (("left", c["wl"]), ("right", c["wr"])):
        xw, dw = offset_curve(c, w)
        lim = tr["reach_side"].get(side) or 0.0
        start = tr["wall_from"].get(side)
        start = lim if start is None else max(2.0, start)
        keep = (dw >= start) & (dw <= min(lim, D_SHOW))
        walls[side] = np.column_stack([xw[keep], dw[keep]]).astype(np.float32)
    ss = np.arange(0.0, D_SHOW + 1, 1.0)
    rec.update({
        "img": _u8(_crop(res["grid"], res["silhouette"]["image"])),
        "walls": walls, "corridor": corridor.astype(np.float32), "hits": hits,
        "marks": marks, "limit": fr["limit"], "reach": res["reach"],
        "slice": np.column_stack([ug[sel], vv[sel]]).astype(np.float32),
        "slice_dist": float(dist), "slice_ht": float(ht),
        "mg": float(fd.margin(np.array([dist]), p)[0]) if p.get("m_slope") else 0.0,
        "side": fr["side"].astype(np.float32), "levels": fr["vinfo"]["levels"],
        "h_ceil": fr["vinfo"]["h_ceil"],
        "prof": (ss, fr["delta"](ss) if fr["delta"] is not None else np.zeros_like(ss)),
        "clusters": cl, "confirmed": conf,
    })
    return rec


def fd_half_thick(s):
    return 0.8 if s <= 20 else (1.5 if s <= 40 else (2.5 if s <= 70 else 4.0))


def truth_tracks(tag, bag, stats):
    """Где стоят предметы по разметке: [(имя, кадры, дальности)] для ленты находок."""
    n = len(stats)
    if tag == "new_synth":
        ds = np.array([s.get("ds", np.nan) for s in stats], float)
        S = track_length({"ds": ds})
        info = json.load(open("results/exp18/new_synth_objects.json"))
        objs = list(info["objects"])
        hold = Path("results/exp18/new_synth_holdout_objects.json")
        if hold.exists():       # предметы 6–10 — только на полном GIF, ниже S_B их вырезает рисунок
            objs += json.load(open(hold))["objects"]
        out = []
        for o in objs:
            d = o["s_obj"] - S
            k = np.flatnonzero((d > 0) & (d < D_SHOW))
            out.append((f"{o['n']}" + ("" if o["positive"] else " (вне)"), k, d[k]))
        return out, S, info["S_B"]
    p = Path(f"output/synthetic_truth/{bag}.json")
    if p.exists():
        tr = json.load(open(p))
        k = np.array([r["idx"] for r in tr["frames"] if r["points"] > 0 and r["idx"] < n])
        d = np.array([r["depth_min"] for r in tr["frames"] if r["points"] > 0 and r["idx"] < n])
        return [("предмет", k, d)], None, None
    if bag == "doubleT_obstacle":
        k = np.arange(4, 76)
        kb = np.arange(50, n)          # коробка на левом рельсе (экспер. 18б)
        return [("человек", k, np.full(len(k), 55.5)),
                ("коробка", kb, np.full(len(kb), 56.4))], None, None
    return [], None, None


def render(recs, stats, bag, p, truth, S=None, S_B=None, dpi=80):
    xs = np.array([s["idx"] for s in stats], float)
    fig = Figure(figsize=(12.0, 9.2), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 2, width_ratios=[0.9, 1.35], height_ratios=[1.35, 1.0, 0.75],
                          left=0.06, right=0.98, top=0.9, bottom=0.05, wspace=0.2, hspace=0.5)
    axT = fig.add_subplot(gs[:, 0])
    axS = fig.add_subplot(gs[0, 1])
    axV = fig.add_subplot(gs[1, 1])
    axR = fig.add_subplot(gs[2, 1])
    cut = lambda k: (S_B - S[k]) if S is not None else np.inf     # дальность до S_B
    raw_pts, conf_pts = [], []
    for s in stats:
        if not s["ok"]:
            continue
        c = cut(s["idx"])
        raw_pts += [(s["idx"], d) for d in s["raw"] if d < c]
        conf_pts += [(s["idx"], d) for d in s["conf"] if d < c]
    raw_pts, conf_pts = np.array(raw_pts).reshape(-1, 2), np.array(conf_pts).reshape(-1, 2)
    images = []
    for i, r in enumerate(recs):
        for a in (axT, axS, axV, axR):
            a.clear()
        c = cut(r["idx"])
        head = f"{bag}  кадр {r['idx']}/{r['n_total']}"
        if not r["ok"]:
            head += "\nгеометрия не построена"
        else:
            vis_d = min(D_SHOW, c)
            axT.imshow(np.ma.masked_equal(r["img"], 0), origin="lower", aspect="auto",
                       cmap="gray_r", vmin=1, vmax=255, interpolation="nearest",
                       extent=[X_LIM[0], X_LIM[1], 2.0, 150.0])
            if vis_d < D_SHOW:
                axT.axhspan(max(vis_d, 0), D_SHOW, color="#e9ecef", zorder=3)
                axT.text(0, min(D_SHOW - 6, max(vis_d, 0) + 6), "отложенная часть\n(не смотрю)",
                         ha="center", fontsize=8, color="#6b7280", zorder=4)
            for side in ("left", "right"):
                w = r["walls"][side]
                w = w[w[:, 1] < vis_d] if len(w) else w
                if len(w):
                    axT.plot(w[:, 0], w[:, 1], c="#2fbf4a", lw=2.0, zorder=4)
            cr = r["corridor"]
            vis = (cr[:, 3] <= r["limit"]) & (cr[:, 3] < vis_d)
            axT.plot(cr[vis, 1], cr[vis, 3], c="#d6336c", lw=1.2, ls="--", zorder=5)
            axT.plot(cr[vis, 0], cr[vis, 3], c="#d1495b", lw=1.4, zorder=5)
            axT.plot(cr[vis, 2], cr[vis, 3], c="#d1495b", lw=1.4, zorder=5)
            h = r["hits"]
            h = h[h[:, 1] < vis_d] if len(h) else h
            if len(h):
                axT.scatter(h[:, 0], h[:, 1], s=6, c="#ff8c1a", marker="x", linewidths=0.7, zorder=6)
            for mx, md, is_c in r["marks"]:
                if md < vis_d:
                    axT.scatter([mx], [md], s=120 if is_c else 60, facecolors="none",
                                edgecolors="#c92a2a" if is_c else "#ff8c1a",
                                linewidths=2.0 if is_c else 1.2, zorder=9)
            axT.axhline(min(r["limit"], vis_d), c="crimson", lw=0.8, ls=":", alpha=0.7)
            axT.set_xlim(*X_LIM)
            axT.set_ylim(0, D_SHOW)
            axT.set_ylabel("глубина, м", fontsize=8)
            axT.set_xlabel("вбок, м", fontsize=8)
            axT.tick_params(labelsize=7)
            axT.set_title("вид сверху; зелёные — стены, красные — края габарита\n"
                          "поезда (сужаются вдали на запас), кольца — находки:\n"
                          "красные — подтверждённые, оранжевые — сырые", fontsize=7.5)

            # срез
            if r["slice_dist"] < c:
                sl = r["slice"]
                axS.scatter(sl[:, 0], sl[:, 1], s=2.0, c="#9aa5b1", alpha=0.6, linewidths=0)
                poly = gauge_poly(p)
                inside = matplotlib.path.Path(gauge_poly(p, r["mg"])).contains_points(sl)
                if inside.any():
                    axS.scatter(sl[inside, 0], sl[inside, 1], s=6, c="#d1495b", linewidths=0)
                axS.add_patch(Polygon(poly, closed=True, fill=False, edgecolor="#d1495b", lw=1.6))
                if r["mg"] > 0.005:
                    axS.add_patch(Polygon(gauge_poly(p, r["mg"]), closed=True, fill=False,
                                          edgecolor="#d1495b", lw=1.0, ls="--"))
                axS.set_title(f"плоскость ⊥ пути на {r['slice_dist']:.0f} м (±{r['slice_ht']:.1f} м); "
                              f"пунктир — габарит, сжатый на запас {r['mg']:.2f} м", fontsize=8)
            else:
                axS.set_title("срез в отложенной части — не показываю", fontsize=8)
            axS.axhline(0, c="#adb5bd", lw=0.6)
            axS.axvline(0, c="#adb5bd", lw=0.6)
            axS.set_xlim(-4, 4)
            axS.set_ylim(-1.2, 5.0)
            axS.set_aspect("equal", adjustable="box")
            axS.tick_params(labelsize=7)
            axS.set_xlabel("вбок от оси пути, м", fontsize=7.5)
            axS.set_ylabel("над головками, м", fontsize=7.5)

            # вид сбоку
            H = r["side"].copy()
            se, ve = fd.SIDE_S, fd.SIDE_V
            if c < D_SHOW:
                H[int(max(c, 0)):] = 0
            Hn = H / np.maximum(H.max(axis=1, keepdims=True), 1)
            axV.imshow(Hn.T ** 0.5, origin="lower", aspect="auto", cmap="magma",
                       extent=[se[0], se[-1], ve[0], ve[-1]])
            ss, dlt = r["prof"]
            keep = ss < min(c, r["limit"])
            axV.plot(ss[keep], dlt[keep], c="#ffd43b", lw=1.6)
            mgv = fd.margin(ss, p) if p.get("m_slope") else 0 * ss
            axV.plot(ss[keep], (dlt + p["top"] - mgv)[keep], c="#51cf66", lw=1.0)
            axV.plot(ss[keep], (dlt + p["bottom"] + mgv)[keep], c="#51cf66", lw=1.0)
            L = r["levels"]
            kl = L[:, 0] < c
            axV.scatter(L[kl, 0], L[kl, 1], s=8, c="#74c0fc", zorder=5)
            axV.scatter(L[kl, 0], L[kl, 2], s=8, c="#ffa94d", zorder=5)
            for q in r["clusters"]:
                if q["dist"] < c:
                    axV.scatter([q["dist"]], [q["v"]], s=50, facecolors="none",
                                edgecolors="#ff6b6b", linewidths=1.5, zorder=6)
            if c < D_SHOW:
                axV.axvspan(max(c, 0), D_SHOW, color="#e9ecef", zorder=4)
            axV.set_xlim(0, D_SHOW)
            axV.set_ylim(-1.5, 6.0)
            axV.tick_params(labelsize=7)
            axV.set_xlabel("вдоль пути, м", fontsize=7.5)
            axV.set_ylabel("над головками, м", fontsize=7.5)
            hc = f"{r['h_ceil']:.2f}" if r["h_ceil"] is not None else "—"
            axV.set_title("вид сбоку (|u| ≤ 1 м): голубые — уровень пола, оранжевые — свода "
                          f"(высота над головками вблизи {hc} м);\nжёлтый — профиль пути по высоте, "
                          "зелёные — низ и верх габарита", fontsize=7.5)

            conf, cl = r["confirmed"], r["clusters"]
            conf = [d for d in conf if d < c]
            if conf:
                head += f"\nПРЕПЯТСТВИЕ в габарите на {min(conf):.0f} м (подтверждено)"
                if len(conf) > 1:
                    head += f"; ещё {len(conf) - 1}"
            elif [q for q in cl if q["dist"] < c]:
                q = [q for q in cl if q["dist"] < c][0]
                head += f"\nскопление в габарите на {q['dist']:.0f} м ({q['n']} вокс.) — не подтверждено"
            else:
                head += "\nгабарит чист"
            head += f";  путь определён до {r['limit']:.0f} м"

        for name, k, d in truth:
            if len(k):
                kk = k[d < np.array([cut(j) for j in k])] if S is not None else k
                dd = d[d < np.array([cut(j) for j in k])] if S is not None else d
                axR.plot(kk, dd, c="#ced4da", lw=4, solid_capstyle="butt", zorder=1)
                if len(kk):
                    axR.text(kk[0], min(dd[0] + 4, D_SHOW - 8), name, fontsize=7, color="#495057")
        if len(raw_pts):
            axR.scatter(raw_pts[:, 0], raw_pts[:, 1], s=4, c="#ff8c1a", zorder=3)
        if len(conf_pts):
            axR.scatter(conf_pts[:, 0], conf_pts[:, 1], s=7, c="#d1495b", zorder=4)
        axR.axvline(r["idx"], c="crimson", lw=1.2)
        axR.set_xlim(xs.min(), max(xs.max(), xs.min() + 1))
        axR.set_ylim(0, D_SHOW)
        axR.tick_params(labelsize=7)
        axR.set_xlabel("кадр записи", fontsize=7.5)
        axR.set_title("дальность находок, м: оранжевое — сырые, красное — подтверждённые; "
                      "серое — где предмет по разметке", fontsize=8)
        fig.suptitle(head, fontsize=9.2)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
        images.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE, colors=128))
        print(f"\r  отрисовано {i + 1}/{len(recs)}", end="", flush=True)
    print()
    return images


def build(dataset, bag, out_dir, target, fps, max_frames, variant, holdout_cut):
    p = fd.VARIANTS[variant]
    n = frame_count(bag_path(dataset, bag))
    every = max(1, round(n / target))
    print(f"\n=== {bag}: все {n} кадров, в GIF каждый {every}-й; вариант {variant} ===")
    recs, stats = collect(dataset, bag, every, p, max_frames)
    tag = "new_synth" if holdout_cut else Path(dataset).name
    truth, S, S_B = truth_tracks(tag, bag, stats)
    if holdout_cut:
        truth = [t for t in truth]
    else:
        S = S_B = None
    images = render(recs, stats, bag, p, truth, S, S_B)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_dev" if holdout_cut else ""
    path = out_dir / f"{bag}{suffix}.gif"
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    if not holdout_cut:
        with open(out_dir / f"{bag}.json", "w") as f:
            json.dump({"bag": bag, "variant": variant, "params": p, "stats": stats}, f,
                      ensure_ascii=False, default=float)
    print(f"  {path} ({path.stat().st_size / 1e6:.1f} МБ)")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--dataset", default="/Volumes/T7/Dataset")
    a.add_argument("--bags", nargs="+", required=True)
    a.add_argument("--variant", default="final")
    a.add_argument("--out", default=f"output/{FOLDER}")
    a.add_argument("--target-frames", type=int, default=130)
    a.add_argument("--max-frames", type=int, default=None)
    a.add_argument("--fps", type=float, default=8.0)
    a.add_argument("--holdout-cut", action="store_true",
                   help="New_synth: вырезать всё дальше S_B (разработка)")
    a.add_argument("--holdout", action="store_true",
                   help="отложенный замер: разрешить отложенные записи эксперимента 20")
    args = a.parse_args()
    guard(args.bags, args.holdout)
    for bag in args.bags:
        build(args.dataset, bag, Path(args.out), args.target_frames, args.fps, args.max_frames,
              args.variant, args.holdout_cut)


if __name__ == "__main__":
    main()
