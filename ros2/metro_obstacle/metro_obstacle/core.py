"""Конвейер кадра без ROS: облако → путь и габарит → препятствие и расстояние.

То же, что делают скрипты замеров (`make_far_gifs.collect`): на каждый кадр
`ParallelGauge.update` (путь, стены, крен, Δs, §31/§36), затем
`FarDetector.update` (габарит кузова, скопления, подтверждение, §34). Здесь
же — перевод найденного обратно в координаты лидара, чтобы показать в RViz2.

Координаты лидара в записях: x — вбок, y — назад (глубина вдоль тоннеля — это
−y), z — вверх. Расстояние до препятствия — длина дуги вдоль пути от лидара
до ближнего края скопления.
"""

import time

import numpy as np

from rail_detection import far_detect as fd
from rail_detection.contrast_gauge import path_at
from rail_detection.loader import to_points
from rail_detection.parallel_path import ParallelGauge

DEFAULT_VARIANT = "low_rest_b0"
CORRIDOR_STEP = 2.0      # м: шаг по глубине, с которым рисуется коридор габарита


def _warm_imports():
    """Импорт всего, что конвейер тянет лениво, и компиляция numba (экспер. 23)
    — в процессе-помощнике тоже: иначе первый кадр ждёт компиляции."""
    import sklearn.cluster  # noqa: F401
    from rail_detection import (contrast_gauge, detector, roll, shift,  # noqa: F401
                                tunnel_frame, views)
    from rail_detection import parallel_path as pp

    s = np.linspace(1.0, 100.0, 40)
    pp._search_side(s, np.full(40, 1.8), np.ones(40), 1.8, pp.TUKEY * pp.SIGMA_EDGE)
    return True


class Pipeline:
    """Состояние конвейера на одну запись (путь, трекер, подтверждение)."""

    def __init__(self, variant=DEFAULT_VARIANT):
        if variant not in fd.VARIANTS:
            raise ValueError(f"неизвестный вариант детектора: {variant}")
        self.variant = variant
        self.p = fd.VARIANTS[variant]
        self.reset()

    @staticmethod
    def warm_up():
        """Поднять процессы-помощники (RAIL_PROCESS=1) и импорты до первого кадра:
        иначе первый кадр идёт ~2 с и узел пропускает всё, что пришло за это время."""
        from rail_detection import parallel as par

        _warm_imports()
        par.start_helpers()
        if par.use_process():
            for job in [par.submit_to(i, _warm_imports) for i in range(par.N_HELPERS)]:
                job.result()

    def reset(self):
        self.pg = ParallelGauge()
        self.det = fd.FarDetector(self.p)

    def process_msg(self, msg, steps=1):
        """Сообщение PointCloud2 (rclpy или rosbags) → результат кадра."""
        return self.process(to_points(msg), steps)

    def process(self, points, steps=1):
        """Точки кадра → словарь результата.

        steps — сколько кадров лидара прошло с прошлого обработанного (при
        пропуске кадров Δs ищется в более широком окне, как в `ParallelGauge`).
        """
        t0 = time.perf_counter()
        res = self.pg.update(points, steps=steps)
        fr = self.det.update(res)
        out = {"ok": fr is not None, "detected": False, "distance": None,
               "confirmed": [], "clusters": [], "limit": None,
               "proc_ms": 0.0, "n_points": int(len(points))}
        if fr is not None:
            conf = sorted(float(d) for d in fr["confirmed"])
            out.update(detected=bool(conf), distance=conf[0] if conf else None,
                       confirmed=conf, limit=float(fr["limit"]),
                       clusters=[{k: c[k] for k in ("dist", "n", "u", "v", "size")}
                                 for c in fr["clusters"]])
            out["geometry"] = self._geometry(res, fr, conf)
        out["proc_ms"] = (time.perf_counter() - t0) * 1000.0
        return out

    # ------------------------------------------------------------ для показа

    def _to_sensor(self, res, s, u, v):
        """(вдоль пути, поперёк, над головками) → (x, y, z) лидара.

        Обратный переход приближённый (без поворота на крен) — только для
        рисования; решение принимается в координатах габарита.
        """
        path = res["path"]
        pgrid, _, uc, vc = res["pose_curves"]
        d = np.interp(s, path["arc"], path["d"])
        x_p, psi, arc = path_at(path, d)
        ucs, vcs = np.interp(arc, pgrid, uc), np.interp(arc, pgrid, vc)
        x = x_p + (ucs + u) / np.cos(psi)
        z = np.polyval(res["floor"], d) + vcs + v
        return np.column_stack([x, -d, z]).astype(np.float32)

    def _geometry(self, res, fr, conf):
        p = self.p
        lim = float(fr["limit"])
        ss = np.arange(2.0, max(lim, 2.0) + 1e-6, CORRIDOR_STEP)
        half = np.full_like(ss, p["half"])
        if p.get("m_slope"):
            half = np.maximum(p["half"] - fd.margin(ss, p), 0.0)
        z0 = np.zeros_like(ss)
        corridor = {
            "left_bottom": self._to_sensor(res, ss, -half, z0 + p["bottom"]),
            "right_bottom": self._to_sensor(res, ss, half, z0 + p["bottom"]),
            "left_top": self._to_sensor(res, ss, -half, z0 + p["top"]),
            "right_top": self._to_sensor(res, ss, half, z0 + p["top"]),
        }
        boxes = []
        for D in conf:
            cl = min(fr["clusters"], key=lambda c: abs(c["dist"] - D), default=None)
            if cl is None:
                continue
            size = np.maximum(np.array(cl["size"], float), 0.2)
            centre = self._to_sensor(res, np.array([D + size[0] / 2]), np.array([cl["u"]]),
                                     np.array([cl["v"]]))[0]
            boxes.append({"dist": D, "centre": centre, "size": size})
        gp = fr["gauge_pts"]
        hits = (self._to_sensor(res, gp[:, 0], gp[:, 1], gp[:, 2]) if len(gp)
                else np.zeros((0, 3), np.float32))
        return {"corridor": corridor, "boxes": boxes, "gauge_hits": hits}
