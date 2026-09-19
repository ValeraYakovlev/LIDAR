"""Связывание геометрии между соседними кадрами.

Всё, что было до сих пор, работало покадрово: каждый кадр обрабатывался с нуля,
как будто предыдущего не существовало. Это честно, но выбрасывает сильное
физическое знание: между кадрами поезд проезжает единицы метров, а тоннель на
таком отрезке не перестраивается. Одиночный кадр, где подгонка ошиблась, видно
невооружённым глазом на GIF — кривизна дёргается там, где путь заведомо прямой.

Геометрия предыдущих кадров используется двояко:

1. Как ПОДСКАЗКА при поиске: прошлые полуширины передаются в подгонку как
   стартовое приближение вместо медианы ближних срезов. Там, где ближняя зона
   неоднозначна (платформа, ниша), это сразу уводит согласование к правильному
   скоплению, а не к случайно более населённому. Подсказка проверяется точками
   текущего кадра и молча уступает, если её скопления больше нет.

2. Как СГЛАЖИВАНИЕ кривизны: выдаётся не сырая α кадра, а медиана по короткому
   окну последних кадров.

Почему медиана, а не ограничение скорости изменения. Сначала здесь стоял именно
ограничитель: "за кадр α не может измениться больше чем на столько-то". На
разработочных прогонах он выглядел хорошо — дрожание падало втрое при неизменной
метрике стен. Но на отложенном станционном прогоне он ухудшил и метрику стен, и
точность оси вдвое, и причина оказалась принципиальной: ограничитель не гасит
ошибку, а РАСТЯГИВАЕТ её. Один неверный кадр сдвигает состояние, и следующие
кадры, даже правильные, не могут вернуться быстрее разрешённого шага —
одиночный выброс превращается в плато.

Медиана по окну такого дефекта лишена: одиночный выброс просто не попадает в
середину выборки и исчезает целиком, а состояние ни на шаг не смещается.
Настоящее изменение отслеживается с задержкой примерно в половину окна — при
окне 5 и шаге 2 это 2 кадра записи, то есть доли секунды.

Сглаживается кривизна (α), а не курс (β) сознательно: α — увод вбок на
фиксированной глубине относительно ПРЯМОЙ опоры, и от выбора этой опоры он
почти не зависит, поэтому сравним между кадрами. А β — наклон относительно
опоры, своей у каждого кадра: при рыскании поезда он законно скачет, и
сглаживать его значило бы бороться с реальным движением.
"""

from collections import deque

import numpy as np

from .accumulate import N_FRAMES, FrameAccumulator
from .tunnel_frame import DEPTH_SCALE, MIN_RADIUS, fit_tunnel_geometry, tracked_depth, wall_x

HISTORY = 5         # кадров в окне сглаживания кривизны
SPIKE_MARK = 0.15   # м: с такой разницы сырой и сглаженной α кадр считается выбросом
MAX_GAP_FRAMES = 6  # после стольких кадров без геометрии состояние сбрасывается


class TunnelTracker:
    """Последовательный проход по кадрам одной записи.

    Состояние сбрасывается методом reset() при переходе к другой записи —
    геометрия соседних записей никак не связана.

    accumulate: сколько кадров складывать в одно облако (см. accumulate.py).
        0 или 1 — как было, покадрово. Накопление живёт здесь, а не у
        вызывающего кода, потому что требует двух проходов по кадру — сначала
        опора по нему одному, потом подгонка по слитому облаку, — и держать эту
        последовательность в одном месте надёжнее, чем повторять её в каждом
        скрипте.
    """

    def __init__(self, history=HISTORY, max_gap=MAX_GAP_FRAMES, accumulate=N_FRAMES):
        self.history = history
        self.max_gap = max_gap
        self.accumulate = int(accumulate or 0)
        self.reset()

    def reset(self):
        self.alpha_hist = deque(maxlen=self.history)
        self.width_hist = {"left": deque(maxlen=self.history),
                           "right": deque(maxlen=self.history)}
        self.bounds = None
        self.gap = 0
        self.accumulator = (FrameAccumulator(n_frames=self.accumulate)
                            if self.accumulate > 1 else None)
        self.merged = None      # слитое облако последнего кадра (для отрисовки)
        self.n_native = 0       # сколько точек в нём от самого кадра
        self.edges = None       # структура разрывов ширины прошлого кадра

    @property
    def state(self):
        """Подсказка для следующего кадра: медианные полуширины по окну,
        границы тоннеля в координатах сенсора для отсечения точек и структура
        разрывов ширины, СДВИНУТАЯ на пройденный путь.

        Сдвиг обязателен и он же делает подсказку осмысленной: разрыв ширины —
        это место в тоннеле (торец платформы, начало раскрытия), и при движении
        вперёд оно приближается ровно на Δs. Передавать его без сдвига значило бы
        утверждать, что тоннель едет вместе с поездом.

        Без измеренного Δs структура не передаётся вовсе: лучше пересчитать её
        заново, чем приложить к текущему кадру устаревшие на полтора метра
        границы.
        """
        if not self.alpha_hist:
            return None
        out = {"widths": {side: (float(np.median(h)) if h else None)
                          for side, h in self.width_hist.items()},
               "bounds": self.bounds}
        shift = (self.accumulator.last.get("shift")
                 if self.accumulator is not None and self.accumulator.last.get("ok")
                 else None)
        if self.edges is not None and shift is not None and np.isfinite(shift):
            out["edges"] = {side: (e - shift) for side, e in self.edges.items()
                            if e is not None and len(e)}
        return out

    def update(self, points, steps=1, **kwargs):
        """Обрабатывает очередной кадр записи. Возвращает тот же dict, что и
        fit_tunnel_geometry, но с кривизной, сглаженной по окну, и (при
        включённом накоплении) с подгонкой по нескольким кадрам сразу.

        steps — сколько кадров записи прошло с прошлого вызова; используется
        чтобы понять, не оборвалась ли последовательность, и насколько широко
        искать продольное смещение.
        """
        self.merged, self.n_native = points, len(points)
        if self.accumulator is not None:
            # Опора считается по ОДНОМУ кадру: координаты пути, в которых
            # складываются кадры, должны принадлежать текущему, а рельсы и пол
            # накопление не трогает вовсе. Она же передаётся во вторую подгонку
            # готовой — детектор рельсов стоит больше половины её времени.
            solo = fit_tunnel_geometry(points, prior=self.state, **kwargs)
            if solo is not None:
                self.merged = self.accumulator.push(points, solo["frame"], steps=steps)
                kwargs = {**kwargs, "frame": solo["frame"]}
        points = self.merged

        res = fit_tunnel_geometry(points, prior=self.state, **kwargs)
        if res is None:
            self.gap += steps
            if self.gap > self.max_gap:
                self.reset()
            return None
        self.gap = 0

        self.alpha_hist.append(res["shape"]["alpha"])
        for side in ("left", "right"):
            if res[side] is not None:
                self.width_hist[side].append(res[side]["widths"][0])
        self.edges = {side: (np.asarray(res[side]["edges"], dtype=float)
                             if res[side] is not None else None)
                      for side in ("left", "right")}

        _apply_alpha(res, float(np.median(self.alpha_hist)))
        self.bounds = _wall_bounds(res)
        return res


def _wall_bounds(res, lo=3.0, hi=45.0):
    """Кривые стен в координатах сенсора — их следующий кадр использует как
    границу "тоннель кончается здесь". Параболы достаточно: сама модель стены
    не сложнее.

    max_depth ограничивает применимость: глубже того, куда стена этого кадра
    дотянулась, утверждать что-либо о границе нельзя.
    """
    dd = np.linspace(lo, hi, 40)
    out = {"left": None, "right": None, "max_depth": lo}
    reaches = []
    for side in ("left", "right"):
        if res[side] is None:
            return None
        out[side] = np.polyfit(dd, wall_x(res, side, dd), 2)
        r = tracked_depth(res, side)
        if r:
            reaches.append(r)
    out["max_depth"] = max(reaches) if reaches else hi
    return out


def _apply_alpha(res, alpha):
    """Подставляет сглаженную кривизну и пересчитывает то, что от неё зависит."""
    shape = res["shape"]
    if abs(alpha - shape["alpha"]) > SPIKE_MARK:
        shape["smoothed"] = True
    shape["alpha"] = alpha
    if abs(alpha) > 1e-9:
        radius = DEPTH_SCALE ** 2 / (2 * abs(alpha))
        shape["kind"], shape["radius"] = (("arc", float(radius)) if radius >= MIN_RADIUS
                                          else ("straight", None))
    else:
        shape["kind"], shape["radius"] = "straight", None
