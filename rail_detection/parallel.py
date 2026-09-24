"""Распараллеливание кадра (эксперимент 19) — без изменения ответа.

Две вещи:

* независимые стадии кадра идут одновременно: рельсы (`rail_samples`) —
  пока строится вид сверху; гипотеза «память» трекера пути — пока считается
  «заново». Каждая стадия считает ровно то же, что и в одиночку;
* поэлементные операции по всем точкам кадра (интерполяция по пути, синусы,
  поворот на крен) — кусками в нескольких потоках. Результат для каждой
  точки от нарезки не зависит, куски склеиваются в исходном порядке.

numpy отпускает GIL в тяжёлых операциях над массивами, поэтому хватает потоков.

Число потоков — `set_workers(n)` или переменная окружения RAIL_WORKERS;
1 (по умолчанию) — всё последовательно, как до распараллеливания.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_workers = max(1, int(os.environ.get("RAIL_WORKERS", "1")))
_pool = None
_lock = threading.Lock()
_inside = threading.local()      # задача пула: вложенное распараллеливание — последовательно

MIN_CHUNK = 40000                # точек на кусок: мельче — накладные расходы дороже выигрыша


def set_workers(n):
    """Число потоков для кадра (1 — последовательно)."""
    global _workers, _pool
    with _lock:
        n = max(1, int(n))
        if _pool is not None and n != _workers:
            _pool.shutdown(wait=True)
            _pool = None
        _workers = n


def workers():
    return _workers


def _get_pool():
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=_workers,
                                       thread_name_prefix="rail",
                                       initializer=lambda: setattr(_inside, "on", True))
        return _pool


class _Done:
    """Результат, посчитанный сразу (последовательный режим), с интерфейсом Future."""

    def __init__(self, value=None, error=None):
        self._v, self._e = value, error

    def result(self):
        if self._e is not None:
            raise self._e
        return self._v


def submit(fn, *args, **kwargs):
    """fn(*args) — в пуле, если он есть; иначе сразу. Возвращает объект с .result()."""
    if _workers <= 1 or getattr(_inside, "on", False):
        try:
            return _Done(fn(*args, **kwargs))
        except Exception as e:          # ошибка — там же, где её ждёт .result()
            return _Done(error=e)
    return _get_pool().submit(fn, *args, **kwargs)


def chunked(fn, arrays, *args):
    """fn(*куски arrays, *args) по кускам точек, склейка в исходном порядке.

    fn обязана быть поэлементной: значение для точки зависит только от этой
    точки (и от args), — тогда ответ тот же, что у fn(*arrays, *args).
    fn возвращает массив или кортеж массивов той же длины, что и вход.
    """
    n = len(arrays[0])
    k = min(_workers, n // MIN_CHUNK)
    if k <= 1 or getattr(_inside, "on", False):
        return fn(*arrays, *args)
    bounds = np.linspace(0, n, k + 1).astype(int)
    pool = _get_pool()
    futs = [pool.submit(fn, *[a[lo:hi] for a in arrays], *args)
            for lo, hi in zip(bounds[:-1], bounds[1:])]
    parts = [f.result() for f in futs]
    if isinstance(parts[0], tuple):
        return tuple(np.concatenate([p[j] for p in parts]) for j in range(len(parts[0])))
    return np.concatenate(parts)


# ---------------------------------------------------------------- отдельные процессы

# Рельсы, «память» трекера, подгонки стен разной степени, перебор ступенек
# одной стены — логика на Python: в соседнем потоке она делит GIL с основной и
# замедляет её (замер: стены вида сверху 32 → 37 мс при 4 потоках). В отдельных
# процессах она идёт по-настоящему параллельно. Процессы постоянные, их
# немного (N_HELPERS); каждая задача идёт в заранее назначенный процесс, задачи
# одного процесса выполняются по очереди, основной поток забирает результат
# каждой. Точки кадра для рельсов передаются через общую память.

import atexit
import importlib
import multiprocessing as _mp
from multiprocessing import shared_memory as _shm

N_HELPERS = 3
_use_process = os.environ.get("RAIL_PROCESS", "0") == "1"
_helpers = {}


def set_process(on):
    """Логика на Python — в отдельных процессах (при workers > 1)."""
    global _use_process
    _use_process = bool(on)


def use_process():
    return _workers > 1 and _use_process


def _helper_main(conn):
    import warnings

    warnings.filterwarnings("ignore")
    segments = {}
    while True:
        msg = conn.recv()
        if msg is None:
            break
        try:
            if msg[0] == "rails":
                from rail_detection.tunnel_frame import rail_samples
                _, name, shape, descr = msg
                if name not in segments:
                    segments[name] = _shm.SharedMemory(name=name)
                pts = np.ndarray(shape, dtype=np.dtype(descr), buffer=segments[name].buf)
                d, xc, _, g = rail_samples(pts)
                conn.send(("ok", (d, xc, None, g)))
            else:
                _, mod, fn, args = msg
                conn.send(("ok", getattr(importlib.import_module(mod), fn)(*args)))
        except Exception as e:              # ошибка — туда, где ждут результат
            conn.send(("err", e))
    for sg in segments.values():
        sg.close()


class _Pending:
    def __init__(self, conn):
        self._conn, self._got = conn, None

    def result(self):
        if self._got is None:
            self._got = self._conn.recv()
        kind, val = self._got
        if kind == "err":
            raise val
        return val


class _Helper:
    def __init__(self):
        ctx = _mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_helper_main, args=(child,), daemon=True)
        self.proc.start()
        self.seg = None

    def rails(self, points):
        if self.seg is None or self.seg.size < points.nbytes:
            if self.seg is not None:
                self.seg.close()
                self.seg.unlink()
            self.seg = _shm.SharedMemory(create=True, size=max(points.nbytes, 8 << 20))
        view = np.ndarray(points.shape, dtype=points.dtype, buffer=self.seg.buf)
        view[...] = points
        self.conn.send(("rails", self.seg.name, points.shape, points.dtype.descr))
        return _Pending(self.conn)

    def call(self, fn, args):
        self.conn.send(("call", fn.__module__, fn.__name__, args))
        return _Pending(self.conn)

    def close(self):
        try:
            self.conn.send(None)
            self.proc.join(timeout=5)
        except Exception:
            pass
        if self.seg is not None:
            self.seg.close()
            self.seg.unlink()
            self.seg = None


def _get_helper(i):
    if i not in _helpers:
        _helpers[i] = _Helper()
        atexit.register(_helpers[i].close)
    return _helpers[i]


def start_helpers():
    """Запустить процессы заранее (запуск — доли секунды, не в кадре)."""
    if use_process():
        for i in range(N_HELPERS):
            _get_helper(i)


def submit_rails(fn, points):
    """rail_samples(points): в процессе 0, в потоке или сразу — по режиму.
    Возвращает (глубины, центры колеи, записи, колея); в режиме процесса записи
    срезов не передаются (None) — конвейеру пути они не нужны."""
    if use_process():
        return _get_helper(0).rails(points)
    return submit(fn, points)


def submit_to(i, fn, *args):
    """fn(*args) в процессе i (fn — функция уровня модуля), в потоке или сразу."""
    if use_process():
        return _get_helper(i % N_HELPERS).call(fn, args)
    return submit(fn, *args)


def submit_refine(fn, *args):
    """«Память» трекера — в процессе 0 (рельсы к этому времени уже посчитаны)."""
    return submit_to(0, fn, *args)
