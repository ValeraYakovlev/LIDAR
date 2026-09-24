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
