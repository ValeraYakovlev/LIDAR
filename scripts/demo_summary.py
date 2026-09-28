#!/usr/bin/env python3
"""Итог живого прогона узла по его журналу (JSON Lines, /obstacle/status на кадр).

    python3 scripts/demo_summary.py output/ros/detections.jsonl

Номер кадра — порядковый номер пришедшего сообщения (с нуля). Он совпадает с
номером кадра в записи, пока узел получает все кадры с первого (сейчас — 200–201
из 201); по меткам времени номер считать нельзя: в записях есть пропуски
лидара (doubleT_obstacle: 201 сообщение на 20.4 с).
"""

import json
import sys


def main(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    if not rows:
        print("кадров нет")
        return
    for r in rows:
        r["rec"] = r["received"] - 1
    proc = sorted(r["proc_ms"] for r in rows)
    print(f"вариант {rows[0]['variant']}: пришло {rows[-1].get('received', '?')} кадров, "
          f"обработано {len(rows)}, обработка — медиана {proc[len(proc) // 2]:.0f} мс")
    det = [r for r in rows if r["detected"]]
    if not det:
        print("препятствие: не найдено ни в одном кадре")
        return
    # отрезки подряд идущих находок (разрыв больше 5 кадров записи — новый отрезок)
    spans, cur = [], [det[0]]
    for r in det[1:]:
        if r["rec"] - cur[-1]["rec"] > 5:
            spans.append(cur)
            cur = []
        cur.append(r)
    spans.append(cur)
    print(f"препятствие: в {len(det)} обработанных кадрах")
    for sp in spans:
        d = [r["distance"] for r in sp]
        print(f"  кадры записи {sp[0]['rec']}–{sp[-1]['rec']}: {min(d):.1f}–{max(d):.1f} м "
              f"({len(sp)} кадров)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/ros/detections.jsonl")
