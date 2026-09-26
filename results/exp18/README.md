# Эксперимент 18 — дальность и точность обнаружения (knowledge.md §34)

Код — тег `far-detection-v1` (вариант `final`, проверен отложенными данными);
после отложенного замера — вариант `final_b2` (18б, запас снизу отдельно).

| файл | что |
|---|---|
| `new_synth_objects.json` | разметка New_synth, предметы 1–5 (разработка) и граница `S_B` = 657.1 м по длине пути; размечено окно за окном, как найден каждый — поле `how` |
| `new_synth_holdout_objects.json` | предметы 6–10 (отложенная часть), размечены ПОСЛЕ заморозки кода |
| `dev_eval.json` | мерило на разработке (`exp_far_eval.py`): эталон §31, `final`, `final_b2` по 9 записям; разметка `doubleT_obstacle` включает коробку на рельсе |
| `holdout_eval.json` | отложенный замер, один раз: New_synth дальше `S_B` и `roundT_squareT_pressureGate_squareT` |

Воспроизвести (нужен диск T7 и кэш `output/exp18_cache`, ~1 ГБ):

    python exp_far_cache.py --dataset /Volumes/T7/Dataset --bags doubleT_obstacle ...
    python exp_far_eval.py --summary --variants base final final_b2
    python exp_far_eval.py --holdout --variants base final        # отложенный (уже использован)
    python exp_far_mirror.py --variant final

GIF — `output/Opus 5.5/far_detection/` (`dev/`, `holdout/`, `18b/`), в git не входят.
