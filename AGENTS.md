# AGENTS.md — с чего начать агенту

Хакатон «Московский транспорт»: по потоку ROS 2 `PointCloud2` от 3D-лидара
беспилотного поезда метро находить препятствие на пути и расстояние до него.
Язык проекта — русский: документы, комментарии, сообщения коммитов.

Этот файл — короткий вход. Решение и его запуск — [README.md](README.md);
подробная карта репозитория — [AgentReadme.md](AgentReadme.md); база знаний по
всем экспериментам — [docs/knowledge.md](docs/knowledge.md) (§1–§40); планы и
итоги экспериментов — [docs/experiments/](docs/experiments/README.md).

## Раскладка

| где | что |
|---|---|
| `rail_detection/` | алгоритм |
| `ros2/metro_obstacle/`, `docker/`, `docker-compose.yml` | решение по ТЗ: ROS 2-узел и Docker |
| `scripts/` | `vm.sh` — облачная ВМ с Mac, `demo_summary.py` |
| `docs/` | ТЗ, база знаний, планы экспериментов |
| `research/` | скрипты экспериментов, `results/` (в git), `output/` (кэши, эталоны, GIF — не в git) |

## Окружение

```bash
python3 -m venv venv && source venv/bin/activate   # проверено на Python 3.9.6
pip install -r requirements.txt                    # версии закреплены — не обновлять без сверки
cd research && export PYTHONPATH=..                # все исследовательские скрипты — отсюда
```

Данные **не в репозитории** — на внешнем диске `/Volumes/T7` (без него скрипты
с записями падают с `AnyReaderError: ... paths are missing`). Пути `output/…`
ниже — внутри `research/`.

| путь | что |
|---|---|
| `/Volumes/T7/Dataset/` | 6 реальных записей (~33 ГБ); препятствие только в `doubleT_obstacle` — человек (кадры 4–75) и коробка на левом рельсе (кадры 50–200) |
| `/Volumes/T7/reversed/` | их зеркальные копии (x → −x) — для зеркальной проверки |
| `/Volumes/T7/Synthetic_data/` | 3 синтетические записи с разметкой (`/lidar_points_labeled`) |
| `/Volumes/T7/New_synth_data/` | синтетика с 10 предметами; открывается через обёртку `output/new_synth/cloud_with_fake_obj/` (ссылка на `.db3` + сделанный нами `metadata.yaml`); `--dataset output/new_synth` |
| `/Volumes/T7/Last_synth_data/` | 8 синтетических выездов в двухпутный `conv_r{R}_a{A}` (рабочий за 5 м до слияния, разметка, `/tf`); распакованы из `.rar`, `.db3` и `yaml/` лежат раздельно — обёртка `output/last_synth/<запись>/` (пересоздаёт `exp20_wrap.py`), `--dataset output/last_synth`; зеркала — `/Volumes/T7/reversed/Last_synth_data` |

## Конвейер кадра

1. `rail_detection.parallel_path.ParallelGauge.update(points)` — путь и стены
   тоннеля одной кривой (§31), крен по рельсам, Δs между кадрами (§36: с вычетом
   рисунка, стоящего на лидаре; прежний — `RAIL_DS_MODE=density RAIL_DS_HOLD=last`).
2. `rail_detection.far_detect.FarDetector(VARIANTS[...]).update(res)` —
   обнаружение (§34): габарит кузова 81-717 (2.67 × 3.65 м, вырез под
   контактный рельс), профиль пути по высоте из вида сбоку, скопления с порогом
   по дальности, подтверждение каждого скопления (3 из 5 кадров).
   **`low_rest_b0` — рабочий и по умолчанию** (§37; слепая проверка на
   FINAL_STEP — §40): видит и низкие предметы на рельсах (коробка 92 из 125
   кадров). Остальные варианты в `VARIANTS` — история экспериментов, в решении
   не используются.
3. Ускорение без изменения ответа: `RAIL_WORKERS=4 RAIL_PROCESS=1` (§35;
   по умолчанию последовательно), `RAIL_JIT=1` — перебор ступенек стен в numba
   (§39, по умолчанию; `0` — прежний код). ~78 мс на кадр на M4, ~157 мс на
   облачной ВМ. **С `RAIL_PROCESS=1` запускающий скрипт обязан иметь защиту
   `if __name__ == "__main__":`** — вспомогательные процессы стартуют через
   spawn и заново выполняют главный модуль (скрипт со стандартного ввода не
   годится).
4. ROS 2-узел `ros2/metro_obstacle` (`obstacle_detector`): приём облака,
   очередь «свежий кадр», тот же конвейер, публикация `/obstacle/*` — README.

## Главные команды (из `research/`, `PYTHONPATH=..`)

```bash
# сверка «ответ тот же» (после ЛЮБОЙ правки кода конвейера) — эталон текущего кода
python exp_speed.py check --all --golden output/exp22_golden --variants low_rest_b0 --jobs 6
python exp_speed.py check --all --golden output/exp22_golden --variants low_rest_b0 --jobs 3 --workers 4 --process
#   Mac ↔ Linux и numba под Linux — с --ignore alt_x (§39)

# замер скорости по стадиям (--bench-variant, по умолчанию low_rest_b0)
python exp_speed.py bench --label my --workers 4 --process

# мерило обнаружения на кэше (дальность, непрерывность, ложные) — секунды
python exp_far_eval.py --summary --variants low_rest_b0

# GIF по записи — отсматривать самому
RAIL_WORKERS=4 RAIL_PROCESS=1 python make_far_gifs.py --dataset /Volumes/T7/Dataset \
    --bags doubleT_obstacle --variant low_rest_b0 --out "output/Opus 5.5/check"

# зеркальная проверка находок
python exp_far_mirror.py --variant low_rest_b0

# выезд в двухпутный (§36): правда по /tf и разметке, мерило, зеркала
python exp20_truth.py
python exp20_eval.py --stats "<папка прогона>/last_synth" --false
```

Решение — из корня: `docker build -f docker/Dockerfile -t metro-obstacle .` и
дальше по README. Облачная ВМ — только `scripts/vm.sh` с Mac (`deploy`, `bag`,
`demo`, `viz`); код на ВМ не править.

## Правила работы — не нарушать

- **Эксперимент — своя ветка** `feature/<тема>`; план в
  `docs/experiments/NN-*.md`, итог — раздел `docs/knowledge.md` §N; в `main` —
  коммитом слияния (`--no-ff`), ветка после слияния удаляется. Постоянная ветка
  одна — `main`.
- **Подгонять под отдельную запись запрещено.** Параметры выбираются на
  разработочном наборе и физике; отложенные данные замораживаются коммитом ДО
  первой правки и меряются ОДИН раз. Все отложенные наборы для алгоритма уже
  использованы (§31, §34, §36) — новому эксперименту нужен новый отложенный
  прогон. Нетронутых записей не осталось.
- **Любое заявленное улучшение — зеркальная проверка** (§32, `exp_far_mirror.py`,
  `eval_mirror.py`).
- **Ускорение не меняет ответ.** Правка ради скорости принимается, только если
  `exp_speed.py check` совпадает с эталоном на всех записях. Эталон разработки —
  `output/exp22_golden` (тег `speed-exact-ref`, = `numba-ref`); эталон привязан к
  платформе: numpy под Linux (AVX-512, SVML) не бит в бит с Mac — там свой
  эталон и `--ignore alt_x`.
- **GIF отсматривать самому** до того, как писать о результате.

## Состояние (2026-09-29)

- `main` — всё: эксперименты до 23, ROS 2-узел, Docker, README по ТЗ.
  Теги: `wall-parallel-v1`, `far-detection-v1`, `speed-ref`, `double-track-v1`,
  `speed-exact-ref`, `numba-ref`.
- Работа AnRiChie (ML, XGBoost) — отдельные ветки на GitHub, в `main` не входит.
- Отчёт, презентация и видео — не в репозитории (локально — `Presentation and report/`).
- Не сделано по ТЗ: замер на стенде (i7-9700E). Видео работы —
  `scripts/vm.sh video <запись>` (живой прогон на ВМ, запись экрана RViz2).
- FINAL_STEP (`/Volumes/T7/FINAL_STEP/mix3`, 2026-09-29; в материалах для жюри —
  «новая синтетическая запись») — слепая проверка `low_rest_b0` проведена (§40),
  запись использована.
- Узел по умолчанию обрабатывает каждый кадр (`queue=all`, §40): на облаке
  ~170 мс на реальный кадр — очередь и задержка растут; `queue:=latest` — без
  очереди, ~половина кадров. Ускорять дальше — подгонка формы стен
  (`_observed_edge`, только до 1e-6) или конвейер через кадр (§39).
- Открытые дефекты: ложные в раструбе двухпутного — обе гипотезы пути
  ошибаются согласно, стены вдали не параллельны пути (§36); низкие предметы
  (0.1–0.2 м) видны не в каждом кадре, тонкие (шнур 5 см) не видны; путь
  строится до 150 м.
