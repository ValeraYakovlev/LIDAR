# wall-parallel-v1 — эталон эксперимента 17 и зеркальной проверки

Замороженный результат, с которым сравнивается любая следующая правка пути и
стен. Разбор — `knowledge.md` §31 (метод, слепой замер) и §32 (зеркальная
проверка, ошибка «память держит изгиб на 70–80 м»).

- **Код** — git-тег `wall-parallel-v1`. Алгоритм (`rail_detection/parallel_path.py`)
  не менялся с коммита `e0d0e71`, зафиксированного до слепого замера.
- **Данные** — шесть записей `/Volumes/T7/Dataset` и их зеркальные копии
  `/Volumes/T7/reversed` (`mirror_dataset.py`, x → −x).

## Что здесь (в репозитории)

| файл | что |
|---|---|
| `original/<запись>.json` | покадровый выход `make_parallel_gifs.py` на исходных записях: стены, ступеньки, находки, дальность, выбор «память / заново», Δs |
| `reversed/<запись>.json` | то же на зеркальных копиях |
| `mirror_summary.json` | `eval_mirror.py` по двум папкам выше |
| `memory_bend.json` | `exp_memory_bend.py`: «память» против «заново» на 75 м |
| `exp17_summary.json` | замеры §31: на кэше разработки (`exp_parallel_dev.py`) и полным конвейером против базы §28 (`exp_parallel_box.py`, все шесть записей) |
| `checksums_artifacts.sha256` | SHA-256 GIF и JSON архива (проверяется в папке архива) |
| `checksums_dataset.sha256` | SHA-256 исходных и зеркальных записей (проверяется из `/Volumes/T7`) |

Таблица зеркальной проверки воспроизводится без нового прогона:

    python eval_mirror.py --orig results/wall-parallel-v1/original --mirror results/wall-parallel-v1/reversed

## Где GIF (в git их нет — 56 МБ)

- `output/archive/wall-parallel-v1/` — замороженная копия на этой машине
  (рабочая папка `output/Opus 5.5/` перезапишется следующим запуском);
- `/Volumes/T7/hakaton_results/wall-parallel-v1/` — копия на внешнем диске, там
  же `hakaton.bundle` — весь репозиторий с тегом (восстановление:
  `git clone hakaton.bundle hakaton`).

Раскладка архива: `original/` и `reversed/` — GIF и JSON по шести записям,
`eval/` — полные замеры (покадровые строки `exp_parallel_box.py`) и сводки.

## Проверить целостность

    cd output/archive/wall-parallel-v1 && shasum -a 256 -c checksums_artifacts.sha256
    cd /Volumes/T7/hakaton_results/wall-parallel-v1 && shasum -a 256 -c checksums_artifacts.sha256
    cd /Volumes/T7 && shasum -a 256 -c <репозиторий>/results/wall-parallel-v1/checksums_dataset.sha256

## Воспроизвести с нуля

    git checkout wall-parallel-v1
    python mirror_dataset.py                                   # /Volumes/T7/reversed
    python make_parallel_gifs.py && python make_parallel_gifs.py --validate
    python make_parallel_gifs.py --dataset /Volumes/T7/reversed --out "output/Opus 5.5/reversed"
    python make_parallel_gifs.py --dataset /Volumes/T7/reversed --out "output/Opus 5.5/reversed" --validate
    python eval_mirror.py
    python exp_parallel_cache.py && python exp_memory_bend.py

## Главное

| | |
|---|---|
| слепой прогон (`roundT_squareT_pressureGate_squareT`) | стена в коробке габарита на 40–120 м: 385 → 27 кадров; находки 1/0 → 0/0 |
| разработка (4 чистые записи) | ложные находки 44/11 → 2/0; скачков стены у поезда 160 → 8 |
| человек (`doubleT_obstacle`) | найден в 57 кадрах из 72, подтверждён в 55, вне интервала 0 (база: 61/57/0) |
| зеркальная проверка | находки совпали в 5 записях из 6; на двухпутном — подтверждённая ложная тревога (кадры 102–103), решение «память / заново» на кадре 99 на грани |
| известная ошибка | память держит изгиб на 70–80 м: где гипотезы расходятся > 0.3 м, память выбрана в 97 кадрах из 101, хотя в 3 записях из 4 хуже ложится на стены |
