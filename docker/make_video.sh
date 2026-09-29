#!/bin/bash
# Видео работы по ТЗ из записи экрана живого прогона (образ показа, ffmpeg).
#
#   make_video.sh <запись экрана.mp4> <журнал узла.jsonl> <выход.mp4>
#   GALLERY=/out/gallery make_video.sh ...    — ещё и галерея GIF прогонов в конце
#
# Титр → как запустить и что на экране → живой прогон с подписью → итог прогона
# по журналу узла. Подпись к прогону — CAPTION (или файл CAPTION_FILE); SPEED — во сколько раз
# ускорить запись экрана (прогон в темпе < 1 → обратно к реальному времени).
set -euo pipefail
RAW=$1 LOG=$2 OUT=$3
CAPTION=${CAPTION:-живой прогон}
W=1920 H=1080 FPS=20
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
MONO=/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf
T=$(mktemp -d)

cat > "$T/title.txt" <<'EOF'
Обнаружение препятствий на пути поезда метро по 3D-лидару

тоннель → облако точек → алгоритм → препятствие → расстояние до него
EOF
cat > "$T/title2.txt" <<'EOF'
Объекты не классифицируются: по каждому кадру строится геометрия тоннеля —
ось пути, стены, крен, профиль по высоте — и вдоль пути ставится габарит
вагона 81-717. Препятствие — всё, что попало в габарит и удержалось
на своём месте в тоннеле 3 кадра из 5. Расстояние — вдоль пути от лидара.
EOF
cat > "$T/run.txt" <<'EOF'
Запуск (Docker: Ubuntu 22.04 + ROS 2 Humble)

docker build -f docker/Dockerfile -t metro-obstacle .
docker run --rm --net=host --ipc=host metro-obstacle                 # узел
docker run --rm --net=host --ipc=host -v <запись>:/bag:ro metro-obstacle \
    ros2 bag play /bag --read-ahead-queue-size 10 --delay 3 --wait-for-all-acked 5000
узел обрабатывает каждый кадр по порядку (queue=all), ни один не пропускается

На экране — RViz2, подписан на топики узла
  облако лидара (каждая 10-я точка, цвет — высота)
  коридор — габарит вагона вдоль найденного пути: зелёный — свободно, красный — препятствие
  красные точки — всё, что попало в габарит; красный блок — подтверждённое препятствие,
  над ним — расстояние вдоль пути; вверху сцены — ответ кадра: OBSTACLE 63.5 m / CLEAR to 140 m
  внизу — ответ узла на каждый кадр (/obstacle/status): есть ли препятствие, расстояние, время
EOF
python3 - "$LOG" "${SPEED:-1}" > "$T/result.txt" <<'EOF'
import json, sys
rows = [json.loads(s) for s in open(sys.argv[1], encoding="utf-8")]
med = lambda a: sorted(a)[len(a) // 2]
rec = rows[-1]["received"]
det = [r for r in rows if r["detected"]]
hz = [r["rx_hz"] for r in rows if r.get("rx_hz")]
print("Итог этого прогона (журнал узла)\n")
speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
print(f"кадров пришло {rec}, обработано {len(rows)} ({len(rows) / rec:.0%}); "
      f"поток {med(hz):.1f} кадров/с")
if speed != 1:
    print(f"(запись проиграна в темпе {1 / speed:.1f} — узел успевает каждый кадр без очереди)")
print(f"обработка кадра — медиана {med([r['proc_ms'] for r in rows]):.0f} мс, "
      f"задержка ответа — медиана {med([r['latency_ms'] for r in rows]):.0f} мс")
print(f"препятствие в габарите — в {len(det)} обработанных кадрах, "
      f"дальше всего — на {max((r['distance'] for r in det), default=0):.0f} м")
print(f"путь построен — медиана до {med([r['limit'] for r in rows if r['ok']]):.0f} м")
EOF

card() {   # card <имя> <секунд> <drawtext...>
    ffmpeg -loglevel error -y -f lavfi -i "color=c=0x14141c:s=${W}x${H}:d=$2:r=$FPS" \
        -vf "$3" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p "$T/$1.mp4"
}
txt() {    # txt <файл> <x> <y> <размер> <цвет> [шрифт]
    echo "drawtext=fontfile=${6:-$FONT}:expansion=none:textfile=$1:x=$2:y=$3:fontsize=$4:fontcolor=$5:line_spacing=14"
}
card c1 8 "$(txt "$T/title.txt" "(w-tw)/2" 330 42 white),$(txt "$T/title2.txt" "(w-tw)/2" 560 32 0xb8b8c8)"
card c2 14 "$(txt "$T/run.txt" 120 170 30 0xe6e6e6 "$MONO")"
card c4 10 "$(txt "$T/result.txt" 160 380 36 white)"
if [ -n "${CAPTION_FILE:-}" ]; then cp "$CAPTION_FILE" "$T/caption.txt"; else printf '%s' "$CAPTION" > "$T/caption.txt"; fi
ffmpeg -loglevel error -y -i "$RAW" -vf "setpts=PTS/${SPEED:-1},scale=$W:$H,fps=$FPS,drawtext=fontfile=$FONT:expansion=none:textfile=$T/caption.txt:\
x=24:y=18:fontsize=28:fontcolor=white:box=1:boxcolor=0x000000@0.55:boxborderw=10" \
    -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p "$T/c3.mp4"
# галерея (GALLERY — папка с NN.gif + NN.txt и заставкой 00_intro.txt): каждая
# GIF во весь рост справа, подпись — слева
: > "$T/list.txt"
for c in c1 c2 c3 c4; do echo "file '$T/$c.mp4'" >> "$T/list.txt"; done
if [ -n "${GALLERY:-}" ]; then
    card g00 14 "$(txt "$GALLERY/00_intro.txt" 120 300 32 0xe6e6e6)"
    echo "file '$T/g00.mp4'" >> "$T/list.txt"
    for g in "$GALLERY"/[0-9][0-9].gif; do
        n=$(basename "$g" .gif)
        ffmpeg -loglevel error -y -i "$g" -vf "scale=-2:1000:flags=lanczos,\
pad=$W:$H:590:40:color=0x14141c,fps=$FPS,$(txt "$GALLERY/$n.txt" 40 90 27 white),format=yuv420p" \
            -c:v libx264 -preset medium -crf 20 "$T/g$n.mp4"
        echo "file '$T/g$n.mp4'" >> "$T/list.txt"
    done
fi
ffmpeg -loglevel error -y -f concat -safe 0 -i "$T/list.txt" \
    -c:v libx264 -preset medium -crf 21 -pix_fmt yuv420p -movflags +faststart "$OUT"
rm -rf "$T"
echo "видео: $OUT ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" | cut -d. -f1) с)"
