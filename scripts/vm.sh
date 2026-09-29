#!/bin/bash
# Облачная ВМ для сборки и проверки Docker-образа (Ubuntu 22.04 + ROS 2 Humble).
#
# Код пишется только здесь, в репозитории; на ВМ лежит его копия (~/metro),
# записи — в ~/data/bags/<имя>. Всё делается отсюда одной командой:
#
#   scripts/vm.sh deploy                  код → ВМ, сборка образов, тесты
#   scripts/vm.sh bag /Volumes/T7/Dataset/doubleT_obstacle   запись → ВМ
#   scripts/vm.sh offline doubleT_obstacle [вариант]         разбор записи целиком
#   scripts/vm.sh demo Dataset/doubleT_obstacle [rate] [вариант]  узел + ros2 bag play, итог
#   scripts/vm.sh video mix3 [rate] [вариант]   то же + запись экрана RViz2 → ролик demo.mp4
#       (video mix3 0.4 — темп 0.4: задержка не растёт, ролик ускорен до реального времени)
#   scripts/vm.sh montage [папка GIF]     ролик заново из последней записи экрана (+ галерея GIF)
#   scripts/vm.sh viz                     RViz2 на ВМ + туннель → http://127.0.0.1:6080
#   scripts/vm.sh fetch                   результаты ВМ → output/vm/
#   scripts/vm.sh sync | build [сервисы] | test | ssh [команда] | stop
#
# Адрес ВМ — переменная VM (по умолчанию ниже; поменялся IP — поправить здесь).

set -euo pipefail

VM=${VM:-ler0_oy@158.160.139.101}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REMOTE=metro                       # папка кода на ВМ (от домашней)
BAGS=data/bags                     # папка записей на ВМ (от домашней)
SSH=(ssh -o BatchMode=yes -o ServerAliveInterval=30)
# что уходит на ВМ — ровно то, из чего собирается образ
FILES=(requirements.txt rail_detection ros2 docker docker-compose.yml .dockerignore scripts)

on_vm() { "${SSH[@]}" "$VM" "cd ~/$REMOTE && $*"; }

sync() {
    cd "$ROOT"
    rsync -az --delete --exclude __pycache__ --exclude '._*' --exclude .DS_Store \
        -e "${SSH[*]}" "${FILES[@]}" "$VM:$REMOTE/"
    echo "код → $VM:~/$REMOTE"
}

build() { on_vm "docker compose build ${*:-detector viz}"; }

run_tests() {
    on_vm "docker run --rm metro-obstacle:latest \
        python3 -m pytest -q -p no:cacheprovider /opt/ws/src/metro_obstacle/test"
}

# после записи — ждать, пока узел разберёт очередь (queue=all): последняя строка
# журнала — с пустой очередью
DRAIN="until tail -n 1 output/ros/detections.jsonl | grep -q '\"backlog\": 0,'; do sleep 2; done; sleep 3"

bag_dir() {
    [ -n "${1:-}" ] || { echo "нужно имя записи — папка в ~/$BAGS на ВМ" >&2; exit 1; }
    echo "\$HOME/$BAGS/$1"
}

cmd=${1:-help}
shift || true
case "$cmd" in
    sync)   sync ;;
    build)  sync; build "$@" ;;
    test)   run_tests ;;
    deploy) sync; build; run_tests ;;
    bag)
        "${SSH[@]}" "$VM" "mkdir -p ~/$BAGS"
        for b in "$@"; do
            rsync -a --partial --exclude '._*' -e "${SSH[*]}" "${b%/}" "$VM:$BAGS/"
            echo "запись $(basename "$b") → $VM:~/$BAGS"
        done ;;
    offline)
        B=$(bag_dir "${1:-}")
        on_vm "mkdir -p output/ros && BAG=$B VARIANT=${2:-low_rest_b0} docker compose run --rm offline" ;;
    demo)
        B=$(bag_dir "${1:-}")
        # запись — сначала в память: сетевой SSD ВМ читает ~110 МБ/с, а плееру
        # нужно ~240 (24 МБ × 10 Гц) — с диска запись играется медленнее реальной.
        # Узел — заново на каждый прогон, журнал — с чистого листа.
        on_vm "cat $B/*.db3 > /dev/null && mkdir -p output/ros && rm -f output/ros/detections.jsonl && \
            VARIANT=${3:-low_rest_b0} docker compose up -d --force-recreate detector && sleep 3 && \
            BAG=$B RATE=${2:-1.0} docker compose run --rm player >/dev/null 2>&1 && $DRAIN && \
            docker compose logs --no-log-prefix detector | grep -E 'кадр|облако' | tail -8 && \
            echo '== итог прогона' && python3 scripts/demo_summary.py output/ros/detections.jsonl" ;;
    video)
        # то же, что demo, но с записью экрана RViz2 (раскладка для видео, 1920x1080)
        # и сборкой ролика по ТЗ → output/ros/demo.mp4 (fetch — забрать).
        # FIXED_FRAME: lidar_livox — реальные записи, hesai_lidar — синтетика;
        # LABEL — как назвать запись в подписи (по умолчанию — имя папки);
        # GALLERY — папка с GIF прогонов для конца ролика (см. montage).
        # Узел обрабатывает каждый кадр (queue=all); темп < 1 — без очереди, прогон в
        # ролике ускоряется обратно до реального времени записи (с подписью об этом).
        B=$(bag_dir "${1:-}")
        RATE=${2:-1.0}
        SPEED=$(awk "BEGIN { print 1 / $RATE }")
        CAP="живой прогон на облачной ВМ (Intel Ice Lake, 16 vCPU) · ${LABEL:-запись $(basename "$1")}"
        if [ "$SPEED" != 1 ]; then
            CAP="$CAP
запись проиграна в темпе $RATE, узел обрабатывает каждый кадр; видео ускорено в $SPEED раза — до реального времени"
        fi
        on_vm "mkdir -p output/ros && sudo -n rm -f output/ros/detections.jsonl output/ros/demo_raw.mp4 \
            output/ros/demo.mp4 && echo $SPEED > output/ros/speed.txt"
        printf '%s' "$CAP" | "${SSH[@]}" "$VM" "cat > $REMOTE/output/ros/caption.txt"
        on_vm "cat $B/*.db3 > /dev/null && \
            FIXED_FRAME=${FIXED_FRAME:-lidar_livox} VIZ_LAYOUT=video VIZ_GEOMETRY=1920x1080 \
                docker compose up -d --force-recreate viz && \
            VARIANT=${3:-low_rest_b0} QUEUE=${QUEUE:-all} \
                docker compose up -d --force-recreate detector && sleep 8 && \
            docker compose exec -d viz ffmpeg -loglevel error -y -f x11grab -draw_mouse 0 -video_size 1920x1080 \
                -framerate 20 -i :1 -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p /out/demo_raw.mp4 && \
            sleep 2 && BAG=$B RATE=$RATE docker compose run --rm player >/dev/null 2>&1 && $DRAIN && \
            docker compose exec viz pkill -INT ffmpeg && sleep 3 && \
            echo '== итог прогона' && python3 scripts/demo_summary.py output/ros/detections.jsonl | head -3"
        "$0" montage "${GALLERY:-}" ;;
    montage)
        # ролик заново из записи экрана и журнала последнего video (+ галерея GIF в
        # конце: папка с NN.gif, NN.txt и заставкой 00_intro.txt) → output/ros/demo.mp4
        G=""
        if [ -n "${1:-}" ]; then
            rsync -a --delete -e "${SSH[*]}" "${1%/}/" "$VM:$REMOTE/output/ros/gallery/"
            G="-e GALLERY=/out/gallery"
        fi
        on_vm "docker compose run --rm -e CAPTION_FILE=/out/caption.txt -e SPEED=\$(cat output/ros/speed.txt) \
            $G viz /opt/viz/make_video.sh /out/demo_raw.mp4 /out/detections.jsonl /out/demo.mp4" ;;
    viz)
        on_vm "docker compose up -d viz"
        echo "RViz2: http://127.0.0.1:6080/vnc.html?autoconnect=1&resize=scale  (Ctrl+C — закрыть туннель)"
        "${SSH[@]}" -N -L 6080:127.0.0.1:6080 "$VM" ;;
    stop)   on_vm "docker compose down" ;;
    fetch)
        mkdir -p "$ROOT/output/vm"
        rsync -az -e "${SSH[*]}" "$VM:$REMOTE/output/ros/" "$ROOT/output/vm/"
        echo "результаты → output/vm/" ;;
    ssh)
        if [ $# -gt 0 ]; then on_vm "$*"; else "${SSH[@]}" -t "$VM" "cd ~/$REMOTE && exec bash -l"; fi ;;
    *)  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
