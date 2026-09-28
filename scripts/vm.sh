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
#   scripts/vm.sh viz                     RViz2 на ВМ + туннель → http://127.0.0.1:6080
#   scripts/vm.sh fetch                   результаты ВМ → output/vm/
#   scripts/vm.sh sync | build [сервисы] | test | ssh [команда] | stop
#
# Адрес ВМ — переменная VM (по умолчанию ниже; поменялся IP — поправить здесь).

set -euo pipefail

VM=${VM:-ler0_oy@158.160.240.202}
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
        on_vm "mkdir -p output/ros && BAG=$B VARIANT=${2:-final} docker compose run --rm offline" ;;
    demo)
        B=$(bag_dir "${1:-}")
        # запись — сначала в память: сетевой SSD ВМ читает ~110 МБ/с, а плееру
        # нужно ~240 (24 МБ × 10 Гц) — с диска запись играется медленнее реальной.
        # Узел — заново на каждый прогон, журнал — с чистого листа.
        on_vm "cat $B/*.db3 > /dev/null && mkdir -p output/ros && rm -f output/ros/detections.jsonl && \
            VARIANT=${3:-final} docker compose up -d --force-recreate detector && sleep 3 && \
            BAG=$B RATE=${2:-1.0} docker compose run --rm player >/dev/null 2>&1 && sleep 3 && \
            docker compose logs --no-log-prefix detector | grep -E 'кадр|облако' | tail -8 && \
            echo '== итог прогона' && python3 scripts/demo_summary.py output/ros/detections.jsonl" ;;
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
