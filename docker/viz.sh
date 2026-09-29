#!/bin/bash
# RViz2 на виртуальном экране :1, экран — в браузер через noVNC.
# VIZ_LAYOUT=video — раскладка для записи видео (экран 1920x1080): сцена RViz2
# сверху во всю ширину, снизу — окно с ответом узла по кадрам (status_view.py).
set -e
source /opt/ros/humble/setup.bash
export DISPLAY=:1
Xtigervnc :1 -geometry "$VIZ_GEOMETRY" -depth 24 -localhost yes -SecurityTypes None \
    -AlwaysShared >/tmp/xvnc.log 2>&1 &
sleep 1
openbox >/dev/null 2>&1 &
websockify --web /usr/share/novnc "$VIZ_BIND:$VIZ_PORT" 127.0.0.1:5901 >/tmp/novnc.log 2>&1 &
sed "s/Fixed Frame: .*/Fixed Frame: $FIXED_FRAME/" /opt/viz/metro.rviz > /tmp/metro.rviz
if [ "${VIZ_LAYOUT:-}" = video ]; then
    W=${VIZ_GEOMETRY%x*}
    H=${VIZ_GEOMETRY#*x}
    TERM_H=300
    # окно RViz2 — над окном ответа; панель Displays убрана: вся ширина — сцене;
    # камера ближе: лидар у нижнего края кадра, путь впереди крупнее
    sed -i -e "s/^  Height: .*/  Height: $((H - TERM_H))/" -e "s/^  Width: .*/  Width: $W/" \
        -e '/^Panels:/,/^Visualization Manager:/{/^Visualization Manager:/!d}' \
        -e 's/^        Y: -30$/        Y: -45/' -e 's/^      Distance: 45$/      Distance: 42/' \
        -e 's/^      Pitch: 0.3$/      Pitch: 0.2/' /tmp/metro.rviz
    sed -i '1i Panels: []' /tmp/metro.rviz
    xterm -geometry 200x14+0+$((H - TERM_H)) -fa "DejaVu Sans Mono" -fs 13 -bg "#14141c" \
        -fg "#e6e6e6" -T "obstacle_detector" -e python3 /opt/viz/status_view.py &
fi
echo "RViz2: http://$VIZ_BIND:$VIZ_PORT/vnc.html?autoconnect=1&resize=scale"
exec rviz2 -d /tmp/metro.rviz
