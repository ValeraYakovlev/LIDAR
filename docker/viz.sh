#!/bin/bash
# RViz2 на виртуальном экране :1, экран — в браузер через noVNC.
set -e
source /opt/ros/humble/setup.bash
export DISPLAY=:1
Xtigervnc :1 -geometry "$VIZ_GEOMETRY" -depth 24 -localhost yes -SecurityTypes None \
    -AlwaysShared >/tmp/xvnc.log 2>&1 &
sleep 1
openbox >/dev/null 2>&1 &
websockify --web /usr/share/novnc "$VIZ_BIND:$VIZ_PORT" 127.0.0.1:5901 >/tmp/novnc.log 2>&1 &
sed "s/Fixed Frame: .*/Fixed Frame: $FIXED_FRAME/" /opt/viz/metro.rviz > /tmp/metro.rviz
echo "RViz2: http://$VIZ_BIND:$VIZ_PORT/vnc.html?autoconnect=1&resize=scale"
exec rviz2 -d /tmp/metro.rviz
