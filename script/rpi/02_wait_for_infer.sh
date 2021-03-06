#!/bin/bash

# Start this script from /etc/rc.local as a background task:
# /home/pi/foodSegmentation/script/rpi/02_wait_for_infer.sh &

TMP_PATH=/tmp/foodSeg
LOG_PATH=${TMP_PATH}/log
RAW_PATH=${TMP_PATH}/raw

mkdir -p ${RAW_PATH}
python3 /home/pi/foodSegmentation/script/rpi/image_watchdog.py /tmp/foodSeg/raw/ 1> ${LOG_PATH} 2>&1

# Note: read the output log with:
#       tail -f /tmp/foodSeg/log
