#!/bin/bash

# Image capture script for Raspberry Pi
# Note: this script is meant to be run in a cyclic way (e.g, cron)

# Use "crontab -e" to edit cron settings
# Example of setting for a capture every minute:
# # m h  dom mon dow   command
# */1 *  *   *   *     /home/pi/foodSegmentation/script/rpi/01_capture.sh

# Destination folder for the images
TMP_DIR=/tmp/foodSeg/raw/
TIMESTAMP=`date +%s`
# Use "date +%s" for UNIX timestamp

mkdir -p ${TMP_DIR}

# take a picture with default settings
raspistill -o ${TMP_DIR}/${TIMESTAMP}.jpg
