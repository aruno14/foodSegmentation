#!/bin/bash

# Copy file to remote server

SRC_PATH=$1

scp -i /home/pi/ssh_key ${SRC_PATH} root@my.server.net:/destination/path/for/images/
