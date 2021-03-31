#!/bin/bash

# Download models from github releases
# => Run this script manually to get the required models

wget https://github.com/aruno14/foodSegmentation/releases/download/v0.2/detection_model.zip
#wget https://github.com/aruno14/foodSegmentation/releases/download/v0.2/plate_model.zip
wget https://github.com/aruno14/foodSegmentation/releases/download/v0.2/plate_model_features.h5 -O plate_model_features.h5
wget https://github.com/aruno14/foodSegmentation/releases/download/v0.2/plate_model.h5 -O plate_model.h5

# to unpack
#unzip detection_model.zip
