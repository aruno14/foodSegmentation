import os
import csv
import cv2
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Generate mask')
parser.add_argument('--file', default="food_json.json",help='Json filepath')
parser.add_argument('--folder', default="images",help='Image folder path')
parser.add_argument('--box', default="boxes",help='Box folder path')
parser.add_argument('--output', default="food_box.csv",help='CSV output file')

args = parser.parse_args()

image_folder = args.folder
box_folder = args.box
json_path = args.file
output_path = args.output

source_folder = os.path.join(os.getcwd(), image_folder)
boxes_folder = os.path.join(os.getcwd(), box_folder)
if not os.path.exists(boxes_folder):
    os.mkdir(boxes_folder)
count = 0
countLabel = {}
file_bbs = {}

with open(json_path) as f:
  data = json.load(f)

for itr in data:
    file_name_json = data[itr]["filename"]
    print("file_name_json", file_name_json)
    file_bbs[file_name_json] = data[itr]["regions"]

print("\nDict size: ", len(file_bbs))

with open(output_path, mode='w') as boxFile:
    boxFile_writer = csv.writer(boxFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for file_name in file_bbs:
        original_image = cv2.imread(os.path.join(source_folder, file_name))
        print(file_name, original_image.shape[0], original_image.shape[1])
        for shape in file_bbs[file_name]:
            type = shape['region_attributes']['categorie'].replace("\n", "")
            if type not in countLabel:
                countLabel[type] = 0
            countLabel[type]+=1
            x_points = shape["shape_attributes"]["all_points_x"]
            y_points = shape["shape_attributes"]["all_points_y"]
            x_min, x_max = np.min(x_points)/original_image.shape[1], np.max(x_points)/original_image.shape[1]
            y_min, y_max = np.min(y_points)/original_image.shape[0], np.max(y_points)/original_image.shape[0]

            print(file_name, type, (x_min, y_min), (x_max, y_max))
            original_image = cv2.rectangle(original_image, (int(x_min*original_image.shape[1]), int(y_min*original_image.shape[0])), (int(x_max*original_image.shape[1]), int(y_max*original_image.shape[0])), (255, 255, 255), 3)
            boxFile_writer.writerow([file_name, type, x_min, y_min, x_max, y_max])
        cv2.imwrite(os.path.join(boxes_folder, file_name), original_image)
        count+=1

print("Box saved:", count, countLabel)
