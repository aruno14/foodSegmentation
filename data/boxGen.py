import os
import csv
import cv2
import json
import numpy as np

source_folder = os.path.join(os.getcwd(), "images")
boxes_folder = os.path.join(os.getcwd(), "boxes")
if not os.path.exists(boxes_folder):
    os.mkdir(boxes_folder)
json_path = "food_json.json"
count = 0
file_bbs = {}

with open(json_path) as f:
  data = json.load(f)

for itr in data:
    file_name_json = data[itr]["filename"]
    print("file_name_json", file_name_json)
    file_bbs[file_name_json] = data[itr]["regions"]

print("\nDict size: ", len(file_bbs))

with open('food_box.csv', mode='w') as boxFile:
    boxFile_writer = csv.writer(boxFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for file_name in file_bbs:
        original_image = cv2.imread(os.path.join(source_folder, file_name))
        print(original_image.shape[0], original_image.shape[1])
        for shape in file_bbs[file_name]:
            #try:
            type = shape['region_attributes']['categorie']
            x_points = shape["shape_attributes"]["all_points_x"]
            y_points = shape["shape_attributes"]["all_points_y"]
            x_min, x_max = np.min(x_points)/original_image.shape[1], np.max(x_points)/original_image.shape[1]
            y_min, y_max = np.min(y_points)/original_image.shape[0], np.max(y_points)/original_image.shape[0]

            print(file_name, type, (x_min, y_min), (x_max, y_max))
            original_image = cv2.rectangle(original_image, (int(x_min*original_image.shape[1]), int(y_min*original_image.shape[0])), (int(x_max*original_image.shape[1]), int(y_max*original_image.shape[0])), (255, 255, 255), 3)
            boxFile_writer.writerow([file_name, type, x_min, y_min, x_max, y_max])

            #except:
            #    print("Not found:", file_name)
            #    continue
        cv2.imwrite(os.path.join(boxes_folder, file_name), original_image)
        count+=1

print("Box saved:", count)
