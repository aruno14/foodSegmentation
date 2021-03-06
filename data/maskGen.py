import os
import cv2
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Generate mask')
parser.add_argument('--file', default="food_json.json",help='Json filepath')
parser.add_argument('--folder', default="images",help='Image folder path')
parser.add_argument('--mask', default="masks",help='Mask folder path')

args = parser.parse_args()

image_folder = args.folder
mask_folder = args.mask
json_path = args.file

source_folder = os.path.join(os.getcwd(), image_folder)
mask_folder = os.path.join(os.getcwd(), mask_folder)
if not os.path.exists(mask_folder):
    os.mkdir(mask_folder)

count = 0
file_bbs = {}

with open(json_path) as f:
  data = json.load(f)

for itr in data:
    file_name_json = data[itr]["filename"]
    print("file_name_json", file_name_json)
    file_bbs[file_name_json] = data[itr]["regions"]

print("\nDict size: ", len(file_bbs))

for file_name in file_bbs:
    print(file_name)
    original_image = cv2.imread(os.path.join(source_folder, file_name))
    mask = np.zeros((original_image.shape[0], original_image.shape[1]))
    print(file_name)
    for shape in file_bbs[file_name]:
        try:
            type = shape['region_attributes']['categorie']
            x_points = shape["shape_attributes"]["all_points_x"]
            y_points = shape["shape_attributes"]["all_points_y"]

            all_points = []
            for i, x in enumerate(x_points):
                all_points.append([x, y_points[i]])
            arr = np.array(all_points)
            if type == "assiette":
                color = (255)
            elif type == "verre":
                color = (200)
            elif type == "pain":
                color = (100)
            else:
                print("UNKNOW", filename, type)
                continue
            cv2.fillPoly(mask, [arr], color=color)
        except:
            print("Not found:", file_name)
            continue
    print("write mask", os.path.join(mask_folder, file_name.replace(".jpg", ".png")))
    cv2.imwrite(os.path.join(mask_folder, file_name.replace(".jpg", ".png")) , mask)
    count+=1

print("Images saved:", count)
