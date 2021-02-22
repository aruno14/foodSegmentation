import os
from glob import glob
import numpy as np
import io

import tensorflow as tf

from PIL import Image, ImageDraw
import csv

IMG_HEIGHT, IMG_WIDTH=128, 128
CLASSES_COUNT = 3

files = {}
with open('data/food_box.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        filename = row[0]
        #x_files.append("data/images/"+row[0])
        labelname = row[1]
        label = 3
        if labelname == "assiette":
            label = 1
        elif labelname == "verre":
            label = 2
        labelname = labelname.encode('utf-8')
        xmin, ymin, xmax, ymax = float(row[2]), float(row[3]), float(row[4]), float(row[5])
        if filename not in files:
            files[filename] = {'xmins':[], 'xmaxs':[], 'ymins':[], 'ymaxs':[], 'classes_text':[], 'classes':[]}
        files[filename]['xmins'].append(xmin)
        files[filename]['ymins'].append(ymin)
        files[filename]['xmaxs'].append(xmax)
        files[filename]['ymaxs'].append(ymax)
        files[filename]['classes_text'].append(labelname)
        files[filename]['classes'].append(label)

def process_img(file_path:str, channels=3):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=channels)
    img = tf.image.convert_image_dtype(img, tf.float32)#0~1
    img = tf.image.resize(img, size=(IMG_HEIGHT, IMG_WIDTH))
    return img

def create_tf_example(filename, xmins, ymins, xmaxs, ymaxs, classes_text, classes):
    file_path = "data/images/"+filename
    fin = open(file_path, 'rb')
    encoded_jpg = fin.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    print(file_path, width, height)

    filename = filename.encode('utf-8')
    image_format = b'jpg'

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))
    return tf_example


def prepareData(files, saveFile_path):
    writer = tf.io.TFRecordWriter(saveFile_path)
    count = 0
    for i, filename in enumerate(files):
        xmins = files[filename]['xmins']
        ymins = files[filename]['ymins']
        xmaxs = files[filename]['xmaxs']
        ymaxs = files[filename]['ymaxs']
        classes_text = files[filename]['classes_text']
        classes = files[filename]['classes']
        tf_example = create_tf_example(filename, xmins, ymins, xmaxs, ymaxs, classes_text, classes)
        writer.write(tf_example.SerializeToString())
        count+=1
    writer.close()

print("#Create tfrecords")
prepareData(files, "train.tfrecords")
