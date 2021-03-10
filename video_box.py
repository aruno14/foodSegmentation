import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import datetime
import csv

import image_similarity_measures
from image_similarity_measures.quality_metrics import rmse, psnr, fsim

PATH_TO_MODEL_DIR = 'detection_model'
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)
crop_margin = 10
similarity_trigger = 0.4

def image_distance(image1, image2):
    return fsim(image1, image2)

def imageDetect(image_np, previousImages):
    now = datetime.datetime.now()
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)
    newImages = []
    labelCount = {0:0, 1:0, 2:0}
    filesName = []
    isNew = False
    for i, box in enumerate(detections['detection_boxes']):
        score = detections['detection_scores'][i]
        classe = detections['detection_classes'][i]
        if score > 0.6 and box[3] - box[1] < 0.8 and box[2] - box[0] < 0.8 and box[3] - box[1] > 0.1 and box[2] - box[0] > 0.1:
            labelCount[classe]+=1

            x1 = box[1] * img.size[0]
            y1 = box[0] * img.size[1]
            x2 = box[3] * img.size[0]
            y2 = box[2] * img.size[1]

            #print('box', x1, y1, x2, y2)

            crop = img.crop((max(x1-crop_margin, 0), max(y1-crop_margin, 0), min(x2+crop_margin, img.size[0]), min(y2+crop_margin, img.size[1])))
            cropNp = np.array(crop)
            cropNpCrop = cv2.resize(cropNp, (64, 64))

            maxSimilarity = 0
            maxImage = ()
            for path, image, scorePre in previousImages:
                similarity = image_distance(image, cropNpCrop)
                if similarity > maxSimilarity:
                    maxSimilarity = similarity
                    maxImage = (path, image, scorePre)
                #print('similarity', similarity)

            if maxSimilarity < similarity_trigger:
                filename = "{}_{}-{}-{}_{}-{}-{}-{}_{}_{}.jpg".format(now.timestamp(), now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond, i, classe)
                print('save', filename)
                filesName.append(filename)
                crop.save(filename)
                newImages.append((filename, cropNpCrop, score))
                isNew = True
            else:
                newImages.append(maxImage)
                filesName.append(maxImage[0])

            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0))
    with open("video_log.csv", "a+") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([now.timestamp(), now.isoformat(), isNew, labelCount[0], labelCount[1], labelCount[2], str([x for x, y, z in newImages]), str([z for x, y, z in newImages])])
    return np.array(img), newImages

video = cv2.VideoCapture('example/assiettes.mp4')
SIZE = (800, 600)
fps = video.get(cv2.CAP_PROP_FPS)

inWriter = cv2.VideoWriter('inpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, SIZE)
outWriter = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, SIZE)

frameCount = 0
previousImage = []
previousFrame = None
while True:
    grabbed, frame = video.read()
    if grabbed != True:
        break
    #print("====New frame", frameCount)
    frame = cv2.resize(frame, SIZE)
    previousFrame = frame
    inWriter.write(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.uint8)
    predicted_image, previousImage = imageDetect(frame, previousImage)

    predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_RGB2BGR)
    outWriter.write(predicted_image)

    frameCount+=1
    previousFrame = frame

video.release()
inWriter.release()
outWriter.release()
