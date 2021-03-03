import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import datetime

import image_similarity_measures
from image_similarity_measures.quality_metrics import rmse, psnr, fsim

PATH_TO_MODEL_DIR = 'detection_model'
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)

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
    for i, box in enumerate(detections['detection_boxes']):
        score = detections['detection_scores'][i]
        classe = detections['detection_classes'][i]
        if score > 0.6 and box[3] - box[1] < 0.8 and box[2] - box[1] < 0.8:
            x1 = box[1] * img.size[0]
            y1 = box[0] * img.size[1]
            x2 = box[3] * img.size[0]
            y2 = box[2] * img.size[1]

            #print('box', x1, y1, x2, y2)
            crop = img.crop((x1, y1, x2, y2))
            cropNp = np.array(crop)
            cropNpCrop = cv2.resize(cropNp, (64, 64))
            newImages.append(cropNpCrop)
            
            maxSimilarity = 0
            for image in previousImages:
                similarity = image_distance(image, cropNpCrop)
                maxSimilarity = max(similarity, maxSimilarity)
                #print('similarity', similarity)

            filename = "{}_{}-{}-{}_{}-{}-{}-{}_{}_{}.jpg".format(now.timestamp(), now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond, i, classe)
            if maxSimilarity < 0.4:
                print('save', filename)
                crop.save(filename)

            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0))

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
    predicted_image, previousImageTmp = imageDetect(frame, previousImage)
    
    if len(previousImageTmp) > 0:
        previousImage = previousImageTmp
    
    predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_RGB2BGR)
    outWriter.write(predicted_image)

    frameCount+=1
    previousFrame = frame

video.release()
inWriter.release()
outWriter.release()
