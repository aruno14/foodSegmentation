import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import datetime

PATH_TO_MODEL_DIR = 'detection_model'
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)

def imageDetect(image_np):
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
    for i, box in enumerate(detections['detection_boxes']):
        score = detections['detection_scores'][i]
        classe = detections['detection_classes'][i]
        if score > 0.6:
            x1 = box[1] * img.size[0]
            y1 = box[0] * img.size[1]
            x2 = box[3] * img.size[0]
            y2 = box[2] * img.size[1]

            print('box', x1, y1, x2, y2)
            crop = img.crop((x1, y1, x2, y2))

            filename = "{}_{}-{}-{}_{}-{}-{}-{}_{}_{}.jpg".format(now.timestamp(), now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond, i, classe)
            #print('save', filename)
            #crop.save(filename)

            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0))

    return np.array(img)

video = cv2.VideoCapture('example/assiettes.mp4')
SIZE = (800, 600)
fps = video.get(cv2.CAP_PROP_FPS)

inWriter = cv2.VideoWriter('inpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, SIZE)
outWriter = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, SIZE)

frameCount = 0
while True:
    grabbed, frame = video.read()
    if grabbed != True:
        break
    print("====New frame", frameCount)
    frame = cv2.resize(frame, SIZE)
    inWriter.write(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.uint8)
    predicted_image = imageDetect(frame)
    predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_RGB2BGR)
    outWriter.write(predicted_image)

    frameCount+=1

video.release()
inWriter.release()
outWriter.release()
cv2.destroyAllWindows()
