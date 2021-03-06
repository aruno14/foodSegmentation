# What does this script do:
# - Listen to filesystem event for input image (e.g., timelapse camera script)
# - Run detection model (tensorflow)
# - Crop image and do something (e.g., upload to remote server over SSH for further processing)

# Rationale:
# This script is useful if loading the model takes significant time, for example on Raspberry Pi.

# Tested with model from:
# https://github.com/aruno14/foodSegmentation/releases/download/detection/detection_model.zip

# Useful documentation:
# - watchdog: https://pythonhosted.org/watchdog/quickstart.html

import os
import sys
import csv
import time
import datetime
import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

### Settings
SCRIPT_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
PATH_TO_MODEL_DIR = SCRIPT_PATH + '/detection_model/'
PATH_TO_MODEL_FEATURES_DIR = SCRIPT_PATH + '/plate_model_features.h5'
PATH_TO_MODEL_PLATE_DIR = SCRIPT_PATH + '/plate_model.h5'
BOX_DRAW_THRESHOLD = 0.5
LISTENING_PATH = sys.argv[1]
OUTPUT_PATH = '/tmp/foodSeg/' #'output'　'/tmp/foodSeg/'

BOX_SIZE_MIN = 0.1
BOX_SIZE_MAX = 0.8
BOX_MARGIN = 10

SIMILARITY_TRIGGER = 4
### End of settings

print('Image box watchdog, running from {}'.format(SCRIPT_PATH), flush=True)

print('Loading model from "{}"...'.format(PATH_TO_MODEL_DIR), flush=True)
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)
print('Model loaded', flush=True)

print('Loading model from "{}"...'.format(PATH_TO_MODEL_PLATE_DIR), flush=True)
features_model = tf.keras.models.load_model(PATH_TO_MODEL_FEATURES_DIR)
plate_model = tf.keras.models.load_model(PATH_TO_MODEL_PLATE_DIR)
print('Models loaded', flush=True)

images_history = []

def image_similarity(features1, features2):
    return np.sqrt(np.sum((features1 - features2) ** 2))

def addToHistory(image):
    images_history.append(image)
    if len(images_history) > 10:
        images_history.pop()

def detect_bbox_from_image_path(image_path):
    global images_history
    print('Running inference for "{}"...'.format(image_path), flush=True)

    image_filename = os.path.basename(image_path)
    image_name = os.path.splitext(image_filename)[0]
    image_ext = os.path.splitext(image_filename)[1]
    # image_name is expected to be a timestamp
    image_timestamp = datetime.datetime.fromtimestamp(int(image_name))
    # TODO: fallback to datetime.now if filename is not timestamp
    image_timestamp_str = str(int(image_timestamp.timestamp()))
    if image_ext != '.jpg':
        print('Skip non JPEG file', flush=True)
        return

    # load image into numpy array
    image_np = np.array(Image.open(image_path))
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    # delete image files
    os.remove(image_path)

    # process results
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()
    img = Image.fromarray(image_np_with_detections)

    # crop and draw each detection box
    box_img = img.copy()
    draw = ImageDraw.Draw(box_img)
    labelCount = {0:0, 1:0, 2:0, 3:0}
    labeledImage = {0:[], 1:[], 2:[], 3:[]}
    images_history_previous = [x for x in images_history]
    for i, box in enumerate(detections['detection_boxes']):
        score = detections['detection_scores'][i]
        classe = detections['detection_classes'][i]
        if score > BOX_DRAW_THRESHOLD:
            width = box[3] - box[1]
            height = box[2] - box[0]
            if not (width < BOX_SIZE_MAX and height < BOX_SIZE_MAX and width > BOX_SIZE_MIN and height > BOX_SIZE_MIN):
                print('Skip box (invalid size)', width, height, flush=True)
                continue
            print(i, box, classe, score)

            x1 = int(box[1] * img.size[0])
            y1 = int(box[0] * img.size[1])
            x2 = int(box[3] * img.size[0])
            y2 = int(box[2] * img.size[1])
            x_len = x2 - x1
            y_len = y2 - y1
            print('box {}x{} from ({}, {}) to ({}, {})'.format(x_len, y_len, x1, y1, x2, y2))
            draw.rectangle((max(x1-BOX_MARGIN, 0), max(y1-BOX_MARGIN, 0), min(x2+BOX_MARGIN, x2*img.size[0]), min(y2+BOX_MARGIN, y2*img.size[1])), outline=(0, 255, 0))

            crop_img = img.crop((max(x1-BOX_MARGIN, 0), max(y1-BOX_MARGIN, 0), min(x2+BOX_MARGIN, x2*img.size[0]), min(y2+BOX_MARGIN, y2*img.size[1])))
            crop_img_resize = crop_img.resize((128, 128))
            crop_filename = "{}_{}-{}.jpg".format(image_timestamp_str, i, classe)

            image_features = features_model.predict(np.expand_dims(np.array(crop_img_resize)/255, axis=0))[0]
            platePrediction = [0, -1]
            if classe == 1:
                platePrediction = plate_model.predict(np.expand_dims(image_features, axis=0))[0]
                if platePrediction[1] < 0.2:
                    print("Seems to be an empty plate", platePrediction, flush=True)
                    continue

            hasSimilar = False
            for date, filename, image_features2 in images_history_previous:
                similarity = image_similarity(image_features, image_features2)
                if similarity < SIMILARITY_TRIGGER:
                    print('Similar to previous images', similarity, filename, flush=True)
                    hasSimilar = True
                    crop_filename = filename
                    break

            labeledImage[classe].append((crop_filename, score, hasSimilar, platePrediction[1]))
            labelCount[classe]+=1

            if hasSimilar:
                print("Do not copy similar image", flush=True)
                continue

            dest_path_crop = "{}/{}".format(OUTPUT_PATH, crop_filename)
            crop_img.save(dest_path_crop)
            addToHistory((image_timestamp, crop_filename, image_features))

            os.system('/home/pi/foodSegmentation/script/rpi/03_send.sh ' + dest_path_crop)
            os.remove(dest_path_crop)

    # for debugging only
#    dest_path_raw  = "{}/{}-raw.jpg".format(OUTPUT_PATH, image_timestamp_str)
#    dest_path_box  = "{}/{}-box.jpg".format(OUTPUT_PATH, image_timestamp_str)
#    img.save(dest_path_raw)
#    box_img.save(dest_path_box)
#    os.system('/home/pi/foodSegmentation/script/rpi/03_send.sh ' + dest_path_raw)
#    os.system('/home/pi/foodSegmentation/script/rpi/03_send.sh ' + dest_path_box)
    # end of debugging only

    if labelCount[0] == 0  and labelCount[1] == 0  and labelCount[2] == 0:
        print('Nothing found, nothing was uploaded...', flush=True)
        return

    dest_path_out_csv  = "{}/{}.csv".format(OUTPUT_PATH, image_timestamp_str)

    with open(dest_path_out_csv, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(
            [image_timestamp_str, image_timestamp.isoformat(),
            labelCount[0], labelCount[1], labelCount[2],

            str([x for x, y, z, a in labeledImage[0]]),
            str([x for x, y, z, a in labeledImage[1]]),
            str([x for x, y, z, a in labeledImage[2]]),

            str([y for x, y, z, a in labeledImage[0]]),
            str([y for x, y, z, a in labeledImage[1]]),
            str([y for x, y, z, a in labeledImage[2]]),

            str([z for x, y, z, a in labeledImage[0]]),
            str([z for x, y, z, a in labeledImage[1]]),
            str([z for x, y, z, a in labeledImage[2]]),

            str([a for x, y, z, a in labeledImage[1]]),
            ]
        )

    # send
    os.system('/home/pi/foodSegmentation/script/rpi/03_send.sh ' + dest_path_out_csv)
    os.remove(dest_path_out_csv)

    print('End of inference', flush=True)

class CustomEventHandler(FileSystemEventHandler):
    def on_moved(self, event):
        #path = event.src_path
        path = event.dest_path
        print('File move event to "{}"'.format(path))
        detect_bbox_from_image_path(path)

# Note: RPi camera tool 'raspistill' will create 'output.jpg~' first, and move it to
#       'output.jpg' once the file is completely written.

# watchdog event loop
print('Listening to file events in {}'.format(LISTENING_PATH))
event_handler = CustomEventHandler()
observer = Observer()
observer.schedule(event_handler, LISTENING_PATH, recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
