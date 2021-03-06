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
import time
import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

### Settings
SCRIPT_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
PATH_TO_MODEL_DIR = SCRIPT_PATH + '/detection_model/'
BOX_DRAW_THRESHOLD = 0.2
LISTENING_PATH = sys.argv[1]
OUTPUT_PATH = '/tmp/foodSeg/'
# Average size of the (square) box in pixel, e.g. 350x350
BOX_SIZE_AVG = 350
# Maximum difference in pixel relative to the average size
# (for a box to be considered valid)
BOX_SIZE_DIFF_MAX = 100
### End of settings

BOX_SIZE_MIN = BOX_SIZE_AVG - BOX_SIZE_DIFF_MAX
BOX_SIZE_MAX = BOX_SIZE_AVG + BOX_SIZE_DIFF_MAX

print('Image box watchdog, running from {}'.format(SCRIPT_PATH), flush=True)

print('Loading model from "{}"...'.format(PATH_TO_MODEL_DIR), flush=True)
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)
print('Model loaded', flush=True)

def detect_bbox_from_image_path(image_path):
    print('Running inference for "{}"...'.format(image_path))

    image_ext = os.path.splitext(image_path)[1]
    if image_ext != '.jpg':
        print('Skip non JPEG file', flush=True)
        return

    # load image into numpy array
    image_np = np.array(Image.open(image_path))
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    # process results
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()
    img = Image.fromarray(image_np_with_detections)

    # crop and draw each detection box
    box_score_max = 0.0
    crop_img = img.copy()
    box_img = img.copy()
    draw = ImageDraw.Draw(box_img)
    for i, box in enumerate(detections['detection_boxes']):
        score = detections['detection_scores'][i]
        classe = detections['detection_classes'][i]
        if score > BOX_DRAW_THRESHOLD:
            print(i, box, classe, score)
            x1 = int(box[1] * img.size[0])
            y1 = int(box[0] * img.size[1])
            x2 = int(box[3] * img.size[0])
            y2 = int(box[2] * img.size[1])
            x_len = x2 - x1
            y_len = y2 - y1
            print('box {}x{} from ({}, {}) to ({}, {})'.format(x_len, y_len, x1, y1, x2, y2))
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0))

            if (classe == 1) and (score > box_score_max):
                if ( x_len > BOX_SIZE_MIN ) and \
                   ( x_len < BOX_SIZE_MAX ) and \
                   ( y_len > BOX_SIZE_MIN ) and \
                   ( y_len < BOX_SIZE_MAX ):
                    crop_img = img.crop((x1, y1, x2, y2))
                    box_score_max = score
                    print('Max score:', box_score_max)
                else:
	                print('Skip box (invalid size)')

    # for debugging only
    image_basename = os.path.basename(image_path)
    dest_path_raw  = "{}/raw.jpg".format(OUTPUT_PATH)
    dest_path_box  = "{}/box.jpg".format(OUTPUT_PATH)
    dest_path_crop = "{}/crop.jpg".format(OUTPUT_PATH)
    img.save(dest_path_raw)
    box_img.save(dest_path_box)
    crop_img.save(dest_path_crop)
    os.system('/home/pi/foodSegmentation/script/rpi/03_send.sh ' + dest_path_raw)
    os.system('/home/pi/foodSegmentation/script/rpi/03_send.sh ' + dest_path_box)
    os.system('/home/pi/foodSegmentation/script/rpi/03_send.sh ' + dest_path_crop)
    # end of debugging only

    if box_score_max == 0.0:
        print('Nothing found, skipping file upload...', flush=True)
        return

    # save output with timestamp
    dest_path_out_crop = "{}/{}".format(OUTPUT_PATH, image_basename)
    dest_path_out_txt  = "{}/{}.txt".format(OUTPUT_PATH, image_basename)
    crop_img.save(dest_path_out_crop)
    with open(dest_path_out_txt, 'w') as f:
        print(box_score_max, file=f)

    # send
    os.system('/home/pi/foodSegmentation/script/rpi/03_send.sh ' + dest_path_out_crop)
    os.system('/home/pi/foodSegmentation/script/rpi/03_send.sh ' + dest_path_out_txt)

    # delete files
    #os.remove(image_path)

    print('End of inference', flush=True)

class CustomEventHandler(FileSystemEventHandler):
#    def on_created(self, event):
#        path = event.src_path
#        print('File creation event from "{}"'.format(path))
#        detect_bbox_from_image_path(path)
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
