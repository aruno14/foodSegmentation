import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

PATH_TO_MODEL_DIR = 'detection_model'
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)
BOX_DRAW_THRESHOLD = 0.6

def load_image_into_numpy_array(path):
   return np.array(Image.open(path))

for idx, image_path in enumerate(["data/imagesTest/20141029_215631_HDR.jpg", "data/images/table.jpg", "data/images/assiette-01.jpg", "data/images/choucroute-01.jpg", "data/images/repas-france-08.jpg"]):
    print('Running inference for {}... '.format(image_path))
    image_np = load_image_into_numpy_array(image_path)
    image_np = image_np[:,:,:3]
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()
    img = Image.fromarray(image_np_with_detections)

    draw = ImageDraw.Draw(img)
    for i, box in enumerate(detections['detection_boxes']):
        score = detections['detection_scores'][i]
        classe = detections['detection_classes'][i]
        if score > BOX_DRAW_THRESHOLD:
            print('\t', i, box, classe, score)
            x1 = box[1] * img.size[0]
            y1 = box[0] * img.size[1]
            x2 = box[3] * img.size[0]
            y2 = box[2] * img.size[1]
            print('\tbox', x1, y1, x2, y2)
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0))
    
    img.save("output-"+ str(idx) + "-box.jpg")
