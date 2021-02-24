import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

PATH_TO_MODEL_DIR = 'detection_model'
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)

def load_image_into_numpy_array(path):
   return np.array(Image.open(path))

for idx, image_path in enumerate(["191208.jpg", "191670.jpg", "191671.jpg", "data/imagesTest/20141029_215631_HDR.jpg", "S__27983895.jpg", "data/images/table.jpg", "data/images/assiette-01.jpg", "data/images/choucroute-01.jpg", "data/images/repas-france-08.jpg"]):
    print('Running inference for {}... '.format(image_path), end='')
    image_np = load_image_into_numpy_array(image_path)
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
        if score > 0.6:
            print(i, box, classe, score)
            x1 = box[1] * img.size[0]
            y1 = box[0] * img.size[1]
            x2 = box[3]* img.size[0]
            y2 = box[2]* img.size[1]
            print('box', x1, y1, x2, y2)
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0))
    img.save(str(idx) + "-output.jpg")
