import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

PATH_TO_MODEL_DIR = 'detection_model'
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)

model_name = "traySegModel"
model = tf.keras.models.load_model(model_name)

BOX_DRAW_THRESHOLD = 0.5
BOX_SIZE_MIN = 0.1
BOX_SIZE_MAX = 0.8
BOX_MARGIN = 10
SIZE = (128, 128)

def applyMask(image):
    predictions = model.predict(np.array([image]))
    predicted_image = predictions[0]

    mask = np.squeeze(predicted_image)
    mask = np.where((mask > 0.8), 1, 0)

    mask = ndimage.binary_opening(mask, iterations=5)
    mask = ndimage.binary_closing(mask, iterations=5)
    mask = ndimage.binary_dilation(mask, iterations=5)

    #MaxPooling
    mask*(mask == ndimage.filters.maximum_filter(mask, footprint=np.ones((3,3))))
    mask = np.where((mask > 0.8), 1.0, 0.5)

    output_image = image * np.expand_dims(mask, axis=-1)
    return output_image


def load_image_into_numpy_array(path):
   return np.array(Image.open(path))

for idx, image_path in enumerate(["example/input-01.png", "data/imagesTest/20141029_215631_HDR.jpg", "data/images/table.jpg", "data/images/assiette-01.jpg", "data/images/choucroute-01.jpg", "data/images/repas-france-08.jpg"]):
    print('Running inference for {}... '.format(image_path), end='')
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

    imgBox = img.copy()
    draw = ImageDraw.Draw(imgBox)
    for i, box in enumerate(detections['detection_boxes']):
        score = detections['detection_scores'][i]
        classe = detections['detection_classes'][i]
        if score > BOX_DRAW_THRESHOLD:
            print(i, box, classe, score)
            x1 = box[1] * img.size[0]
            y1 = box[0] * img.size[1]
            x2 = box[3]* img.size[0]
            y2 = box[2]* img.size[1]
            print('box', x1, y1, x2, y2)
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0))
            
            width = box[3] - box[1]
            height = box[2] - box[0]
            if not (width < BOX_SIZE_MAX and height < BOX_SIZE_MAX and width > BOX_SIZE_MIN and height > BOX_SIZE_MIN):
                print('Skip box (invalid size)', width, height)
                continue
            crop_img = img.crop((max(x1-BOX_MARGIN, 0), max(y1-BOX_MARGIN, 0), min(x2+BOX_MARGIN, x2*img.size[0]), min(y2+BOX_MARGIN, y2*img.size[1])))
            crop_img_resized = crop_img.resize(SIZE)
            masked_img_np = applyMask(np.array(crop_img_resized))
            masked_img = Image.fromarray(masked_img_np.astype('uint8'))
            masked_img.save("output-object-{}-{}.jpg".format(idx, i))
    imgBox.save("output-{}-box.jpg".format(idx))

