import cv2
import numpy as np
import tensorflow as tf
from scipy import ndimage

video = cv2.VideoCapture('example/assiettes.mp4')
SIZE = (128, 128)

model_name = "traySegModel"
model = tf.keras.models.load_model(model_name)

int = cv2.VideoWriter('inpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, SIZE)
maskt = cv2.VideoWriter('maskpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, SIZE)
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, SIZE)

while True:
    grabbed, frame = video.read()
    if grabbed != True:
        break
    #print("====New frame====")
    frame = cv2.resize(frame, SIZE)
    int.write(frame)
    frame = frame.astype(np.float32)
    frame/=255

    predictions = model.predict(np.asarray([frame]))
    predicted_image = predictions[0]

    predicted_image = (predicted_image*255).astype('uint8')
    predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_GRAY2RGB)
    maskt.write(predicted_image)

    mask = np.squeeze(predictions[0])
    mask = np.where((mask > 0.8), 1, 0)

    mask = ndimage.binary_opening(mask, iterations=5)
    mask = ndimage.binary_closing(mask, iterations=5)
    mask = ndimage.binary_dilation(mask, iterations=5)

    #MaxPooling
    mask*(mask == ndimage.filters.maximum_filter(mask, footprint=np.ones((3,3))))
    mask = np.where((mask > 0.8), 1.0, 0.5)

    output_image = frame * np.expand_dims(mask, axis=-1)

    output_image_uint8 = (output_image*255).astype('uint8')
    out.write(output_image_uint8)

video.release()
int.release()
out.release()
maskt.release()
cv2.destroyAllWindows()
