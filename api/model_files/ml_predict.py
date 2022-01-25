from PIL import Image
from tensorflow import cast
from tensorflow.image import resize
import numpy as np
from tensorflow.keras.models import load_model

reference = {0: "Crazing", 1: "Inclusion", 2: "Patches", 3: "Pitted", 4: "Rolled", 5:"Scratches"}

def preprocess_img(image):
    image = cast(image, tensorflow.float32)
    image = image / 255.
    image = resize(image, [256, 256])
    return image


def predict_defect(model,image):
    image = preprocess_img(image)
    result = np.argmax(model.predict(np.array([image])))
    return reference[result]

