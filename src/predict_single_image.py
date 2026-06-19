import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

MODEL_PATH = "lung_cancer_resnet50_final.h5"
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

img_path = "sample_ct.jpg"   # put one CT image here

img = image.load_img(img_path, target_size=IMG_SIZE)
img = image.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
class_index = np.argmax(pred)

class_names = ['Benign', 'Malignant', 'Normal']  # adjust to your dataset

print("Prediction:", class_names[class_index])
