import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

MODEL_PATH = r"E:\FINAL YEAR\lung cancer1\lung_cancer_resnet50_finetuned.h5"

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

img_path = r"E:\FINAL YEAR\lung cancer1\sample_test_1.png"

img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"🫁 Prediction: {predicted_class}")
print(f"📈 Confidence: {confidence:.2f}%")
