import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# PATHS & SETTINGS
# -----------------------------
train_dir = "dataset/train"
val_dir   = "dataset/valid"
test_dir  = "dataset/test"


MODEL_PATH = "lung_cancer_resnet50_final.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_INITIAL = 40
EPOCHS_FINE = 20

# -----------------------------
# DATA GENERATORS
# -----------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_data.num_classes

# -----------------------------
# LOAD OR CREATE MODEL
# -----------------------------
if os.path.exists(MODEL_PATH):
    print("🔄 Loading existing trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

else:
    print("🆕 Creating new ResNet50 model...")

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

model.summary()

# -----------------------------
# INITIAL TRAINING
# -----------------------------
print("\n🚀 Initial Training Started...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_INITIAL
)

# -----------------------------
# FINE-TUNING
# -----------------------------
print("\n🔧 Fine-Tuning Started...\n")

for layer in model.layers[-80:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINE
)

# -----------------------------
# MODEL EVALUATION
# -----------------------------
test_loss, test_acc = model.evaluate(test_data)
print("\n✅ Test Accuracy:", test_acc)

# -----------------------------
# CONFUSION MATRIX & REPORT
# -----------------------------
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = test_data.classes
class_names = list(test_data.class_indices.keys())

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\n📊 Classification Report\n")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save(MODEL_PATH)
print("\n💾 Model saved successfully")

# -----------------------------
# TRAINING CURVES
# -----------------------------
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(acc, label="Train Accuracy")
plt.plot(val_acc, label="Val Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.legend()
plt.title("Loss")

plt.show()
