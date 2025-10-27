# Cancerous vs Non-Cancerous Cell Classification using Transfer Learning

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')  # Avoid Tkinter issues on Windows
import matplotlib.pyplot as plt


# 1. Dataset Path

dataset_path = "dataset"  # folder containing 'cancerous' & 'non_cancerous'


# 2. Data Preparation

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.4
)

# Force class order: 0 = non_cancerous, 1 = cancerous
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=16,
    class_mode='binary',
    classes=['non_cancerous', 'cancerous'],  # important to fix label mapping
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=16,
    class_mode='binary',
    classes=['non_cancerous', 'cancerous'],
    subset='validation'
)

print("Class indices:", train_generator.class_indices)  # check mapping


# 3. Build Model (Transfer Learning)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


# 4. Train the Model

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=100,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)


# 5. Plot Accuracy

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.savefig("training_accuracy.png")  # saves plot as PNG


# 6. Predict a Single Image

def predict_image(img_path):
    # Handle path safely
    img_path = os.path.normpath(img_path)
    img = load_img(img_path, target_size=(224,224))
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)[0][0]
    if pred > 0.5:
        print("Cancerous")
    else:
        print("Non-cancerous")


# Example Usage

predict_image(r"dataset\non_cancerous\image_Non_Cancerous_0_3421.jpg")
predict_image(r"dataset\cancerous\image_edo_0_1511.jpg")
