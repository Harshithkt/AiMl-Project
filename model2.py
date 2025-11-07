# Cancerous vs Non-Cancerous Cell Classification with K-Fold and Prediction

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')  # Avoid Tkinter issues on Windows
import matplotlib.pyplot as plt


# 1. Dataset Path & Collect Images

dataset_path = "dataset"
classes = ['non_cancerous', 'cancerous']

file_paths = []
labels = []

for idx, cls in enumerate(classes):
    cls_folder = os.path.join(dataset_path, cls)
    for img_name in os.listdir(cls_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_paths.append(os.path.join(cls_folder, img_name))
            labels.append(idx)

file_paths = np.array(file_paths)
labels = np.array(labels)

print("Total images:", len(file_paths))


# 2. K-Fold Setup

k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

fold_no = 1
acc_per_fold = []

for train_index, val_index in skf.split(file_paths, labels):
    print(f"\n----- Fold {fold_no} -----")
    
    train_files, val_files = file_paths[train_index], file_paths[val_index]
    train_labels, val_labels = labels[train_index], labels[val_index]
    

    # 3. Data Generators

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                                       zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': train_files, 'class': train_labels.astype(str)}),
    x_col='filename',
    y_col='class',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=True
    )

    val_generator = val_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': val_files, 'class': val_labels.astype(str)}),
    x_col='filename',
    y_col='class',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
    )


    # 4. Build Model
   
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
    

    # 5. Train Model

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=44,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator)
    )
    
    # Save fold accuracy
    scores = model.evaluate(val_generator, verbose=0)
    print(f"Fold {fold_no} - Validation Accuracy: {scores[1]*100:.2f}%")
    acc_per_fold.append(scores[1]*100)
    
    fold_no += 1


# 6. K-Fold Results

print("\n===== K-Fold Results =====")
for i, acc in enumerate(acc_per_fold):
    print(f"Fold {i+1}: {acc:.2f}%")
print(f"Average Accuracy: {np.mean(acc_per_fold):.2f}%")

# ------------------------------
# 7. Predict a Single Image
# ------------------------------
def predict_image(img_path):
    img_path = os.path.normpath(img_path)
    img = load_img(img_path, target_size=(224,224))
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)[0][0]
    if pred > 0.5:
        print("Cancerous")
    else:
        print("Non-cancerous")

# ------------------------------
# Example usage
# ------------------------------
predict_image(r"dataset\non_cancerous\5.jpg")
predict_image(r"dataset\cancerous\image_edo_0_1552.jpg")

model.save("model2/cell_classifier.h5")