import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 10

def build_feature_extractor():
    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu')
    ])
    return model

#this is the siamese network thing
def build_siamese():
    base_model = build_feature_extractor()
    input_a = layers.Input(shape=(*IMG_SIZE, 3))
    input_b = layers.Input(shape=(*IMG_SIZE, 3))

    feature_a = base_model(input_a)
    feature_b = base_model(input_b)

    # calculatee L1 distance
    distance = tf.abs(feature_a - feature_b)
    output = layers.Dense(1, activation='sigmoid')(distance)

    model = keras.Model(inputs=[input_a, input_b], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_data(clean_dir='data/clean', dirty_dir='data/dirty'):
    datagen = ImageDataGenerator(rescale=1./255)
    clean_gen = datagen.flow_from_directory(clean_dir, target_size=IMG_SIZE, class_mode=None, batch_size=BATCH_SIZE)
    dirty_gen = datagen.flow_from_directory(dirty_dir, target_size=IMG_SIZE, class_mode=None, batch_size=BATCH_SIZE)

    clean_imgs = next(clean_gen)
    dirty_imgs = next(dirty_gen)

    pairs_a = np.concatenate([clean_imgs, clean_imgs])
    pairs_b = np.concatenate([clean_imgs, dirty_imgs])
    labels = np.concatenate([np.zeros(len(clean_imgs)), np.ones(len(clean_imgs))])
    return (pairs_a, pairs_b, labels)

#trainingg
pairs_a, pairs_b, labels = load_data()
model = build_siamese()
model.summary()

model.fit([pairs_a, pairs_b], labels, epochs=EPOCHS, batch_size=BATCH_SIZE)
model.save("siamese_model.h5")

#convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("siamese_clean_detector.tflite", "wb").write(tflite_model)

print("model trained and saved as siamese_clean_detector.tflite")
