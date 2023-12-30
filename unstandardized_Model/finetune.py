import tensorflow as tf 
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop 

model = tf.keras.models.load_model('machine.keras')
print(model.summary())

BATCH_SIZE = 8
IMG_SIZE = (299, 299)
train_dir = "initialDataset"
validation_dir = "initialDataset/validation"
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
#!Might have to rethink how we do the fine tuning depending on how many layers there are
print("Number of layers in the base model: ", len(model.layers)) # --> ~320 or 8

# Fine-tune from this layer onwards
fine_tune_at = 220

# Freeze all the layers before the `fine_tune_at` layer
for layer in model.layers[:fine_tune_at]:
  layer.trainable = False

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy')])
path = 'data/testing'

model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
model.save('machineFineTune.keras')