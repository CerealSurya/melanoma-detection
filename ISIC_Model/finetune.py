import tensorflow as tf 
import numpy as np
import os 
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop 
from keras.callbacks import CSVLogger

"""Split: 
Benign: 37574
Malignant: 6857
"""
totalData = 44375
benign = 37572
malignant = 6803


tf.keras.mixed_precision.set_global_policy('mixed_float16')

print("Num GPUs Available: ", tf.config.list_physical_devices())

configproto = tf.compat.v1.ConfigProto() 
configproto.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=configproto) 
tf.compat.v1.keras.backend.set_session(sess)

BATCH_SIZE = 2 #Might need to reduce to run the training

IMG_SIZE = (224, 224)
preprocess_input = tf.keras.applications.vgg16.preprocess_input

train_dir = "combinedDataset/Newtrain"
validation_dir = "combinedDataset/test"
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

firstModel = tf.keras.models.load_model('machine.h5')
#print(firstModel.summary())
#oldModel = tf.keras.models.load_model('../unstandardized_Model/machineFineTune.h5')
#print(oldModel.summary())
base_model = tf.keras.models.load_model('../unstandardized_Model/fineTunedBase.h5')


# Fine-tune from this layer onwards
fine_tune_at = 17
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

#https://www.tensorflow.org/tutorials/images/transfer_learning
image_batch, label_batch = next(iter(train_dataset)) #Utilizes keras api to get the images and labels separated
feature_batch = base_model(image_batch) # Gets the features (inputs of the layers) of the model from passing in the images
global_average_layer = tf.keras.layers.GlobalAveragePooling2D() 
#prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid") #Fully connected layer, getting prediction
# oldPrediction = tf.keras.models.Sequential(oldModel.layers[len(oldModel.layers) - 2]) #Previous classification head
# oldPrediction.trainable = True #Just training new classification head
prediction_layer = tf.keras.models.Sequential(firstModel.layers[len(firstModel.layers) - 2]) #Fully connected layer, getting new prediction of benign or malignant
print(len(prediction_layer.layers))

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

inputs = tf.keras.Input(shape=(224, 224, 3))
#Our defined model with the steps in sequential order
x = data_augmentation(inputs) #Augment all of the inputs
x = preprocess_input(x) #Process all of the inputs. Manipulating their size and color values to fit requirements of the model

#!Do we need the training=False??
x = base_model(x, training=False) #Pass in the inputs without manipulating weight values, running BatchNormalization layers in inference mode
x = global_average_layer(x) #Precursor to fully connected layer (Pools everything together)
x = tf.keras.layers.Dropout(0.4)(x)
#firstOutput = oldPrediction(x) #Get our outputs from the fully connected layer we defined above
outputs = prediction_layer(x)
#print(outputs.summary())
outputs = tf.keras.layers.Activation('linear', dtype='float32')(outputs) #identitiy function to increase computing w/ mixedfloat16

model = tf.keras.Model(inputs, outputs)
print(model.summary())
base_learning_rate = 0.00001
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

weight_for_0 = (1 / benign) * (totalData / 2.0)
weight_for_1 = (1 / malignant) * (totalData / 2.0)

weights = {0: weight_for_0, 1: weight_for_1}

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), 
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), 
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy'), 
             tf.keras.metrics.AUC(name="AUC")]
    )

model.fit(train_dataset, epochs=20, validation_data=validation_dataset, callbacks=[callback])
model.save('machineFineTune.h5')
