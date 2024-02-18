import tensorflow as tf 
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers
#from tensorflow.keras.models import saving_lib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop 
from keras.callbacks import CSVLogger

#TODO: Downsample benign class by x%
"""Split: 
Benign: 37574
Malignant: 6857
"""
totalData = 37574
benign = 37574
malignant = 6857

# @tf.keras.saving.register_keras_serializable(name="weighted_binary_crossentropy")
# def weighted_binary_crossentropy(target, output, weights):
#   target = tf.convert_to_tensor(target)
#   output = tf.convert_to_tensor(output)
#   weights = tf.convert_to_tensor(weights, dtype=target.dtype)

#   epsilon_ = tf.constant(tf.keras.backend.epsilon(), output.dtype.base_dtype)
#   output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

#   # Compute cross entropy from probabilities.
#   bce = weights[1] * target * tf.math.log(output + epsilon_)
#   bce += weights[0] * (1 - target) * tf.math.log(1 - output + epsilon_)
#   return -bce



tf.keras.mixed_precision.set_global_policy('mixed_float16')

print("Num GPUs Available: ", tf.config.list_physical_devices())

configproto = tf.compat.v1.ConfigProto() 
configproto.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=configproto) 
tf.compat.v1.keras.backend.set_session(sess)

BATCH_SIZE = 4
IMG_SIZE = (224, 224)
preprocess_input = tf.keras.applications.vgg16.preprocess_input

train_dir = "combinedDataset/Newtrain"
validation_dir = "combinedDataset/test"
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

oldModel = tf.keras.models.load_model('../unstandardized_Model/machineFineTune.h5')

base_model = tf.keras.models.load_model('../unstandardized_Model/fineTunedBase.h5')
base_model.trainable = False
print(len(base_model.layers))
print(base_model.summary())


#https://www.tensorflow.org/tutorials/images/transfer_learning
image_batch, label_batch = next(iter(train_dataset)) #Utilizes keras api to get the images and labels separated
feature_batch = base_model(image_batch) # Gets the features (inputs of the layers) of the model from passing in the images
global_average_layer = tf.keras.layers.GlobalAveragePooling2D() 

prediction_layer = tf.keras.layers.Dense(1, bias_initializer=tf.constant_initializer(np.log(37574/6857))) #Fully connected layer, getting new prediction of benign or malignant


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
outputs = prediction_layer(x) #firstOutput
outputs = tf.keras.layers.Activation('linear', dtype='float32')(outputs) #identitiy function to increase computing w/ mixedfloat16

model = tf.keras.Model(inputs, outputs)
print(model.summary())
base_learning_rate = 0.00001

# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / benign) * (totalData / 2.0) #Zero class should be ordered alphabetically
weight_for_1 = (1 / malignant) * (totalData / 2.0)

weights = {0: weight_for_0, 1: weight_for_1}

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

#weighted_binary_crossentropy(inputs, outputs, weights=[6.48, 1.18]) loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), 
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), 
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy'), 
             tf.keras.metrics.AUC(name="AUC")],
    class_weight = weights)


model.fit(train_dataset, epochs=5, validation_data=validation_dataset, callbacks=[callback])
model.save('machine.h5')
