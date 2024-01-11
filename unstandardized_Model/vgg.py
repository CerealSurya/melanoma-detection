import tensorflow as tf 
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop 
from keras.callbacks import CSVLogger

BATCH_SIZE = 2
IMG_SIZE = (224, 224)
train_dir = "initialDataset/train"
validation_dir = "initialDataset/validation"
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
preprocess_input = tf.keras.applications.vgg16.preprocess_input


base_model = tf.keras.applications.vgg16.VGG16(input_shape= IMG_SIZE + (3,), include_top=False, weights='imagenet')
print(len(base_model.layers))
print(base_model.summary())
base_model.trainable = False

image_batch, label_batch = next(iter(train_dataset)) #Utilizes keras api to get the images and labels separated
feature_batch = base_model(image_batch) # Gets the features (inputs of the layers) of the model from passing in the images
global_average_layer = tf.keras.layers.GlobalAveragePooling2D() 
feature_batch_average = global_average_layer(feature_batch) #Pools the features together to simplify them to get an output. Helps simplify network size and computing. Precursor to fully connected layer
prediction_layer = tf.keras.layers.Dense(1) #Fully connected layer, getting prediction
prediction_batch = prediction_layer(feature_batch_average) #Applying it to the features


data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

inputs = tf.keras.Input(shape=(224, 224, 3))
#Our defined model with the steps in sequential order
x = data_augmentation(inputs) #Augment all of the inputs
x = preprocess_input(x) #Process all of the inputs. Manipulating their size and color values to fit requirements of the model

x = base_model(x, training=False) #Pass in the inputs without manipulating weight values
x = global_average_layer(x) #Precursor to fully connected layer (Pools everything together)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x) #Get our outputs from the fully connected layer we defined above

model = tf.keras.Model(inputs, outputs)
print(model.summary())
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy'), tf.keras.metrics.AUC(name="AUC")])

csv_logger = CSVLogger('log.csv', append=True, separator=';')
model.fit(train_dataset, epochs=8, callbacks=[csv_logger], validation_data=validation_dataset, shuffle=True)

model.save('machineVGG.h5')
#loaded_model1 = tf.keras.models.load_model('machineVGG.h5')