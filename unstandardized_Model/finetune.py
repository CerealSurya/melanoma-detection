import tensorflow as tf 
import keras
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop 
#For the second transfer we reconstruct this same model using the trainedInception.keras as the base training only from like 270 onwards
# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
#oldModel = tf.keras.models.load_model('machine.keras') 
tf.keras.mixed_precision.set_global_policy('mixed_float16')

print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
BATCH_SIZE = 1
IMG_SIZE = (224, 224)
preprocess_input = tf.keras.applications.vgg16.preprocess_input

train_dir = "initialDataset/train"
validation_dir = "initialDataset/validation"
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

# with strategy.scope(): #Makes things work with multiple gpus
oldModel = tf.keras.models.load_model('machineVGG.h5')

base_model = tf.keras.applications.vgg16.VGG16(input_shape= IMG_SIZE + (3,), include_top=False, weights='imagenet')
print(len(base_model.layers))
# Fine-tune from this layer onwards
fine_tune_at = 17
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

#https://www.tensorflow.org/tutorials/images/transfer_learning
image_batch, label_batch = next(iter(train_dataset)) #Utilizes keras api to get the images and labels separated
feature_batch = base_model(image_batch) # Gets the features (inputs of the layers) of the model from passing in the images
global_average_layer = tf.keras.layers.GlobalAveragePooling2D() 
feature_batch_average = global_average_layer(feature_batch) #Pools the features together to simplify them to get an output. Helps simplify network size and computing. Precursor to fully connected layer
#prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid") #Fully connected layer, getting prediction
prediction_layer = tf.keras.models.Sequential(oldModel.layers[len(oldModel.layers) - 1])


data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

inputs = tf.keras.Input(shape=(224, 224, 3))
#Our defined model with the steps in sequential order
x = data_augmentation(inputs) #Augment all of the inputs
x = preprocess_input(x) #Process all of the inputs. Manipulating their size and color values to fit requirements of the model

x = base_model(x, training=False) #Pass in the inputs without manipulating weight values, running BatchNormalization layers in inference mode
x = global_average_layer(x) #Precursor to fully connected layer (Pools everything together)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x) #Get our outputs from the fully connected layer we defined above
outputs = tf.keras.layers.Activation('linear', dtype='float32')(outputs)

model = tf.keras.Model(inputs, outputs)
print(model.summary())
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy'), tf.keras.metrics.AUC(name="AUC")])


model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
base_model.save('fineTunedBase.h5')
model.save('machineFineTune.h5')