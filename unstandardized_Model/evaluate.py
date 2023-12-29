import tensorflow as tf 
import numpy as np
import os 
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop 
from keras.callbacks import CSVLogger

#TODO: Load keras model, load validation dataset(dermnet) --> reformat it all to fit inception, run evaluate on everything, log accuracy
model = tf.keras.models.load_model('machine.keras')
print(model.summary)