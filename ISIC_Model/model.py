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
Benign: 37,540
Malignant: 5,180
"""


