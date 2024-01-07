import tensorflow as tf 
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop 
import sys

"""" Use this for importing singular img
from PIL import Image
img_data = np.random.random(size=(299, 299, 3))
img = tf.keras.utils.array_to_img(img_data)
array = tf.keras.utils.image.img_to_array(img)
model.predict(array)
"""
kerasFile = sys.argv[1]
print(f'Running evaluation on file {kerasFile}')
#TODO: Load keras model, load validation dataset(dermnet) --> reformat it all to fit inception, run evaluate on everything, log accuracy
model = tf.keras.models.load_model(kerasFile)

print(model.summary())
preprocess_input = tf.keras.applications.inception_v3.preprocess_input
validation_dataset = tf.keras.utils.image_dataset_from_directory("initialDataset/validation", shuffle=False, batch_size=8, image_size=(299, 299))

labels = np.concatenate([labels for imgs, labels in validation_dataset], axis=0)
# for images, labels in validation_dataset.take(1):  # only take first element of dataset
#     newLabel = labels.numpy().tolist()
#     for l in newLabel:
#         print(l)
#         labelsList.append(l)
    
#     processedImg.append(preprocess_input(images.numpy()))

#processedImg = np.array(processedImg) #Convert python list into numpy array

predictions = model.predict(validation_dataset)
print("Total Predictions: ", len(predictions))

fpr_keras, tpr_keras, thresholds_keras = roc_curve(labels, predictions)
auc_keras = auc(fpr_keras, tpr_keras)
print(fpr_keras, "\nbreak\n")
print(tpr_keras, "\nbreak\n")
print(thresholds_keras, "\nbreak\n")
print("AUC: ", auc_keras, "\nbreak\n")

with open("fpr.txt", "w") as f:
    for i in fpr_keras:
        f.write(str(i) + "\n")
with open("tpr.txt", "w") as f:
    for i in tpr_keras:
        f.write(str(i) + "\n")
with open("thresholds.txt", "w") as f:
    for i in thresholds_keras:
        f.write(str(i) + "\n")      