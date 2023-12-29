import tensorflow as tf 
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop 

#TODO: Load keras model, load validation dataset(dermnet) --> reformat it all to fit inception, run evaluate on everything, log accuracy
model = tf.keras.models.load_model('machine.keras')
print(model.summary())
preprocess_input = tf.keras.applications.inception_v3.preprocess_input
validation_dataset = tf.keras.utils.image_dataset_from_directory("initialDataset/validation", shuffle=False, batch_size=None, image_size=(299, 299))

processedImg = []
for images, labels in validation_dataset.take(1):  # only take first element of dataset
    print(len(images), images)
    processedImg.append(preprocess_input(images.numpy()))
    print("Howdy", preprocess_input(images.numpy()))

#processedImg = np.array(processedImg) #Convert python list into numpy array

predictions = model.predict(processedImg)

for yPrediction in predictions:
    print(yPrediction)

fpr_keras, tpr_keras, thresholds_keras = roc_curve([0,1], predictions)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()