import tensorflow as tf 
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers
#from tensorflow.keras.models import saving_lib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop 
from keras.callbacks import CSVLogger

"""Split: 
Benign: 37574
Malignant: 6857
"""
@tf.keras.saving.register_keras_serializable(name="weighted_binary_crossentropy")
def weighted_binary_crossentropy(target, output, weights):
  target = tf.convert_to_tensor(target)
  output = tf.convert_to_tensor(output)
  weights = tf.convert_to_tensor(weights, dtype=target.dtype)

  epsilon_ = tf.constant(tf.keras.backend.epsilon(), output.dtype.base_dtype)
  output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

  # Compute cross entropy from probabilities.
  bce = weights[1] * target * tf.math.log(output + epsilon_)
  bce += weights[0] * (1 - target) * tf.math.log(1 - output + epsilon_)
  return -bce

# @tf.keras.saving.register_keras_serializable(name="WeightedBinaryCrossentropy")
# class WeightedBinaryCrossentropy:
#     def __init__(
#         self,
#         label_smoothing=0.0,
#         weights = [1.0, 1.0],
#         axis=-1,
#         name="weighted_binary_crossentropy",
#         fn = None,
#     ):
#         """Initializes `WeightedBinaryCrossentropy` instance.
#         Args:
#           from_logits: Whether to interpret `y_pred` as a tensor of
#             [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
#             assume that `y_pred` contains probabilities (i.e., values in [0,
#             1]).
#           label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When >
#             0, we compute the loss between the predicted labels and a smoothed
#             version of the true labels, where the smoothing squeezes the labels
#             towards 0.5.  Larger values of `label_smoothing` correspond to
#             heavier smoothing.
#           axis: The axis along which to compute crossentropy (the features
#             axis).  Defaults to -1.
#           name: Name for the op. Defaults to 'weighted_binary_crossentropy'.
#         """
#         super().__init__()
#         self.weights = weights # tf.convert_to_tensor(weights)
#         self.label_smoothing = label_smoothing
#         self.name = name
#         self.fn = weighted_binary_crossentropy if fn is None else fn

#     def __call__(self, y_true, y_pred):
#         y_pred = tf.convert_to_tensor(y_pred)
#         y_true = tf.cast(y_true, y_pred.dtype)
#         self.label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype)

#         def _smooth_labels():
#             return y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

#         y_true = tf.__internal__.smart_cond.smart_cond(self.label_smoothing, _smooth_labels, lambda: y_true)

#         return tf.reduce_mean(self.fn(y_true, y_pred, self.weights),axis=-1)
    
#     def get_config(self):
#         config = {"name": self.name, "weights": self.weights, "fn": self.fn}

#         # base_config = super().get_config()
#         return dict(list(config.items()))

#     @classmethod
#     def from_config(cls, config):
#         """Instantiates a `Loss` from its config (output of `get_config()`).
#         Args:
#             config: Output of `get_config()`.
#         """
#         if saving_lib.saving_v3_enabled():
#             fn_name = config.pop("fn", None)
#             if fn_name:
#                 config["fn"] = get(fn_name)
#         return cls(**config)

tf.keras.mixed_precision.set_global_policy('mixed_float16')

print("Num GPUs Available: ", tf.config.list_physical_devices())

configproto = tf.compat.v1.ConfigProto() 
configproto.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=configproto) 
tf.compat.v1.keras.backend.set_session(sess)

BATCH_SIZE = 4
IMG_SIZE = (224, 224)
preprocess_input = tf.keras.applications.vgg16.preprocess_input

train_dir = "combinedDataset/train"
validation_dir = "combinedDataset/test"
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

oldModel = tf.keras.models.load_model('../unstandardized_Model/machineFineTune.h5')

base_model = tf.keras.models.load_model('../unstandardized_Model/fineTunedBase.h5')
base_model.trainable = False
print(len(base_model.layers))
print(base_model.summary())
# # Fine-tune from this layer onwards
# fine_tune_at = 16
# # Freeze all the layers before the `fine_tune_at` layer
# for layer in base_model.layers[:fine_tune_at]:
#   layer.trainable = False

#https://www.tensorflow.org/tutorials/images/transfer_learning
image_batch, label_batch = next(iter(train_dataset)) #Utilizes keras api to get the images and labels separated
feature_batch = base_model(image_batch) # Gets the features (inputs of the layers) of the model from passing in the images
global_average_layer = tf.keras.layers.GlobalAveragePooling2D() 
#prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid") #Fully connected layer, getting prediction
# oldPrediction = tf.keras.models.Sequential(oldModel.layers[len(oldModel.layers) - 1]) #Previous classification head
# oldPrediction.trainable = False #Just training new classification head
prediction_layer = tf.keras.layers.Dense(1, bias_initializer= [np.log(37574/6857)]) #Fully connected layer, getting new prediction of benign or malignant


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

# wbce = WeightedBinaryCrossentropy(weights = [6.48, 1.18])
# wbce(inputs,outputs)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

#weighted_binary_crossentropy(inputs, outputs, weights=[6.48, 1.18]) loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy'), tf.keras.metrics.AUC(name="AUC")])


model.fit(train_dataset, epochs=5, validation_data=validation_dataset, callbacks=[callback])
model.save('machine.h5')
