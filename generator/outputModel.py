import tensorflow as tf
from tensorflow.keras import layers, Model, config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

config.set_dtype_policy("mixed_float16")


#!TODO: delete mn500, n500, m500 as they are all synthetic images created via GAN. Use half of HAM 10000 dataset as training

# 1. Setup Environment

# Define image size and batch size
image_size = (224, 224)
IMG_SIZE = 224
batch_size = 8
timesteps = 64 #steps from noisy image to clear
time_bar = 1 - np.linspace(0, 1.0, timesteps + 1) # linspace for timesteps

# 2. Define the Model

def block(x_img, x_ts):
    x_parameter = layers.Conv2D(128, kernel_size=3, padding='same')(x_img)
    x_parameter = layers.Activation('relu')(x_parameter)

    time_parameter = layers.Dense(128)(x_ts)
    time_parameter = layers.Activation('relu')(time_parameter)
    time_parameter = layers.Reshape((1, 1, 128))(time_parameter)
    x_parameter = x_parameter * time_parameter
    
    # -----
    x_out = layers.Conv2D(128, kernel_size=3, padding='same')(x_img)
    x_out = x_out + x_parameter
    x_out = layers.LayerNormalization()(x_out)
    x_out = layers.Activation('relu')(x_out)
    
    return x_out
def build_unet():
    x = x_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='x_input')
    
    x_ts = x_ts_input = layers.Input(shape=(1,), name='x_ts_input')
    x_ts = layers.Dense(192)(x_ts)
    x_ts = layers.LayerNormalization()(x_ts)
    x_ts = layers.Activation('relu')(x_ts)
    
    # ----- left ( down ) -----
    x = x32 = block(x, x_ts)
    x = layers.MaxPool2D(2)(x)
    
    x = x16 = block(x, x_ts)
    x = layers.MaxPool2D(2)(x)
    
    x = x8 = block(x, x_ts)
    x = layers.MaxPool2D(2)(x)
    
    x = x4 = block(x, x_ts)
    
    # ----- MLP -----
    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, x_ts])
    x = layers.Dense(128)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(4 * 4 * 32)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Reshape((4, 4, 32))(x)
    
    # ----- right ( up ) -----
    x_upsampled = layers.UpSampling2D(size=(7, 7))(x)  # (None, 28, 28, 32)

    x = layers.Concatenate()([x_upsampled, x4])
    x = block(x, x_ts)
    x = layers.UpSampling2D(2)(x)
    
    x = layers.Concatenate()([x, x8])
    x = block(x, x_ts)
    x = layers.UpSampling2D(2)(x)
    
    x = layers.Concatenate()([x, x16])
    x = block(x, x_ts)
    x = layers.UpSampling2D(2)(x)
    
    x = layers.Concatenate()([x, x32])
    x = block(x, x_ts)
    
    # ----- output -----
    x = layers.Conv2D(3, kernel_size=1, padding='same')(x)
    model = tf.keras.models.Model([x_input, x_ts_input], x)
    return model


# Instantiate UNet model

unet = build_unet()

# Define diffusion model parameters and training loop
class DiffusionModel:
    def __init__(self, model, timesteps=64):
        self.model = model
        self.timesteps = timesteps
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def forward_noise(self, x, t):
        a = time_bar[t]      # base on t
        b = time_bar[t + 1]  # image for t + 1
        #Creates random timesteps ^^

        noise = np.random.normal(size=x.shape)  # noise mask
        a = a.reshape((-1, 1, 1, 1))
        b = b.reshape((-1, 1, 1, 1))
        img_a = x * (1 - a) + noise * a
        img_b = x * (1 - b) + noise * b
        return img_a, img_b
    
    def generate_ts(self, num):
        return np.random.randint(0, timesteps, size=num)
             
    def sample(self, num_samples=1):
        samples = []
        for _ in range(num_samples):
            x = tf.random.normal((1, 224, 224, 3))
            for t in range(self.timesteps):
                t_batch = tf.ones((1, 1)) * t
                predicted_noise = self.model(x, training=False)
                x = x - predicted_noise / (self.timesteps ** 0.5)
            samples.append(x)
        return samples
    
checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt.weights.h5")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= checkpoint_prefix,
    save_weights_only=True
)

# Load latest checkpoint if it exists
items = os.listdir(checkpoint_dir)
if items:
    print("\n\nRunning on latest checkpoint\n\n")
    unet.load_weights(f"{checkpoint_dir}/{items[0]}")

diffusion_model = DiffusionModel(unet)

diffusion_model.sample(num_samples=5)