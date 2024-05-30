import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

model = load_model("gen.h5")

# Define diffusion model parameters and training loop
class DiffusionModel:
    def __init__(self, model, timesteps=100):
        self.model = model
        self.timesteps = timesteps
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
    
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

def display_samples(samples):
    num_samples = len(samples)
    plt.figure(figsize=(15, 15))
    for i, sample in enumerate(samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(tf.squeeze(sample).numpy())
        plt.axis('off')
    plt.show()

def save_samples(samples, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, sample in enumerate(samples):
        sample_path = os.path.join(output_dir, f'sample_{i+1}.png')
        tf.keras.preprocessing.image.save_img(sample_path, tf.squeeze(sample).numpy())

diffusion = DiffusionModel(model)

samples = diffusion.sample(num_samples=5)
save_samples(samples)
