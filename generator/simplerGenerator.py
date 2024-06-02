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
batch_size = 4

# Create data generator for loading images
def preprocess_image(image):
    image = tf.image.resize(image, image_size)  # Resize the image
    image = image / 255.0  # Scale the image pixels to range [0, 1]
    return image

# Function to load images from a directory
def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return preprocess_image(image)

data_path = 'dataset/HAM10000_images_part_1'
file_pattern = data_path + '/*.jpg'  # Adjust the pattern based on your image format
file_paths = tf.data.Dataset.list_files(file_pattern)

# Apply the loading and preprocessing functions
dataset = file_paths.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Batch the dataset
dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# 2. Define the Model

def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder
    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = layers.UpSampling2D((2, 2))(c8)
    u9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs, outputs)
    return model

# Instantiate UNet model
input_shape = (224, 224, 3)
unet = build_unet(input_shape)

# Define diffusion model parameters and training loop
class DiffusionModel:
    def __init__(self, model, timesteps=100):
        self.model = model
        self.timesteps = timesteps
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
    
    def train_step(self, x):
        with tf.GradientTape() as tape:
            noise = tf.random.normal(shape=tf.shape(x))
            noisy_images = x + noise
            reconstructed_images = self.model(noisy_images, training=True)
            loss = self.loss_fn(x, reconstructed_images)
            #tape.watch(reconstructed_images)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        if np.all(gradients == 0):
            print("Warning: All gradients are zero.")

        return loss

    def train(self, dataloader, epochs, checkpoint_callback):
        checkpoint_callback.set_model(self.model)  # Attach the model to the checkpoint callback
        for epoch in range(epochs):
            for batch in dataloader:
                loss = self.train_step(batch)
                print(f"Epoch: {epoch+1}, Loss: {loss.numpy()}")

            with open("onEpoch.txt", "r") as f:
                l = f.read()
                with open("onEpoch.txt", "w") as j:
                    epochs = str(int(l) + 1)
                    j.write(epochs)  
            checkpoint_callback.on_epoch_end(epoch, logs={'loss': loss})
            
    
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
    
#Checkpointing
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



# 3. Train the Model
diffusion_model = DiffusionModel(unet)
epochs = 5  # Number of training epochs
diffusion_model.train(dataset, epochs, checkpoint_callback)

# 4. Evaluate and Save the Model
unet.save('newgen.h5')

# samples = diffusion_model.sample(num_samples=5)
# display_samples(samples)