import tensorflow as tf
from tensorflow.keras import layers, Model, config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
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
    
    def train_step(self, x):

        x_ts = self.generate_ts(len(x))  # Creates random timestep for each batch
        x_a, x_b = self.forward_noise(x, x_ts)

        with tf.GradientTape() as tape:
            predictions = self.model([x_a, tf.expand_dims(x_ts, axis=-1)], training=True)
            loss = self.loss_fn(x_b, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]  # Gradient clipping
        self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))
        
        return loss

    def train(self, dataloader, epochs, checkpoint_callback):
        checkpoint_callback.set_model(self.model)  # Attach the model to the checkpoint callback
        for epoch in range(epochs):
            for batch in dataloader:
                loss = self.train_step(batch)
                print(f"Epoch: {epoch+1}, Loss: {loss}")

            with open("onEpoch.txt", "r") as f:
                l = f.read()
                with open("onEpoch.txt", "w") as j:
                    epochs = str(int(l) + 1)
                    j.write(epochs)  
            checkpoint_callback.on_epoch_end(epoch, logs={'loss': loss})
            
    
    def sample(self, num_samples=1):
        samples = []
        for i in range(num_samples):
            x = tf.random.normal((1, 224, 224, 3))
            for t in range(self.timesteps):
                t_batch = tf.ones((1, 1)) * t
                predicted_noise = self.model(x, training=False)
                x = x - predicted_noise / (self.timesteps ** 0.5)
                # Process the tensor and save the image
            sample = tf.clip_by_value(x, 0, 1)  # Ensure values are in the range [0, 1]
            sample = tf.image.convert_image_dtype(sample[0], dtype=tf.uint8)  # Convert to uint8

            # Convert to numpy array
            sample_np = sample.numpy()

            # Save the image using PIL
            image = Image.fromarray(sample_np)
            image.save(f"./Image_{i}.png")

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
epochs = 20  # Number of training epochs
diffusion_model.train(dataset, epochs, checkpoint_callback)

# 4. Evaluate and Save the Model
unet.save('newgen.h5')

# samples = diffusion_model.sample(num_samples=5)
# display_samples(samples)