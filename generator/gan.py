import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
BATCH_SIZE = 4
EPOCHS = 5
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])
# Load and preprocess images from a directory
def load_data(image_dir, img_height=224, img_width=224):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        label_mode=None,  # Since it's an unconditional generator, we don't need labels
        image_size=(img_height, img_width),  # Resize images to the target size
        batch_size=BATCH_SIZE  # We'll handle batching later
    )

    # Normalize images to [-1, 1]
    dataset = dataset.map(lambda x: (x - 127.5) / 127.5)

    # Convert dataset to a numpy array
    #images = np.concatenate([x for x in dataset], axis=0)
    print(dataset)
    return dataset

# Generator model for 224x224x3 images
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(14*14*512, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((14, 14, 512)))
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    return model

generator = build_generator()

# Discriminator model for 224x224x3 images
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[224, 224, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

discriminator = build_discriminator()

# GAN model
# def build_gan(generator, discriminator):
#     discriminator.compile(optimizer='adam', loss='binary_crossentropy')
#     discriminator.trainable = False
#     gan_input = layers.Input(shape=(100,))
#     generated_image = generator(gan_input)
#     gan_output = discriminator(generated_image)
#     gan = tf.keras.Model(gan_input, gan_output) 
#     gan.compile(optimizer='adam', loss='binary_crossentropy')
#     discriminator.trainable = True
#     return gan

# # Training the GAN
# def train_gan(generator, discriminator, gan, image_dir, epochs=1, batch_size=4, sample_interval=100):
#     train_images = load_data(image_dir)
#     real = np.ones((batch_size, 1))
#     fake = np.zeros((batch_size, 1))
#     for epoch in range(epochs):
#         for _ in range(int(5000 / batch_size)):
#             #discriminator.trainable = False
#             idx = np.random.randint(0, train_images.shape[0], batch_size)
#             real_images = train_images[idx]
#             noise = np.random.normal(0, 1, (batch_size, 100))
#             generated_images = generator.predict(noise)
#             d_loss_real = discriminator.train_on_batch(real_images, real)
#             d_loss_fake = discriminator.train_on_batch(generated_images, fake)
#             d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
#             noise = np.random.normal(0, 1, (batch_size, 100))

#             #discriminator.trainable = True
#             g_loss = gan.train_on_batch(noise, real, return_dict=True)
            
#             print(f"\n{epoch} [D loss: {d_loss}] [G loss: {g_loss['loss']}]")

#         generator.save(f'generator_epoch_{epoch + 1}.h5')
#         discriminator.save(f'discriminator_epoch_{epoch + 1}.h5')
#         gan.save(f'gan_epoch_{epoch + 1}.h5')

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images, epoch):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        print('zero')
        generated_images = generator(noise, training=True)
        print('first')
        real_output = discriminator(images, training=True)
        print('seconds')
        fake_output = discriminator(generated_images, training=True)
        print('third')
        
        gen_loss = generator_loss(fake_output)
        print('fourth')
        disc_loss = discriminator_loss(real_output, fake_output)
        print('fifth')
        print(f"\n\nEpoch: {epoch} | Generator Loss: {gen_loss} | Discriminator Loss: {disc_loss}")

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    print('negative')
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    print('beyond')
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    print('final')

# Sample and save images
def sample_images(generator, image_grid_rows=4, image_grid_columns=4):
    noise = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, 100))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(4, 4), sharey=True, sharex=True)
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(generated_images[cnt])
            axs[i, j].axis('off')
            cnt += 1
    plt.show()

def train(dataset, epochs):
  for epoch in range(epochs):
    for image_batch in dataset:
        train_step(image_batch, epoch)
    generator.save(f"gen_epoch_{epoch}")
    discriminator.save(f"disc_epoch_{epoch}")



  
# Main function to run the GAN
#if __name__ == "__main__":
#gan = build_gan(generator, discriminator)
image_directory = './dataset/HAM10000_images_part_1'
#train_gan(generator, discriminator, gan, image_directory)

train(load_data(image_directory), EPOCHS)
#sample_images(generator)
    