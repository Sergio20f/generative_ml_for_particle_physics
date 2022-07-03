import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt



def build_generator(inputs, image_size):
    
    image_resize = image_size // 4
    # network parameters
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]
    
    x = Dense(image_resize * image_resize * layer_filters[0])(inputs)
    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)
    
    for filters in layer_filters:
        # First two convolution layers use strides = 2
        # the last two use strides = 1
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1
        
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2DTranspose(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding='same')(x)
        
    x = Activation('sigmoid')(x)
    generator = Model(inputs, x, name='generator')
    
    return generator

def build_discriminator(inputs):
    
    kernel_size = 5
    layer_filters  = [32, 64, 128, 256]
    
    x = inputs
    for filters in layer_filters:
        #first 3 convolution layers use stride = 2
        #last one uses stride = 1
        if filters == layer_filters[-1]:
            strides = 1
        
        else:
            strides = 2
        
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same')(x)
    
    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    discriminator = Model(inputs, x, name='discriminator')
    
    return discriminator

def build_and_train_models():
    # Load MNIST dataset
    (x_train, _), (_, _) = mnist.load_data()
    
    # Reshape data for CNN as (28, 28, 1) and normalise
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32')/255
    
    model_name = 'dc_glash'
    # Network parameters
    # the latent or z vector is 100-dim
    latent_size = 100
    batch_size = 64
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    
    # Build discriminator model
    inputs = Input(shape=input_shape, name='discriminator_input')
    discriminator = build_discriminator(inputs)
    
    # Original paper uses Adam, but the discrinator converges easily with RMSprop
    optimizer = RMSprop(lr, decay=decay)
    discriminator.compile(loss='binary_crossentropy',
                         optimizer=optimizer,
                         metrics=['accuracy'])
    discriminator.summary()
    
    # Build generator model
    input_shape = (latent_size, )
    inputs = Input(shape=input_shape, name='z_input')
    generator = build_generator(inputs, image_size)
    generator.summary()
    
    # Build adversarial model
    optimizer = RMSprop(lr=lr * 0.5, decay=decay * 0.5)
    # Freeze the weights of discriminator during adversarial training
    discriminator.trainable = False
    # adversarial = generator + discriminator
    adversarial = Model(inputs,
                       discriminator(generator(inputs)),
                       name=model_name)
    adversarial.compile(loss='binary_crossentropy',
                       optimizer=optimizer,
                       metrics=['accuracy'])
    adversarial.summary()
    
    # train discriminator
    models = (generator, discriminator, adversarial)
    params = (batch_size, latent_size, train_steps, model_name)
    train(models, x_train, params)
    
def train(models, x_train, params):
    

    # The GAN component models
    generator, discriminator, adversarial = models
    # network parameters
    batch_size, latent_size, train_steps, model_name = params
    # the generaotr image is saved every 500 steps
    save_interval = 500
    # noise vector to see how the generator output evolves during training
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    
    # number of elements in train dataset
    train_size = x_train.shape[0]
    for i in range(train_steps):
        # train the discriminator for 1 batch
        # 1 batch of real (label=1) and fake images (label=0)
        # randomly pick real images from dataset
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_images = x_train[rand_indexes]
        
        # generate fake images from noise using generator
        # generate noise using uniform distribution
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        
        # generate fake images
        fake_images = generator.predict(noise)
        # real + fake images = 1 batch of train data
        x = np.concatenate((real_images, fake_images))
        # label real images is 1
        y = np.ones([2 * batch_size, 1])
        # fake images label is 0
        y[batch_size:, :] = 0 
        # train discriminator network, log the loss and accuracy
        loss, acc = discriminator.train_on_batch(x, y)
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)
        
        # train the adversarial network for 1 batch
        # since the discriminator weights are frozen in adversarial network only the generator is trained
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        
        y = np.ones([batch_size, 1])
        # train the adversarial network
        loss, acc = adversarial.train_on_batch(noise, y)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)
        
        if (i + 1) % save_interval == 0:
            # plot generator images on a periodic basis
            plot_images(generator,
                       noise_input=noise_input,
                       show=False,
                       step=(i + 1),
                       model_name=model_name)
        
        # save the model after training the generator
        # the trained generator can be reloaded for future MNIST digit generatio
        generator.save(model_name + ".h5")
       
build_and_train_models()
