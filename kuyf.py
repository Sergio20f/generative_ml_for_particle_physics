import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Sampling(keras.layers.Layer):
    """
    samples codings from a normal distribution.samples a random vector from a normal     distribution with mean 0 and standard deviation of 1 
    """
    def call(self, inputs):
        mean, log_var = inputs
        return tf.random.normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean


class Kuyf:

    def __init__(self, features_num, coding_size=2):
        """
        Creates a Variational Autoencoder that works with 1D data
        
        Inputs:
        - features_num: how many datapoints there are in the dataset
        - codings_size: the lenght of the codings vector, the codings here are not             generated directly from the input  
        
        """

        # Encoder part

        inputs = keras.layers.Input(shape=[features_num])
        z = keras.layers.Dense(100, activation='selu')(inputs)
        z = keras.layers.Dense(50, activation='selu')(z)
        codings_mean = keras.layers.Dense(coding_size)(z)
        codings_log_var = keras.layers.Dense(coding_size)(z)
        codings = Sampling()([codings_mean, codings_log_var])
        encoder = keras.Model(inputs=inputs, outputs=[
                              codings_mean, codings_log_var, codings])

        # Decoder part

        decoder_inputs = keras.layers.Input(shape=[coding_size])
        x = keras.layers.Dense(50, activation='selu')(decoder_inputs)
        x = keras.layers.Dense(100, activation='selu')(x)
        outputs = keras.layers.Dense(
            features_num, activation=tf.keras.layers.LeakyReLU())(x)
        decoder = keras.Model(inputs=decoder_inputs, outputs=outputs)

        # Combine Encoder and Decoder

        _, _, codings = encoder(inputs)
        reconstructions = decoder(codings)
        kuyf = keras.Model(inputs=inputs, outputs=reconstructions)

        self.kuyf = kuyf
        self.encoder = encoder
        self.decoder = decoder
        self.codings = codings
        self.codings_log_var = codings_log_var
        self.codings_mean = codings_mean
        self.coding_size = coding_size

    def kuyf_compile(self):

        latent_loss = -0.5 * tf.reduce_sum(1 + self.codings_log_var - tf.exp(
            self.codings_log_var) - tf.square(self.codings_mean))
        self.kuyf.add_loss(latent_loss)
        self.kuyf.compile(loss=tf.keras.losses.KLDivergence(),
                          optimizer=keras.optimizers.Adam())

    def kuyf_fit(self, dataset, k_epochs, k_batch_size):

        history = self.kuyf.fit(
            dataset, dataset, epochs=k_epochs, batch_size=k_batch_size,verbose=0)

        return history

    def kuyf_generate(self):

        codings = tf.random.normal([1, self.coding_size])
        distributions = self.decoder(codings)

        return distributions
