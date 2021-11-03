import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()
import numpy as np


class Glash:

    def __init__(self, no_samples, input_shape):
        """
        Init method constructs the GAN model with a simple generator and discriminator part. Global variables for the model and each of its parts are also defined.
        
        **Args:
        - no_samples: number of data samples contained in the data distribution.
        - input_shape: shape of the input vector to the network.
        """
        
        self.no_samples = no_samples
        # GAN Architecture
        # Generator part
        generator = tf.keras.models.Sequential([
            tf.keras.layers.Dense(50, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(150, activation='relu'),
            tf.keras.layers.Dense(self.no_samples, activation='sigmoid')
        ])

        # Discriminator side
        discriminator = tf.keras.models.Sequential([
            tf.keras.layers.Dense(150, activation='relu', input_shape=[self.no_samples]),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        glash = tf.keras.models.Sequential([generator, discriminator])
        self.glash = glash
        self.discriminator = discriminator
        self.generator = generator
        
    def glash_compile(self):
        """
        Compiles the model using binary crossentropy and Adam as optimizer.
        Sets discriminator.trainable to False.
        """
        self.discriminator.compile(loss='binary_crossentropy', optimizer='Adam')
        self.discriminator.trainable = False
        self.glash.compile(loss='binary_crossentropy', optimizer='Adam')

    def train_glash(self, dataset, batch_size, no_epochs=35):
        """
        Uses batches of data to feed into the network. The generator part simulates a new data distribution which gets to the discriminator network and this one gets trained to classify whether the newly generated data is similar enough to the training data. The training process consists of 2 phases: training the discriminator and training the generator.
        
        **Arguments:
        - dataset: dataset, normally the output of self.data_processing(). In this case it needs to be a numpy array divided into batches.
        - batch_size: Batch size desired.
        - no_epochs: number of epochs for the training.
        """ 
        generator, discriminator = self.glash.layers
        for epoch in range(no_epochs):
            for X_batch in dataset:
                # Phase 1- Training the discriminator
                noise = tf.random.normal(shape=[batch_size, self.no_samples])
                generated_dist = generator(noise)
                X_fake_and_real = tf.concat([generated_dist, X_batch], axis=0)
                y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
                discriminator.trainable = True
                discriminator.train_on_batch(X_fake_and_real, y1)
                
                # Phase 2 - Training the generator
                noise = tf.random.normal(shape=[batch_size, self.no_samples])
                y2 = tf.constant([[1.]] * batch_size)
                discriminator.trainable = False
                self.glash.train_on_batch(noise, y2)
                
    def data_processing(self, dataset, batch_size):
        """
        Takes a list or numpy array containing data distributions, splits them into batches of size -> batch_size and returns the new processed data ready for training.
        
        **Arguments:
        - dataset: list or numpy array containing a set of data distributions.
        - batch_size: Batch size desired.
        """
        if type(dataset) == list:
            dataset = np.array(dataset)

        if type(dataset) == np.ndarray:
            proc_dataset = np.array(np.split(dataset, int(len(dataset) / batch_size)))

        else:
            return "Input dataset has the wrong format"

        return proc_dataset

    def glash_predict(self, sample):
        """
        Returns a prediction based on the input new sample.
        """            
        return self.generator(sample)