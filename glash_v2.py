import tensorflow as tf


def glash_discriminator(n=2):
    """
    Discriminator network side of the GAN. It consists of a sequence of fully connected dense layers that output a
    probability of the input being real (1) or fake (0). It aims to distinguish between real and fake distributions
    (generation samples from the generative network).

    Arguments:
        - n: number of samples it takes as an input in 1D
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n,)),
        tf.keras.layers.Dense(25, activation='relu',
                              kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name='glash_discriminator')

    return model
    

def glash_generator(latent_dim=5):
    """
    Generative network component of the GAN. It consists of a sequence of fully connected dense layers that output a
    newly generated distribution. It aims to generate a distribution that is close to the input distribution improving
    its weights and biases until the discriminator network is no longer able to recognise fake from real data.

    Arguments:
        - latent_dim: number of dimensions of the latent space
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim,)),
        tf.keras.layers.Dense(25, activation='relu',
                              kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(2, activation='tanh') # Test different activation functions for different shapes
    ], name='glash_generator')

    return model


class Glash(tf.keras.Model):

    def __init__(self, discriminator, generator, latent_dim=5):
        super(Glash, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, optimizerD, optimizerG, loss_fn):
        super(Glash, self).compile()
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG
        self.loss_fn = loss_fn

    def train_step(self, real_data):
        if isinstance(real_data, tuple):
            real_data = real_data[0]

        # Generate fake data with the appropriate shape and batch size
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_data = self.generator(noise)

        # Combine real and fake data for training the discriminator
        combined_data = tf.concat([real_data, fake_data], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        # Training the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_data)
            d_loss = self.loss_fn(labels, predictions)

        # Update the weights of the discriminator
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.optimizerD.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Generate fake labels to train the generator
        misleading_labels = tf.ones((batch_size, 1))
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Training the generator
        with tf.GradientTape() as tape:
            fake_predictions = self.discriminator(self.generator(noise))
            g_loss = self.loss_fn(misleading_labels, fake_predictions)

        # Update the weights of the generator
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.optimizerG.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {'d_loss': d_loss, 'g_loss': g_loss}
