import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera


def toy_data(n, rmin, rmax):
    """
    Function to generate toy data. Random uniform data is generated in the range provided and then it goes through a
    mathematical function to generate a second dimension of data. This function is for 2D data.

    Parameters:
        - n: Number of data points to generate
        - rmin: Minimum value of the range
        - rmax: Maximum value of the range
    """
    x1 = np.random.uniform(rmin, rmax, n)
    x2 = x1 ** 2
    x1 = x1.reshape((n, 1))
    x2 = x2.reshape((n, 1))
    samples = np.hstack((x1, x2))

    return samples


def show_samples(generated_point_list, epoch, generator, data, n=100, l_dim=5):
    """
    Function that shows the generated samples every 20 epochs and compares them with the real data.

    Parameters:
        - generated_point_list: List of generated points
        - epoch: Current epoch
        - generator: Generator model
        - data: Real data
        - n: Number of samples to show
        - l_dim: Dimension of the latent space
    """
    if epoch % 20 == 0:
        noise = tf.random.normal(shape=(n, l_dim))
        generated_data = generator(noise)
        generated_point_list.append(generated_data)


def make_animation(real_data:tuple, generated_point_list):
    """
    Function that makes an animation of the generated data. It can be used to evaluate how the generator is doing with
    respect to time or epochs.

    Parameters:
        - real_data: Real data as a tuple
        - generated_point_list: List of generated points
    """
    real_x, real_y = real_data

    camera = Camera(plt.figure())

    plt.xlim(real_x.min()-0.2, real_x.max()+0.2)
    plt.ylim(real_y.min()-0.05, real_y.max()+0.05)

    for i in range(len(generated_point_list)):
        plt.scatter(real_x, real_y, c='blue')
        fake_x, fake_y = generated_point_list[i][:, 0], generated_point_list[i][:, 1]
        plt.scatter(fake_x, fake_y, c='red')
        camera.snap()

    animation = camera.animate(blit=True)
    plt.close()
    animation.save('animation.gif', fps=10)