import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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


def make_animation(real_data: tuple, generated_point_list):
    """
    Function that makes an animation of the generated data. It can be used to evaluate how the generator is doing with
    respect to time or epochs.

    Parameters:
        - real_data: Real data as a tuple
        - generated_point_list: List of generated points
    """
    real_x, real_y = real_data

    camera = Camera(plt.figure())

    plt.xlim(real_x.min() - 0.2, real_x.max() + 0.2)
    plt.ylim(real_y.min() - 0.05, real_y.max() + 0.05)

    for i in range(len(generated_point_list)):
        plt.scatter(real_x, real_y, c='blue')
        fake_x, fake_y = generated_point_list[i][:, 0], generated_point_list[i][:, 1]
        plt.scatter(fake_x, fake_y, c='red')
        camera.snap()

    animation = camera.animate(blit=True)
    plt.close()
    animation.save('animation.gif', fps=10)


def stats_dist(dist, dist2=None, prnt=True):
    """
    Calculates the mean, standard deviation, variance, and covariance matrix of a distribution. The function also includes
    the option to calculate the covariance matrix of two distributions. To do this, pass the second distribution as the
    second parameter.

    Parameters:
        - dist (numpy array): The distribution to be analyzed.
        - dist2 (numpy array): The second distribution to be analyzed.
        - prnt (bool): Whether or not to print the results. If print is False, the function will simply return the
        results.
    """
    if prnt:
        print("-------------------------------")
        print("Mean:", np.mean(dist))
        print("-------------------------------")
        print("Standard deviation:", np.std(dist))
        print("-------------------------------")
        print("Variance:", np.var(dist))
        print("-------------------------------")
        if dist2 is not None:
            print("Covariance matrix:", np.cov(dist, dist2))
            return np.mean(dist), np.std(dist), np.var(dist), np.cov(dist, dist2)
        else:
            print("Covariance:", np.cov(dist))  # Will be a scalar

    return np.mean(dist), np.std(dist), np.var(dist), np.cov(dist)


def df_style(_):
    return "font-weight: bold"


def df_generator(stats_or_x, stats_gen_x, stats_or_y, stats_gen_y, rows: np.ndarray, df_style):
    """
    Generates a dataframe with the given parameters.

    Parameters:
        - stats_or_x: The stats of the x-dimension from the train data.
        - stats_gen_x: The stats of the x-dimension from the generated data.
        - stats_or_y: The stats of the y-dimension from the train data.
        - stats_gen_y: The stats of the y-dimension from the generated data.
        - rows: Numpy array with the rows of the dataframe as strings.
        - df_style: The style of the dataframe. Should be coming from a function like "df_style"
    """
    stats_df = pd.DataFrame()
    stats_df[" "] = rows
    stats_df["original_x_dim"] = np.array([stats_or_x[0], stats_or_x[1], stats_or_x[2], stats_or_x[3].item()])
    stats_df["generated_x_dim"] = np.array([stats_gen_x[0], stats_gen_x[1], stats_gen_x[2], stats_gen_x[3].item()])
    stats_df["original_y_dim"] = np.array([stats_or_y[0], stats_or_y[1], stats_or_y[2], stats_or_y[3].item()])
    stats_df["generated_y_dim"] = np.array([stats_gen_y[0], stats_gen_y[1], stats_gen_y[2], stats_gen_y[3].item()])
    stats_df = stats_df.style.applymap(df_style, subset=[" "])

    return stats_df
