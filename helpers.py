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
    
    Output:
        - Mean, standard deviation, variance, covariance of the first distribution. If a dist2 is included then, the covariance
        takes into account dist2 as well.
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
    else:
        if dist2 is not None:
            return np.mean(dist), np.std(dist), np.var(dist), np.cov(dist, dist2)
        
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


def chi_squared(h1, h2):
    """
    Computes the chi squared between two histograms. For the computation, it is designed to just take into consideration the
    heights of each of the bins.
    
    Parameters:
        - h1: array with the heights of each of the bins of histogram 1.
        - h2: array with the heights of each of the bins of histogram 2.
       
    Output:
        - Chi squared coefficient
    """
    coeff = np.sum(((h2-h1)**2)/h1)
    
    return coeff


def align_hist(plot_ht, plot_ht_2, nbins):
    """
    Function that returns a list with new values for the bins' bounds. This is done by redefining the range of both histograms,
    generalise it, and divide that range over the number of desired bins. This list can be used further in the function
    "norm_hist".
    
    Parameters:
        - plot_ht: the output of the function plt.hist(HIST_1)
        - plot_ht_2: the output of the function plt.hist(HIST_2)
        - nbins: number of bins desired
    """
    h1 = plot_ht[0]; x_val = plot_ht[1]
    h2 = plot_ht_2[0]; x_val_2 = plot_ht_2[1]
    
    min_point = min([min(x_val), min(x_val_2)])
    max_point = max([max(x_val), max(x_val_2)])
    rge = max_point - min_point
    
    bin_sep = rge/nbins
    bins_val = [min_point]
    for i in range(nbins):
        point = bins_val[i] + bin_sep
        bins_val.append(point)
    
    return bins_val


def norm_hist(h1, bins_val):
    """
    Function that gets rid of the bins with 0 height by joining them together with the closest non-zero bin. This avoids getting
    an undefined result when "chi_squared" is used.
    
    Parameters:
        - h1: height of the histogram 1 (Expected values/real histogram)
        - bins_val: aligned list of values corresponding to the bounds of the bins. (Generally, output of align_hist)
        
    Output:
        - List with the new bins with non-zero height.
    """
    zero_idx = np.where(h1 == 0)[0]
    for i in zero_idx:
        non_zero_idx = i
        while non_zero_idx in zero_idx:
            non_zero_idx += 1
        
        bins_val[i] = bins_val[non_zero_idx]
    
    bins_val = list(dict.fromkeys(bins_val))
    
    return bins_val