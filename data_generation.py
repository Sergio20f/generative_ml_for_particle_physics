import numpy as np


class simple_generation:

    def __init__(self):
        pass
        
    def gaussian_fit(self, mean, std, no_samples):
        """
        Returns array corresponding to a gaussian distribution.
        
        Arguments:
        
        - mean: the mean of the distribution
        - std: standard deviation of the distribution
        - no_samples: number of generated samples
        """
        gauss_dist = np.random.normal(mean, std, no_samples)
        
        return gauss_dist
    
    def linear_fit(self, start_point, step_size, bins):
        """
        Returns array corresponding to a linear distribution.
        
        Arguments: 
        
        - start_point: the starting point of the distribution.
        - step_size: the step size when changing x.
        - bins: the amount of bins on the histogram
        """
        dist = []
        function = lambda x: x
        
        for i in range(bins):
            n_points = int(function(start_point))
            x_1 = start_point
            x_2 = step_size + start_point
            bin_ = np.random.uniform(x_1, x_2, abs(n_points))
            start_point += step_size
            dist.append(bin_)

        dist_conc = np.concatenate(dist, axis=0)
    
        return dist_conc

    def exp_fit(self, no_samples):
        """
        Returns array corresponding to a exponential distribution.
        
        Arguments:
        
        - no_samples: number of samples.
        """
        exp_dist = np.random.exponential(1, no_samples)
        
        return exp_dist

    def quad_fit(self, start_point, step_size, bins):
        """
        Returns array corresponding to a exponential distribution.
        
        Arguments:
        
        - start_point: the starting point of the distribution.
        - step_size: the step size when changing x.
        - bins: the amount of bins on the histogram
        """
        dist = []
        function = lambda x: x**2
        
        for i in range(bins):
            n_points = int(function(start_point))
            x_1 = start_point
            x_2 = step_size + start_point
            bin_ = np.random.uniform(x_1, x_2, n_points)
            start_point += step_size
            dist.append(bin_)

        dist_conc = np.concatenate(dist, axis=0)
    
        return dist_conc