import numpy as np


class simple_generation:

    def __init__(self, no_samples):
        
        self.no_samples = no_samples
        
    def gaussian_fit(self, mean, std):
        """
        Returns array corresponding to a gaussian distribution.
        """
        gauss_dist = np.random.normal(mean, std, self.no_simples)
        
        return gaussian_dist
    
    def linear_fit(self, start_point, step_size, bins):
        """
        Returns array corresponding to a linear distribution.
        """
        dist = []
        function = lambda x: x
        
        for i in range(bins):
            n_points = int(function(start_point))
            x_1 = start_point
            x_2 = step_size + start_point
            bin_ = np.random.uniform(x_1, x_2, n_points)
            start_point += step_size
            dist.append(bin_)

        dist_conc = np.concatenate(dist, axis=0)
    
        return dist_conc

    def exp_fit(self):
        """
        Returns array corresponding to a exponential distribution.
        """
        exp_dist = np.random.exponential(1,self.no_samples)
        
        return exp_dist

    def quad_fit(self, start_point, step_size, bins):
        """
        Returns array corresponding to a exponential distribution.
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