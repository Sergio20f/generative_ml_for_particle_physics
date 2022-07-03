import numpy as np

class simple_generation:
    """
Class containing the necessary methods in order to be able to generate toy data that will be fed into the variational  autoenconder and GAN. It has several methods used to create and transform data distributions from one space to others being able to go back.
    """
    
    def __init__(self, mean, std, no_samples):
        """
        Object initiates creating a gaussian distribution of data taking the inputs:
        
        - mean (float/int)
        - std (float/int): standard deviation.
        - no_samples (int): number of samples that go into the distributions.
        """
        self.mean = mean
        self.std = std
        self.no_samples = no_samples
        self.init_dist = np.random.normal(mean, std, no_samples)
        
    def init_dist(self):
        """
        Method that returns the initial gaussian distribution.
        """
        return self.init_dist
    
    def quad_fit(self):
        """
        Method that fits the normally distributed data into a quadatric distribution.
        """
        c = np.log(1/(np.sqrt(2*np.pi)*self.std))
        self.c = c
        quad_function = lambda x: c - ((x-self.mean)**2)/(2*self.std**2)
        quad_dist = quad_function(self.init_dist)

        self.quad_function = quad_function
        
        return quad_dist

    def linear_fit(self):
        """
        Method that fits the normally distributed data into a linear distribution.
        """
        lin_function = np.sqrt(-self.quad_fit() + self.c)
        return lin_function
    
    def c_return(self):
        """
        Simple method that returns the value of the constant used to do the data transformations, c.
        """
        return self.c