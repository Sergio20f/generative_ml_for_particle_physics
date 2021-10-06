import numpy as np


class simple_generation:

    def __init__(self, std, no_samples, no_dist):
        self.no_samples = no_samples
        self.std = std
        self.no_dist = no_dist

        dists = []
        for i in range(self.no_dist):
            normalised = np.random.normal(0, self.std, self.no_samples)
            # self.normalised = normalised
            dists.append(normalised)

        self.dists = dists

    def linear_fit(self):
        """
        Returns array of tuples, each corresponding to a linear function.
        """
        lines = []
        for i in self.dists:
            line = (i, i)
            lines.append(line)
        return lines

    def exp_fit(self):
        """
        Returns array of tuples, each corresponding to an exponential function.
        """
        exponentials = []
        for i in self.dists:
            exponential = (i, np.exp(i))
            exponentials.append(exponential)

        return exponentials

    def quad_fit(self):
        """
        Returns array of tuples, each corresponding to a quadratic function.
        """
        quads = []
        for i in self.dists:
            quad = (i, i ** 2)
            quads.append(quad)

        return quads
