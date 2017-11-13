import numpy as np
import logging as log

class AbstractSignalHelper(object):

    def __init__(self, speechClassifier):
        self.speechClassifier = speechClassifier

    def std(self, variance):
        return np.sqrt(variance)

    def average(self, l):
        return reduce(lambda x, y: x + y, l) / len(l)

    def variance(self, l, mean):
        """
        Calculate the variance by subtracting the mean and squaring.
        The average of the result is returned
        """
        return reduce(lambda x, y: x + np.power(y - mean, 2), l) / len(l)

    def normal_prob_distribution(self, x, mean, std):
        """
        Calculates the Gaussian normal probability distribution for the trained values
        using mean and standard deviation
        """
        den = std * np.sqrt(2 * np.pi * std)
        exp = (-np.power((x - mean), 2)) / (2 * np.power(std, 2))
        return (1 / den) * np.exp(exp)

    def seg_to_ms(self, value):
        ms_val = 0
        try:
            ms_val = int(value) * 10
        except ValueError:
            log.info('Could not convert seg value {} to ms'.format(value))
        return ms_val