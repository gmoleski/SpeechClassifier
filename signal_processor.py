import numpy as np
from itertools import islice

class SignalProcessor(object):

    def window_size(self, size, t=30, sec=0.3):
        return int(((size/sec) * t) / 1000)

    def time_domain(self, signals, ms=300.0):
        l = ms/len(signals)
        t_list = [l]
        for i in range(1, len(signals)):
            t_list.append(t_list[i - 1] + l)
        return t_list

    def ideal_delay(self, signals, t=30):
        k = self.window_size(len(signals), t)
        return [0 if ((n - k) < 0) else signals(n - k) for n in range(0, len(signals))]

    def sliding_window(self, signals, k):
        """Sliding window to extract features from the signals"""
        it = iter(signals)
        result = tuple(islice(it, k))
        if len(result) == k:
            yield result
        for val in it:
            result = result[1:] + (val,)
            yield result

    def convolution(self, signals, t=30, mode=None):
        """ Use convolution to extract features of the audio signal """
        e_list = []
        k = self.window_size(len(signals), t)
        for w in self.sliding_window(signals, k):
            window = np.array(w)
            val = ((window[:-1] * window[1:]) < 0).sum() if mode == 'zero' else window.sum()

            if mode == 'energy':
                val = np.power(val, 2)
            elif mode == 'magnitude':
                val = np.abs(val)
            elif mode == 'average':
                val = val / len(window)
            e_list.append(val)

        return e_list
