from abstract_signal_helper import AbstractSignalHelper
from random import shuffle
import os
import common


class SignalTrainer(AbstractSignalHelper):

    def train_classifier(self, mode, signals=None, validate=False):
        if mode == common.SPEECH_SILENCE_MODE:
            emz_speech, emz_silence = self.getEMZData()
            # TODO train and validate data


    def k_fold_validation(self, signals, k=10):
        """
        Splits the list of signals to K slices, shuffles
        TODO: Needs work to handle multiple lists of data
        :param signals: list of signals (int)
        :param k: Num of iterations
        :return:
        """
        slices = [signals[i::k] for i in xrange(k)]
        for i in xrange(k):
            # For each slice representing the validation
            validation = slices[i]
            # Training data are the rest of the values not equal to the validation slice
            training = [signal
                        for slice in slices if slice != validation
                        for signal in slice]
            yield training, validation

    @common.timeit
    def getEMZData(self):
        emz_speech = {}
        emz_silence = {}
        train_root = os.path.join(common.DATA_ROOT, 'samples')
        for filename in os.listdir(train_root):
            if any(filename.startswith(t) for t in common.AUDIO_TYPE) and filename.endswith('.dat'):
                with open(os.path.join(train_root, filename)) as file:
                    signals = [float(line.strip()) for line in file]
                    for type in common.SIGNAL_TYPES:
                        val = self.average(self.speechClassifier.signalProcessor.convolution(signals, mode=type))
                        if filename.startswith(common.SPEECH_SIGNAL):
                            emz_speech[type] = emz_speech.get(type, []) + [val]
                        if filename.startswith(common.SILENCE_SIGNAL):
                            emz_silence[type] = emz_silence.get(type, []) + [val]
        return emz_speech, emz_silence