import unittest
from speech_classifier import SpeechClassifier
from signal_trainer import SignalTrainer


class SignalTrainerTest(unittest.TestCase):
    def setUp(self):
        speechClassifier = SpeechClassifier()
        self.signalTrained = SignalTrainer(speechClassifier)


if __name__ == '__main__':
    unittest.main()