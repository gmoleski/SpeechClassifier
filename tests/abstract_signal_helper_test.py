import unittest
from speech_classifier import SpeechClassifier
from abstract_signal_helper import AbstractSignalHelper


class AbstractSignalHelperTest(unittest.TestCase):
    def setUp(self):
        speechClassifier = SpeechClassifier()
        self.abstractSignalHelper = AbstractSignalHelper(speechClassifier)


if __name__ == '__main__':
    unittest.main()