import unittest
from audio_processor import AudioProcessor


class AudioProcessorTest(unittest.TestCase):
    def setUp(self):
        self.signalTrained = AudioProcessor()


if __name__ == '__main__':
    unittest.main()