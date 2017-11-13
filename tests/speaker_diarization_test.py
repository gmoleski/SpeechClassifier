import unittest
import mock
import os
from speech_classifier import SpeechClassifier
from speaker_diarization import SpeakerDiarization
from itertools import izip

# Remove dependency from other modules in the test module
CURR = os.path.dirname(os.path.realpath(__name__))

class SpeakerDiarizationTest(unittest.TestCase):
    def setUp(self):
        speechClassifier = SpeechClassifier()
        self.speakerDiarization = SpeakerDiarization(speechClassifier)

    def tst_path(self, file):
        return os.path.join(CURR, 'test_data', file)

    @mock.patch('common.seg_path')
    @mock.patch('common.file_path')
    def test_diariazation(self, mock_file_path, mock_seg_path):
        """
        Tests the diariazation process including segmentation and speech-to-text conversion based on testDiarization2.wav test data
        :param mock_file_path:
        :param mock_seg_path:
        :return:
        """
        mock_seg_path.side_effect = self.tst_path
        mock_file_path.side_effect = self.tst_path

        data = self.speakerDiarization.diarization('testDiarization2.wav')
        speaker_id = 'S0'
        test_data =  "Brian thank you very much she's always a tough living out there"
        self.assertEquals(data.get(speaker_id, ""), test_data)

    @mock.patch('common.seg_path')
    def test_build_speakers_segments(self, mock_seg_path):
        """ Data directory is overriden to use the test files
            This test checks the data from build speakers based on the testDiarization.seg test file"""
        mock_seg_path.side_effect = self.tst_path
        data = self.speakerDiarization.build_speakers_segments('testDiarization.seg', 'testDiarization')
        speaker_ids = ['S1', 'S0']
        genders = ['F', 'M']
        starts = [[1010, 5020], [0]]
        durations = [[4000, 1000], [1000]]
        ends = [[5010, 6020], [1000]]
        for dt, speaker, gender, start, dur, end in izip(data, speaker_ids, genders, starts, durations, ends):
            self.assertEquals(dt['speaker_id'], speaker)
            self.assertEquals(dt['gender'], gender)
            self.assertEquals(dt['start'], start)
            self.assertEquals(dt['duration'], dur)
            self.assertEquals(dt['end'], end)

    @mock.patch('common.seg_path')
    def test_build_speakers_transcript(self, mock_seg_path):
        """
        Tests speakers transcript returned data based on the testDiarization_S0.wav test data
        :param mock_seg_path:
        :return:
        """
        mock_seg_path.side_effect = self.tst_path
        speaker_id = 'S0'
        data = self.speakerDiarization.build_speakers_transcript({speaker_id: 'testDiarization_S0.wav'})
        test_value = "I always receiving any United Kingdom and I was told that I didn't need it on here because of them"
        self.assertEquals(data.get(speaker_id, ''), test_value)

    def test_build_speakers_transcript_no_files(self):
        """
        Checks empty data handling
        :return:
        """
        data = self.speakerDiarization.build_speakers_transcript([])
        self.assertEquals(data, {})


if __name__ == '__main__':
    unittest.main()