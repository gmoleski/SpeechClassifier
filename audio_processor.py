from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy.io.wavfile import read
import numpy as np
import math
import os
import sox
import logging as log
import common
from itertools import izip
import wave

log.getLogger().setLevel(log.INFO)

class AudioProcessor(object):

    def split_to_equal_chunks(self, file):
        """
        Splits Wav file to smaller chunks of equal duration for training and better signal processing
        :param file: fileName with extension
        :return:
        """
        audio = AudioSegment.from_file(common.file_path(file), "wav")
        duration_in_sec = len(audio) / 1000
        bit_rate = int((audio.frame_rate * audio.frame_width * 8 * audio.channels) / 1000)
        size = (bit_rate * audio.frame_rate * audio.channels * duration_in_sec) / 8

        len_in_sec = math.ceil((duration_in_sec * 10000000 ) /size)
        len_in_ms = len_in_sec * 1000
        chunks = make_chunks(audio, len_in_ms)
        self.export_audio_chunks(file, chunks)

    def audio_segmentation(self, file, start_list, end_list, concat=False, file_name=None):
        """
        Breaks the file into small parts based on time slices and puts it back together if
        the concat option is True
        :param file: filename with extension
        :param start_list: list of ints representing start time ms
        :param end_list: list of ints representing en time ms
        :param concat: option to merge the file
        :param file_name: new file name for export
        :return: new file name/s
        """

        file_names = []
        baseName, ext = common.split_file_ext(file)
        seg_name = '{0}_{1}.{2}'.format(baseName, file_name if file_name else 'seg', ext)
        audio = AudioSegment.from_file(common.file_path(file), "wav")
        duration_in_ms = len(audio) * 1000
        audio_segs = [audio[start:end] for start, end in izip(start_list, end_list) if (duration_in_ms >= start >= 0) and (duration_in_ms >= end > 0)]
        if not audio_segs:
            return file_names
        if concat:
            seg_path = common.seg_path(seg_name)
            audio_concat = reduce(lambda x, y: x + y, audio_segs)
            audio_concat.export(seg_path, format="wav")
            file_names.append(seg_name)
            common.file_exists(seg_path)
        else:
            file_names =self.export_audio_chunks(seg_name, audio_segs)
        return file_names

    def export_audio_chunks(self, file, chunks):
        """
        For each chunk of audio it gets exported to wav
        :param file: file name with extension
        :param chunks: small chunks of wav
        :return: new exported file names
        """
        chuck_names = []
        for i, chunk in enumerate(chunks):
            file_path = common.file_path(file)
            chunk_name = "{0}_{1}.wav".format(file, i)
            chunk.export(file_path, format="wav")
            chuck_names.append(chunk_name)
            common.file_exists(file_path)
        return chuck_names

    def wav_to_signal(self, file, format=int):
        """ Extracts the digital signal of the wav file for processing and feature extraction"""
        signals = read(os.path.join(common.DATA_ROOT, file))
        return np.array(signals[1], dtype=format)


    def export_to_dat(self, signals, file):
        """ Exports list of digital signals to .dat file"""
        datFileName = '{0}.dat'.format(file)
        with open(os.path.join(os.path.dirname(common.DATA_ROOT), datFileName), mode='wt') as datFile:
            datFile.write('\n'.join(str(signal) for signal in signals))
            datFile.write('\n')

    def resample(self, file):
        """Uses Sox to resample the wav file to 16kHz, 1 channel, 16 bit wav which is
        the ideal format for processing"""
        sampler = sox.Transformer()
        sampler.convert(samplerate=16000, n_channels=1, bitdepth=16)
        resampled_file = '{0}_sampled'.format(file)
        resampled_path = common.file_path(resampled_file)
        sampler.build(common.file_path(file), resampled_path)
        common.file_exists(resampled_path)
        return resampled_file


    def is_wav_compatible(self, file):
        """Check if the wave is in the correct wav format for processing.
            wave provides the file's property params and the audio's channel, length and bit rate are checked
            Also the file must not be compressed"""
        try:
            wav_file = wave.open(common.file_path(file))
            params = wav_file.getparams()
            wav_file.close()
        except wave.Error, exc:
            log.info('Failed to retrieve properties of the wav file: {}'.format(exc))
            return False
        return params[:3] == (1, 2, 16000) and params[-1:] == ('not compressed',)
