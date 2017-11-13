from abstract_signal_helper import AbstractSignalHelper
import argparse
import base64
import os
import logging as log
from googleapiclient import discovery
import httplib2
from oauth2client.client import GoogleCredentials
import common

log.getLogger().setLevel(log.INFO)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(common.CUR_DIR, 'auth', 'speech_classifier.json')
os.environ['GClOUD_PROJECT']='SpeechClassifier'

class SpeakerDiarization(AbstractSignalHelper):

    def __init__(self, speechClassifier):
        super(SpeakerDiarization, self).__init__(speechClassifier)
        self.speechToTextService = GoogleSpeechToTextService()


    def diarization(self, file):
        """Take a wav file in the right format and build a segmentation file.
        The seg file stores the speaker, start time, duration, gender and also additional info for speech recognition"""
        name, _ = common.split_file_ext(file)
        seg_file = '{}.seg'.format(name)
        seg_path = common.seg_path(seg_file)
        args = [common.JAVA_EXE, '-Xmx{}m'.format(common.JAVA_MEM),
                '-jar', common.LIUM_PATH,
                '--fInputMask={}'.format(common.file_path(file)),  # Input file
                '--sOutputMask={}'.format(seg_path),  # Output file
                '--doCEClustering', name] # Add cluster for each speaker
        log.info('Processing diariazation for {}'.format(file))
        common.call_subproc(args)
        common.file_exists(seg_path)
        log.info('File {} successfully diarized!'.format(file))
        data = self.build_speakers_segments(seg_file, name)
        # Put together audio files for each speaker's part
        sp_file_names = {}
        for speaker in data:
            speaker_id_file = speaker['speaker_id']
            file_names = self.speechClassifier.audioProcessor.audio_segmentation(file, speaker['start'], speaker['end'], concat=True, file_name=speaker_id_file)
            if not file_names:
                log.warn('Waring! Failed to perform audio segmentation for {}'.format(speaker_id_file))
            sp_file_names[speaker_id_file] = file_names[0]
        return self.build_speakers_transcript(sp_file_names)

    def build_speakers_segments(self, seg_file, name):
        """
        Parses the Seg file generated from diarization and creates a list
        of speaker dictionaries with all the details from the file
        :param seg_file: Seg file generated from LIUM
        :param name: basename
        :return: list of dir with speakers' details
        """
        speakers_dict = {}
        with open(common.seg_path(seg_file)) as s_file:
            try:
                for line in s_file:
                    line_sp = line.strip().split(' ')
                    if line_sp[0] != name:
                        continue
                    speaker_id = line_sp[7]
                    sp = speakers_dict.get(speaker_id, {})
                    if not sp:
                        sp['speaker_id'] = speaker_id
                        sp['gender'] = line_sp[4]
                    start = self.seg_to_ms(line_sp[2])
                    duration = self.seg_to_ms(line_sp[3])
                    end = start + duration
                    sp.setdefault('start', []).append(start)
                    sp.setdefault('duration', []).append(duration)
                    sp.setdefault('end', []).append(end)
                    speakers_dict[speaker_id] = sp
            except:
                log.info('Error! Failed to parse {} seg file'.format(seg_file))
                raise
        return speakers_dict.values()

    def build_speakers_transcript(self, speaker_files):
        """
        Processes an audio file for each speaker and returns the transcribed text
        from the input audio file
        :param speaker_files: Dict with Speaker ID and file Names
        :return: Dict with speaker ID and transcribed text
        """
        if not speaker_files:
            return {}
        tr_data = {}
        for id, file in speaker_files.iteritems():
            speech = self.speechToTextService.trascribe(file)
            tr_data[id] = speech
        return tr_data

    def spk_train_init(file):
        """
            TODO: Generate train data
            This improves the Speech Recognition so that the system can recognize speakers recorded from past data
            Train the speaker model using the Gaussian mixture model (GMM) model."""
        name, _ = common.split_file_ext(file)
        args = [common.JAVA_EXE, '-Xmx256m',
                '-cp', common.LIUM_PATH, 'fr.lium.spkDiarization.programs.MTrainInit',
                '--sInputMask={}.seg'.format(common.seg_path(name)),
                '--fInputMask={}'.format(common.file_path(file)),
                '--sInputMask={}.ubm.gmm'.format(common.seg_path(name)),
                '--emInitMethod=copy',
                '--tOutputMask={}.init.gmm'.format(common.seg_path(name)),
                name]
        common.call_subproc(args)
        common.file_exists('%s.init.gmm' % name)


    def spk_train_map(file):
        """
           TODO: Generate train data
           This improves the Speech Recognition so that the system can recognize speakers recorded from past data
           Train the speaker model using the Maximum a posteriori (MAP) adaptation of GMM
        """
        name, _ = common.split_file_ext(file)
        args = [common.JAVA_EXE, '-Xmx256m',
                '-cp', common.LIUM_PATH, 'fr.lium.spkDiarization.programs.MTrainMAP',
                '--sInputMask={}.ident.seg'.format(common.seg_path(name)),
                '--fInputMask={}'.format(common.file_path(file)),
                '--sInputMask={}.init.gmm'.format(common.seg_path(name)),
                '--emCtrl=1,5,0.01',
                '--varCtrl=0.01,10.0',
                '--tOutputMask={}.gmm'.format(common.seg_path(name)),
                name]
        common.call_subproc(args)
        common.file_exists(name + '.gmm')



class GoogleSpeechToTextService(object):

    def get_speech_service(self):
        credentials = GoogleCredentials.get_application_default().create_scoped(
            ['https://www.googleapis.com/auth/cloud-platform'])
        http = httplib2.Http()
        credentials.authorize(http)

        return discovery.build('speech', 'v1beta1', http=http, discoveryServiceUrl=common.DISCOVERY_URL)

    def trascribe(self, file):
        """Transcribe the given audio file.

        Args:
            file: the name of the audio file.
        """
        with open(common.seg_path(file), 'rb') as speech:
            speech_content = base64.b64encode(speech.read()).decode('utf-8')
        log.info('Read file {} for trascription'.format(file))
        service = self.get_speech_service()
        service_request = service.speech().syncrecognize(
            body={
                'config': {
                    'encoding': 'LINEAR16',  # raw 16-bit signed LE samples
                    'sampleRate': 16000,  # 16 khz
                    'languageCode': 'en-US',  # a BCP-47 language tag
                },
                'audio': {
                    'content': speech_content.decode('UTF-8')
                }
            })
        log.info('Sending file {} to Google Speech-To-Text service for processing'.format(file))
        response = service_request.execute()
        log.info('File {} successfully processed by Google Speech-To-Text service'.format(file))
        return self.response_parser(response)

    def response_parser(self, response):
        """
        Parses API's response and extracts the transcribed text
        :param response: List dicts from GTTS API
        :return: Transcribed text str
        """
        try:
            text_response = str(response[u'results'][0][u'alternatives'][0][u'transcript'])
        except:
            msg = 'Failed to parse transcription for {}'.format(response)
            log.info(msg)
            raise Exception(msg)
        return text_response

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('speech_file', help='Full path of audio file to be recognized')
    args = parser.parse_args()
    speechToText = GoogleSpeechToTextService()
    speechToText.trascribe(args.speech_file)