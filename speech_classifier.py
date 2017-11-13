import argparse
from audio_processor import AudioProcessor
from signal_processor import SignalProcessor
from signal_trainer import SignalTrainer
from speaker_diarization import SpeakerDiarization
import common
import logging as log

log.getLogger().setLevel(log.INFO)

class SpeechClassifier(object):

    def __init__(self):
        self.signalTrainer = SignalTrainer(self)
        self.speakerDiarization = SpeakerDiarization(self)
        self.signalProcessor = SignalProcessor()
        self.audioProcessor = AudioProcessor()


    def train_classifier(self, file, mode=common.SPEECH_SILENCE_MODE):
        self.signalTrainer.train_classifier(mode)
        if file:
            self.audioProcessor.split_to_equal_chunks(file)
            signals = self.audioProcessor.wav_to_signal(file)
            self.signalTrainer.train_classifier(mode, signals=signals)


    def speaker_diarization(self, file):
        if not self.audioProcessor.is_wav_compatible(file):
            file = self.audioProcessor.resample(file)
        data = self.speakerDiarization.diarization(file)
        self.format_in_console(data)

    def speech_silence(self, file):
        pass

    def format_in_console(self, data):
        for id, speech in data.iteritems():
            print '{} has said {}'.format(id, speech)

    def read_data(self):
        """ Read data from storage """
        pass

    def write_data(self):
        """ Write data to storage """
        pass

    def send_data(self):
        """ Send data to remote client """
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Speech classifier for analyzing audio signals")
    tasks = parser.add_subparsers(title="subcommands", description="available tasks", dest="task", metavar="")

    trainClassifier = tasks.add_parser("trainClassifier", help="Train classifier to recognize speech")
    trainClassifier.add_argument("-f", "--file", required=True, help="Enter audio file path")

    speakerDir = tasks.add_parser("speakerDiarization", help="Identify Speakers")
    speakerDir.add_argument("-f", "--file", required=True, help="Enter audio file path")

    speechSilence = tasks.add_parser("trainClassifier", help="Train classifier to recognize speech")
    speechSilence.add_argument("-f", "--file", required=True, help="Enter audio file path")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    speechClassifier = SpeechClassifier()

    if args and args.task == "trainClassifier":  # Train classifier and output performance
        speechClassifier.train_classifier(args.file)
    elif args and args.task == "speakerDiarization":  # Identify speakers
        speechClassifier.speaker_diarization(args.file)
    elif args and args.task == 'speechSilence':
        speechClassifier.speech_silence(args.file)