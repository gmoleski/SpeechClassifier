import sys
import os
import subprocess
import time
import logging as log

# Paths
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = os.path.join(CUR_DIR, 'data')

# Signals
SIGNAL_TYPES = ('energy', 'magnitude', 'zero')
SPEECH_SIGNAL = 'speech'
SILENCE_SIGNAL = 'silence'
AUDIO_TYPE = (SPEECH_SIGNAL, SILENCE_SIGNAL)

SPEECH_SILENCE_MODE = "SPEECH_SILENCE"

# CMD args
JAVA_MEM = '2048'
if sys.platform == 'win32':
    JAVA_MEM = '1024'
    JAVA_EXE = 'javaw'
LIUM_JAR = 'LIUM_SpkDiarization-8.4.1.jar'
LIUM_PATH = os.path.join(CUR_DIR, 'LIUM', LIUM_JAR)

# Server
PORT = 8080
DISCOVERY_URL = ('https://{api}.googleapis.com/$discovery/rest?'
                 'version={apiVersion}')

log.getLogger().setLevel(log.INFO)

def call_subproc(args):
    process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, err = process.communicate(b"input data that is passed to subprocess' stdin")
    ret_code = process.returncode

    if ret_code != 0:
        err = OSError("Subprocess {} closed unexpectedly with code {}".format(str(process), ' '.join(args)))
        err.errno = ret_code
        raise err

def split_file_ext(file):
    fileSplit = file.split('.')
    return ''.join(fileSplit[:-1]), fileSplit[-1]

def file_exists(filename):
    if not os.path.exists(filename):
        raise IOError("Could not create file %s" % filename)
    if os.path.getsize(filename) < 0:
        raise IOError("Invalid file size for %s" % filename)

def timeit(func):
    """Time Decorator to measure the performance of ML methods"""
    def wrapper(*args, **kw):
        t_start = time.time()
        result = func(*args, **kw)
        t_end = time.time()
        log.info('{} took {} sec to process'.format(func.__name__, int(t_end-t_start)))
        return result
    return wrapper

def synchronized(lock):
    """Sync decorator"""
    def wrap(f):
        def newFunction(*args, **kw):
            with lock:
                return f(*args, **kw)
        return newFunction
    return wrap

def file_path(file):
    return os.path.join(DATA_ROOT, 'minutes', file)

def seg_path(file):
    return os.path.join(DATA_ROOT, 'segmentation', file)

def train_path(file):
    return os.path.join(DATA_ROOT, 'train', file)

def sample_path(file):
    return os.path.join(DATA_ROOT, 'samples', file)