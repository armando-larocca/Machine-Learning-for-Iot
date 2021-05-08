import numpy as np
import tensorflow as tf
import time
import datetime
from scipy.io import wavfile
import argparse
import timeit
import pyaudio
import wave
import sys
from scipy import signal
import io
from io import BytesIO
import subprocess

# ARUGUMENTS INITIALIZATION
parser = argparse.ArgumentParser()
parser.add_argument("--num-samples", type=int, help="number of recordings")
parser.add_argument("--output", type=str, help="output directory")
args = parser.parse_args()

subprocess.Popen(['sudo sh -c "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'], shell=True)
print("frequency:")
subprocess.run(['cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq'], shell=True)

#RECORDING PART
form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 48000 # 44.1kHz sampling rate
chunk = 4800 # 2^12 samples for buffer
record_secs = 1 # seconds to record
dev_index = 1 # device index found by p.get_device_info_by_index(ii)

#SAMPLING data
sampling_ratio = 16000

# STFT data
frame_length = 640
frame_step = 320

# MFCCS data
num_mel_bins = 40
sampling_rate = 16000
lower_frequency = 20
upper_frequency = 4000
num_spectrogram_bins = 321

subprocess.Popen(['sudo sh -c "echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset"'], shell=True)

audio = pyaudio.PyAudio()
stream = audio.open(format = form_1, rate = samp_rate, channels = 1, \
                    input_device_index = 1 ,input = True, \
                    frames_per_buffer=chunk)
stream.stop_stream()


linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                                num_mel_bins,
                                num_spectrogram_bins,
                                sampling_rate, # 16000
                                lower_frequency,
                                upper_frequency)

for i in range(args.num_samples):

    #START TIMING
    #print("##############", i )
    starttime = timeit.default_timer()
    
    ##** RECORDING **##

    stream.start_stream()
    buf = io.BytesIO()
    
    subprocess.Popen(['sudo', 'sh', '-c', 'echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']
            )
    
    # loop through stream and append audio chunks to frame array
    for ii in range(0,int((samp_rate/chunk) )):
        buf.write( stream.read(chunk) )
        
        if ii == int(samp_rate/chunk) - 2 :
            subprocess.Popen(['sudo', 'sh', '-c', 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'])

    stream.stop_stream()
    #print("recording finished ",  timeit.default_timer() - starttime )

    ###** PREPROCESSING **###
    
    # SAMPLING
    audio_res = signal.resample_poly(np.frombuffer(buf.getvalue(),dtype= np.int16), 1, 3)
    audio_res = audio_res.astype(np.float32)
    #print("sampling ok", timeit.default_timer() - starttime)
    
    # SPECTROGRAM
    tf_audio = tf.convert_to_tensor(audio_res)
    #print("Conversione", timeit.default_timer() - starttime)
    
    stft = tf.signal.stft(tf_audio,
                    frame_length=frame_length,
                    frame_step=frame_step,
                    fft_length=frame_length)

    #print("stft", timeit.default_timer() - starttime)
    
    spectrogram = tf.abs(stft)
    #print("spectrogram shape",spectrogram.shape) # 49 x 321
    #print("spectrogram ok",  timeit.default_timer() - starttime )
    
    #MFCCS
    #num_spectrogram_bins = spectrogram.shape[-1]

    mel_spectrogram = tf.tensordot( spectrogram,
                              linear_to_mel_weight_matrix,
                              1)
    #print("mel spectrogram",  timeit.default_timer() - starttime )

    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(
                                linear_to_mel_weight_matrix.shape[-1:]))

    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms( log_mel_spectrogram)[:, :10]
    #print("mfccs ok", timeit.default_timer() - starttime )
    
    # SAVING PART
    to_save = tf.io.serialize_tensor(mfccs)
    tf.io.write_file("./"+ args.output + "/mfccs" + str(i) + ".bin" , to_save)
    
    
    #TAKE TIME
    #print("The time difference is :", timeit.default_timer() - starttime)
    print( timeit.default_timer() - starttime )
    #print("\n\n\n")


stream.close()
audio.terminate()
subprocess.Popen(['cat /sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state'], shell=True)
