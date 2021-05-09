import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
import cherrypy
import json
import base64

sampling_rate = 16000

MFCC_OPTIONS = {'frame_length': 640,
                'frame_step': 320,
                'mfcc': True,
                'lower_frequency': 20,
                'upper_frequency': 4000,
                'num_mel_bins': 40,
                'num_coefficients': 10
                }

interpreter = tf.lite.Interpreter(model_path="big.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']


class Big_Model_s(object):
    exposed = True

    def GET(self, *path, **query):
        pass

    def POST(self, *path, **query):

        req = cherrypy.request.body.read()
        body = json.loads(req)
        
        audio_string = body["e"][0]["vd"]
        audio_binary = base64.b64decode(audio_string)
        
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)
        
        zero_padding = tf.zeros(sampling_rate - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape(sampling_rate)
        
        stft = tf.signal.stft(audio, frame_length= MFCC_OPTIONS['frame_length'],
        frame_step=MFCC_OPTIONS['frame_step'], fft_length=MFCC_OPTIONS['frame_length'])
        spectrogram = tf.abs(stft)
        
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                                MFCC_OPTIONS['num_mel_bins'],
                                MFCC_OPTIONS['frame_length'] // 2 + 1,
                                sampling_rate,
                                MFCC_OPTIONS['lower_frequency'],
                                MFCC_OPTIONS['upper_frequency']
                                )
        mel_spectrogram = tf.tensordot(spectrogram,linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[:, : MFCC_OPTIONS['num_coefficients']]
        mfccs = tf.expand_dims(mfccs, -1)
        mfccs = np.expand_dims(mfccs, axis=0).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], mfccs)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        response = {
            'pred' : int(np.argmax(output_data))
        }
        response = json.dumps(response)

        return response
        
    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':
    conf = {
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True,
        }
    }
    cherrypy.tree.mount (Big_Model_s(), "/",conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()


