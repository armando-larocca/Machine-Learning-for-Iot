import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt
import paho.mqtt.client as PahoMQTT
import time
import tensorflow as tf
import json
import numpy as np
import argparse
import pathlib
import os
from datetime import datetime
from collections import Counter


class SignalGenerator:
        def __init__(self, labels, sampling_rate, frame_length, frame_step,
                num_mel_bins=None, lower_frequency=None, upper_frequency=None,
                num_coefficients=None, mfcc=False):
            self.labels = labels
            self.sampling_rate = sampling_rate
            self.frame_length = frame_length
            self.frame_step = frame_step
            self.num_mel_bins = num_mel_bins
            self.lower_frequency = lower_frequency
            self.upper_frequency = upper_frequency
            self.num_coefficients = num_coefficients
            num_spectrogram_bins = (frame_length) // 2 + 1
    
            if mfcc is True:
                self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                        self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                        self.lower_frequency, self.upper_frequency)
                self.preprocess = self.preprocess_with_mfcc
            else:
                self.preprocess = self.preprocess_with_stft
    
        def read(self, file_path):
            parts = tf.strings.split(file_path, os.path.sep)
            label = parts[-2]
            label_id = tf.argmax(label == self.labels)
            audio_binary = tf.io.read_file(file_path)
            audio, _ = tf.audio.decode_wav(audio_binary)
            audio = tf.squeeze(audio, axis=1)
    
            return audio, label_id
    
        def pad(self, audio):
            zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
            audio = tf.concat([audio, zero_padding], 0)
            audio.set_shape([self.sampling_rate])
    
            return audio
    
        def get_spectrogram(self, audio):
            stft = tf.signal.stft(audio, frame_length=self.frame_length,
                    frame_step=self.frame_step, fft_length=self.frame_length)
            spectrogram = tf.abs(stft)
    
            return spectrogram
    
        def get_mfccs(self, spectrogram):
            mel_spectrogram = tf.tensordot(spectrogram,
                    self.linear_to_mel_weight_matrix, 1)
            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
            mfccs = mfccs[..., :self.num_coefficients]
    
            return mfccs
    
        def preprocess_with_stft(self, file_path):
            audio, label = self.read(file_path)
            audio = self.pad(audio)
            spectrogram = self.get_spectrogram(audio)
            spectrogram = tf.expand_dims(spectrogram, -1)
            spectrogram = tf.image.resize(spectrogram, [32, 32])
    
            return spectrogram, label
    
        def preprocess_with_mfcc(self, file_path):
            audio, label = self.read(file_path)
            audio = self.pad(audio)
            spectrogram = self.get_spectrogram(audio)
            mfccs = self.get_mfccs(spectrogram)
            mfccs = tf.expand_dims(mfccs, -1)
    
            return mfccs, label
    
        def make_dataset(self, files, train):
            ds = tf.data.Dataset.from_tensor_slices(files)
            ds = ds.map(self.preprocess, num_parallel_calls=4)
            ds = ds.batch(32)
            ds = ds.cache()
            if train is True:
                ds = ds.shuffle(100, reshuffle_each_iteration=True)
    
            return ds


class MyCooperative_client:
        def __init__(self, clientID, labels ):
            self.clientID = clientID
            # create an instance of paho.mqtt.client
            self._paho_mqtt = PahoMQTT.Client(clientID,False)

            # register the callback
            self._paho_mqtt.on_connect = self.myOnConnect
            self._paho_mqtt.on_message = self.myOnMessageReceived

            self.topic0 = "Istanza/"
            self.topic1 = "pred/1"
            self.topic2 = "pred/2"
            self.topic3 = "pred/4"
            self.messageBroker = 'mqtt.eclipseprojects.io'

            self.pred1 = []
            self.pred2 = []
            self.pred3 = []
            self.labels = labels

            self.running_corrects1 = 0
            self.running_corrects2 = 0
            self.running_corrects3 = 0
            #self.rec1 = 0
            #self.rec2 = 0
            #self.rec3 = 0
            self.invio = 0
            self.final_acc = 0



        def start (self):

            self._paho_mqtt.connect(self.messageBroker, 1883)
            self._paho_mqtt.loop_start()
            self._paho_mqtt.subscribe(self.topic0, 2)
            self._paho_mqtt.subscribe(self.topic1, 2)
            self._paho_mqtt.subscribe(self.topic2, 2)
            self._paho_mqtt.subscribe(self.topic3, 2)

        def stop (self):
            self._paho_mqtt.unsubscribe(self.topic0)
            self._paho_mqtt.unsubscribe(self.topic1)
            self._paho_mqtt.unsubscribe(self.topic2)
            self._paho_mqtt.unsubscribe(self.topic3)
            self._paho_mqtt.loop_stop()
            self._paho_mqtt.disconnect()
            
        def loop(self):
            self._paho_mqtt.connect(self.messageBroker, 1883)
            self._paho_mqtt.subscribe(self.topic0, 2)
            self._paho_mqtt.subscribe(self.topic1, 2)
            self._paho_mqtt.subscribe(self.topic2, 2)
            self._paho_mqtt.subscribe(self.topic3, 2)
            self._paho_mqtt.loop_forever()
            
  
        def myOnConnect (self, paho_mqtt, userdata, flags, rc):
            print ("Connected to %s with result code: %d" % (self.messageBroker, rc))


        def myOnMessageReceived (self, paho_mqtt , userdata, msg):
   
            if msg.topic == self.topic1:
                #print ("Topic:'" + msg.topic +"', QoS: '"+str(msg.qos)+"' Message: '    "+str(msg.payload) + "'")
                #print("Topic:{}; N:{}".format( msg.topic, self.rec1 ))
                
                str_msg = msg.payload.decode()
                dict_msg = json.loads(str_msg)

                self.flag1 = True
                preds = np.array(dict_msg['e'][0]['vd'])
                c_pred = np.argmax(preds)
                self.pred1.append(c_pred)
                #self.labels.append(int(dict_msg['label']))
                #self.rec1 += 1
                
                #if int(dict_msg['label']) == c_pred:
                #    self.running_corrects1 += 1
                
                
            elif msg.topic == self.topic2:
                #print("Topic:{}; N:{}".format( msg.topic, self.rec2 ))
                
                str_msg = msg.payload.decode()
                dict_msg = json.loads(str_msg)
                
                self.flag2 = True
                preds = np.array(dict_msg['e'][0]['vd'])
                c_pred = np.argmax(preds)
                self.pred2.append(c_pred)
                #self.rec2 += 1
                
                #if int(dict_msg['label']) == c_pred:
                #    self.running_corrects2 += 1
                    
                    
            elif msg.topic == self.topic3:
                #print("Topic:{}; N:{}".format( msg.topic, self.rec3 ))
                
                str_msg = msg.payload.decode()
                dict_msg = json.loads(str_msg)
                
                self.flag3 = True
                preds = np.array(dict_msg['e'][0]['vd'])
                c_pred = np.argmax(preds)
                self.pred3.append(c_pred)
                #self.rec3 += 1
                
                #print(dict_msg['label'])
                #print(c_pred)
                #if int(dict_msg['label']) == c_pred:
                    #print("dentro")
                #    self.running_corrects3 += 1


        def majority_vote(self):
        
            matrix = np.array(self.pred1)
            matrix = np.stack((matrix, self.pred2), axis=0)
            matrix = np.vstack((matrix, self.pred3))
            
            for x in range(matrix.shape[1] ):
                pred = max(dict(Counter(matrix[:,x])), key=dict(Counter(matrix[:,x])).get)
    
                if int(self.labels[x]) == pred :
                    self.final_acc += 1
          
                
        def init(self):
            self._paho_mqtt.connect(self.messageBroker, 1883)
            self._paho_mqtt.subscribe(self.topic0, 2)
            self._paho_mqtt.subscribe(self.topic1, 2)
            self._paho_mqtt.subscribe(self.topic2, 2)
            self._paho_mqtt.subscribe(self.topic3, 2)

        def loop(self):
            self._paho_mqtt.loop_forever()
                
            
        def myPublish(self, message):
            
            print("Sending sample n:{};".format( self.invio ))
            self._paho_mqtt.publish( self.topic0, message, 2 )

            self.invio += 1





if __name__ == '__main__':

    ## .1  ##
    ## DATASET PREPARATION ##
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    zip_path = tf.keras.utils.get_file(
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')

    data_dir = os.path.join('.', 'data', 'mini_speech_commands')

    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    test_files= []
    LABELS = []
    
    test_names = open("kws_test_split.txt", "r")
    test = test_names.readlines()
    for x in test:
        test_files.append(x[:-1])
    print(len(test_files))
    test_names.close()
    
    with open("labels.txt", "r") as lab_file:
        for line in lab_file:
            for word in line.split():
                LABELS.append(word)
    print(LABELS)

    MFCC_OPTIONS = {
        'frame_length': 640,
        'frame_step': 320,
        'mfcc': True,
        'lower_frequency': 20,
        'upper_frequency': 4000,
        'num_mel_bins': 40,
        'num_coefficients': 10
    }

    
    
    generator = SignalGenerator(LABELS, 16000, **MFCC_OPTIONS)
    test_ds = generator.make_dataset(test_files, False)
    print(type(test_ds))

    test_np = np.stack(list(test_ds))
    
    lab = []
    for l in range(test_np.shape[0]):
        lab.extend(list(test_np[l,1,:]))
    
    #print(len(lab))
    timestamp = datetime.timestamp( datetime.now() )
    device_name = 'raspberrypi'
    
    test = MyCooperative_client("Cooperative",lab)
    test.start()


    test_ds_tf = test_ds.unbatch().batch(1)

    for i, (mfcc, label) in enumerate(test_ds_tf):
            
        senml_msg = {
            "bn" : device_name,
            "bt" : timestamp,
            "e" :[
                {"n": "s", "u":"/", "t":0, "vd": mfcc.numpy().tolist() }
                ]
            }

        senml_msg = json.dumps(senml_msg)
        test.myPublish(senml_msg)
        time.sleep(0.1)
        
    time.sleep(4)
    

    test.majority_vote()
    total_samples = len(test_ds)*32
        
    '''
    print("Model 1")
    print(test.running_corrects1)
    print('Accuracy: %.3f %%'%( test.running_corrects1/total_samples*100))
    print("Model 2")
    print(test.running_corrects2)
    print('Accuracy: %.3f %%'%( test.running_corrects2/total_samples*100))
    print("Model 3")
    print(test.running_corrects3)
    print('Accuracy: %.3f %%'%( test.running_corrects3/total_samples*100))
    '''
    print("MAJORITY VOTE")
    #print(test.final_acc)
    print('Accuracy: %.3f %%'%( test.final_acc/total_samples*100))
    test.stop()
    print("Done")


