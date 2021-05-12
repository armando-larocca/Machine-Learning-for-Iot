import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='version constraint')
args = parser.parse_args()

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

train_files= []
val_files = []
test_files = []
LABELS = []

train_names = open("kws_train_split.txt", "r")
train = train_names.readlines()
for x in train:
    train_files.append(x[:-1])
print(len(train_files))
train_names.close()

val_names = open("kws_val_split.txt", "r")
val = val_names.readlines()
for x in val:
    val_files.append(x[:-1])
print(len(val_files))
val_names.close()

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

## .2 ##
## CLASS ##
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


#if args.version == "1":
MFCC_OPTIONS = {'frame_length': 640,
                'frame_step': 320,
                'mfcc': True,
                'lower_frequency': 20,
                'upper_frequency': 4000,
                'num_mel_bins': 40,
                'num_coefficients': 10
                }
'''
elif args.version == "1":
    MFCC_OPTIONS = {'frame_length':1000,
                'frame_step': 600,
                'mfcc': True,
                'lower_frequency': 20,
                'upper_frequency': 4000,
                'num_mel_bins': 40,
                'num_coefficients': 10
                }
'''

generator = SignalGenerator(LABELS, 16000, **MFCC_OPTIONS)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)


if args.version == "1" :
    filters = [ 200,90,64]
    d_out = 0.4
    saved_model_dir = "model_1"
    tflite_model_dir = "1.tflite"
    tflite_compressed_dir = "1.tflite.zlib"
    ep = 24
    sp1 = [32,49,10,1]
    sp2 = [1,49,10,1]
    init_lr = 0.01
    
    def scheduler(epoch, lr):
        if epoch < 5 :
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=filters[0], kernel_size=[3, 3], strides=[2,1], use_bias=False, activation='relu') ,
        tf.keras.layers.BatchNormalization(momentum=0.1),

        tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False, activation='relu') ,
        tf.keras.layers.Conv2D(filters=filters[1], kernel_size=[1, 1], strides=[1, 1], use_bias=False, activation='relu'),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        
        tf.keras.layers.Dropout(d_out),
        tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False, activation='relu') ,
        tf.keras.layers.Conv2D(filters=filters[2], kernel_size=[1, 1], strides=[1, 1], use_bias=False, activation='relu'),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        
        tf.keras.layers.Dropout(d_out),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=8),
    ])
    
    

elif args.version == "2":
    filters = [ 256, 256, 512, 1024, 1024]
    d_out = 0.4
    saved_model_dir = "2_big"
    tflite_model_dir = "2.tflite"
    tflite_compressed_dir = "2.tflite.zlib"
    ep = 10
    sp1 = [32,49,10,1]
    sp2 = [1,49,10,1]
    init_lr = 0.001
    
    def scheduler(epoch, lr):
        if (epoch%5) != 0  or epoch == 0:
            return lr
        else:
            return lr * 0.2
            
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=filters[0], kernel_size=[3, 3], strides=[2,1], use_bias=False, activation='relu') ,
        tf.keras.layers.BatchNormalization(momentum=0.1),

        tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False, activation='relu') ,
        tf.keras.layers.Conv2D(filters=filters[1], kernel_size=[1, 1], strides=[1, 1], use_bias=False, activation='relu'),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        tf.keras.layers.Conv2D(filters=filters[1], kernel_size=[1, 1], strides=[1, 1], use_bias=False, activation='relu'),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        
        #tf.keras.layers.Dropout(d_out),
        tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False, activation='relu') ,
        tf.keras.layers.Conv2D(filters=filters[2], kernel_size=[1, 1], strides=[1, 1], use_bias=False, activation='relu'),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False, activation='relu') ,
        tf.keras.layers.Conv2D(filters=filters[3], kernel_size=[1, 1], strides=[1, 1], use_bias=False, activation='relu'),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        
        #tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(d_out),
        tf.keras.layers.Dense(units=8),
    ])
    
    
elif args.version == "3":
    filters = [ 2048, 1024]
    d_out = 0.4
    saved_model_dir = "3_big"
    tflite_model_dir = "3.tflite"
    tflite_compressed_dir = "3.tflite.zlib"
    ep = 11
    sp1 = [32,49,10,1]
    sp2 = [1,49,10,1]
    init_lr = 0.001
    
    def scheduler(epoch, lr):
        if epoch < 5 :
            return lr
        else:
            return lr * tf.math.exp(-0.1)
            
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=filters[0], kernel_size=[3, 3], strides=[2,1], use_bias=False, activation='relu') ,
        tf.keras.layers.BatchNormalization(momentum=0.1),
        
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None),

        tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False, activation='relu') ,
        tf.keras.layers.Conv2D(filters=filters[1], kernel_size=[1, 1], strides=[1, 1], use_bias=False, activation='relu'),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None),

                
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=256),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=8),
    ])
    
elif args.version == "4":
    filters = [256, 512, 1024]
    d_out = 0.3
    saved_model_dir = "4_big"
    tflite_model_dir = "4.tflite"
    tflite_compressed_dir = "4.tflite.zlib"
    ep = 8
    sp1 = [32,49,10,1]
    sp2 = [1,49,10,1]
    init_lr = 0.001
    
    def scheduler(epoch, lr):
        if (epoch%5) != 0  or epoch == 0:
            return lr
        else:
            return lr * 0.2
            
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=filters[0], kernel_size=[3, 3], strides=[2,1], use_bias=False, activation='relu') ,
        tf.keras.layers.BatchNormalization(momentum=0.1),

        tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False, activation='relu') ,
        tf.keras.layers.Conv2D(filters=filters[1], kernel_size=[1, 1], strides=[1, 1], use_bias=False, activation='relu'),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        
        tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False, activation='relu') ,
        tf.keras.layers.Conv2D(filters=filters[2], kernel_size=[1, 1], strides=[1, 1], use_bias=False, activation='relu'),
        tf.keras.layers.BatchNormalization(momentum=0.1),
        
        tf.keras.layers.Dropout(d_out),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=256),
        tf.keras.layers.Dropout(d_out),
        tf.keras.layers.Dense(units=64),
        tf.keras.layers.Dropout(d_out),
        tf.keras.layers.Dense(units=8),
    ])
  

# Training
opt = tf.keras.optimizers.Adam(learning_rate= init_lr)

model.compile(optimizer=opt,
                loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
lr_callback = tf.keras.callbacks.LearningRateScheduler( scheduler ,verbose=1 )
 
model.build(sp1)
cb = [es_callback, lr_callback]

history = model.fit(x = train_ds,
                    validation_data = val_ds,
                    epochs= ep,
                    callbacks=cb )


# Evalutation
loss,metric = model.evaluate(x=test_ds)
print("Accuracy: ",metric)


# Tflite model generation
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec(sp2 ,tf.float32))
model.save(saved_model_dir, signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

#if post_q == True :
#    converter.optimizations = [tf.lite.Optimize.DEFAULT]
   
tflite_model = converter.convert()

with open(tflite_model_dir, 'wb') as fp:
    fp.write(tflite_model)
    
print( "TFLite Dimension:{:.2f} kB".format(os.path.getsize(tflite_model_dir)/ 2**10)  )


# TFLITE Accuracy evaluation
interpreter = tf.lite.Interpreter(model_path= tflite_model_dir)
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test_ds_tf = test_ds.unbatch().batch(1)
numerator = 0
count = 0

for x, y in test_ds_tf:
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    y_pred = y_pred.squeeze()
    y_pred = np.argmax(y_pred)
    y = y.numpy().squeeze()
    
    if y_pred == y :
        numerator += 1
    count += 1
    
acc = numerator / count
print("Accuracy tflite model:", acc)

'''
#Zip part
import zlib
tflite_model = converter.convert()
with open(tflite_compressed_dir, 'wb') as fp:
    tflite_compressed = zlib.compress(tflite_model)
    fp.write(tflite_compressed)

print( "TFLite (Compressed) Dimension:{:.2f} kB".format(os.path.getsize(tflite_compressed_dir)/ 2**10)  )
'''
