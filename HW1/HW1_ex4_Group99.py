import numpy as np
import tensorflow as tf
import csv
import time
import datetime
from scipy.io import wavfile
import argparse
import os

def get_directory_size(directory):
    """Returns the `directory` size in bytes."""
    total = 0
    try:
        # print("[+] Getting the size of", directory)
        for entry in os.scandir(directory):
            if entry.is_file():
                # if it's a file, use stat() function
                total += entry.stat().st_size
            elif entry.is_dir():
                # if it's a directory, recursively call this function
                total += get_directory_size(entry.path)
    except NotADirectoryError:
        # if `directory` isn't a directory, get the file size then
        return os.path.getsize(directory)
    except PermissionError:
        # if for whatever reason we can't open the folder, return 0
        return 0
    return total


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="input directory")
parser.add_argument("--output", type=str, help="output name")
args = parser.parse_args()

print(args.input)
print(args.output)

#output = './numbers.tfrecord'
#csv_file = open('./raw_data/samples.csv')
csv_file = open(args.input + "/samples.csv")
csv_reader = csv.reader(csv_file, delimiter=',')

posix = []
temp = []
hum = []
audio = []

for x in csv_reader :
    posix.append(int(time.mktime(datetime.datetime.strptime( \
                    str(x[0])+","+str(x[1]), '%Y-%m-%d,%I:%M:%S').timetuple()) ))
                    
    temp.append(float(x[2]))
    hum.append(float(x[3]))
    audio.append (np.ravel( tf.io.read_file( args.input + "/" + x[4] )))

    
print("audio l:",len(audio))
print(len(audio[1]))
    
with tf.io.TFRecordWriter(args.output) as writer:

    for i in range(0,len(posix)):
        posix_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[posix[i]]))
        temp_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[temp[i]]))
        hum_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[hum[i]]))
        audio_feature = tf.train.Feature(bytes_list =tf.train.BytesList(value=audio[i] ) )

        mapping = {'datetime': posix_feature, 'temperature' : temp_feature,\
                   'humidity' : hum_feature, 'audio': audio_feature }
        
        example = tf.train.Example(features=tf.train.Features(feature=mapping))
        writer.write(example.SerializeToString())
        
#input_size = os.path.getsize("raw_data") #/ 2.**10
input_size = get_directory_size("raw_data")
output_size = os.path.getsize(args.output) #/ 2.**10

print("input size: {:.2f}KB".format(input_size))
print("output size: {:.2f}KB".format(output_size))

raw_dataset = tf.data.TFRecordDataset(args.output)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
