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
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

input_width = 6
output_width = 6

## .2 ##
## CLASS ##


class WindowGenerator:
    def __init__(self, input_width, output_width, mean, std):
        self.input_width = input_width
        self.output_width = output_width
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

        
    def split_window(self, features):
        inputs = features[:, :-self.output_width, :]
        labels = features[:, self.output_width:, :]

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, self.output_width, 2])
        
        return inputs, labels

    
    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)
        return features

    
    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)
        return inputs, labels

    
    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.input_width+self.output_width,
                sequence_stride=1,
                batch_size=64)
        
        #for batch in ds:
        #    inputs = batch
            #print(inputs.shape)
            #print(inputs,"\n")
        
        ds = ds.map(self.preprocess)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds
    
    
    
generator = WindowGenerator(input_width, output_width , mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)

tf.data.experimental.save(test_ds, './th_test')

class MultiOutputMAE(tf.keras.metrics.Metric):

    def __init__(self, name='mean_absolute_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros', shape=[2])
        self.count = self.add_weight(name='count', initializer='zeros')
        

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=[0,1])
        #print(error.shape)
        self.total.assign_add(error)
        self.count.assign_add(1)
        

    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))

        
    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)


def scheduler(epoch, lr):
    if epoch < 5 :
        return lr
    else:
        return lr * tf.math.exp(-0.1)


if args.version == "a":
    n_c = 18
    ep_c = 5
    saved_model_dir = "model_a"
    tflite_model_dir = "Group99_th_a.tflite"
    tflite_compressed_dir = "Group99_th_a.tflite.zlib"


else :
    n_c = 11
    ep_c = 10
    saved_model_dir = "model_b"
    tflite_model_dir = "Group99_th_b.tflite"
    tflite_compressed_dir = "Group99_th_b.tflite.zlib"


model = tf.keras.Sequential( [
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units= 50, activation='relu'),
    tf.keras.layers.Dense(units=12),
    tf.keras.layers.Reshape((6, 2))
  ])


# Training
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=opt,
                loss= tf.keras.losses.MeanSquaredError(),
                metrics=[MultiOutputMAE()])

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
lr_callback = tf.keras.callbacks.LearningRateScheduler( scheduler ,verbose=0 )
 

history = model.fit(x = train_ds, validation_data = val_ds, epochs=6,
                    callbacks=[es_callback, lr_callback,
                              ])

loss,metric = model.evaluate(x=test_ds)
print("Test MAE",metric)

# Weight clustering optimization
print("\nWEIGHT CLUSTERING")
model = tfmot.clustering.keras.cluster_weights(
                    model,
                    number_of_clusters= n_c,
                    cluster_centroids_init = tfmot.clustering.keras.CentroidInitialization.LINEAR )
                        
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0005),
                loss= tf.keras.losses.MeanSquaredError(),
                metrics=[MultiOutputMAE()])
                
model.fit(x = train_ds,
                    validation_data = val_ds,
                    epochs= ep_c
                )


# Evalutation Keras Model
loss,metric = model.evaluate(x=test_ds)
print("Test MAE",metric)

model = tfmot.clustering.keras.strip_clustering(model)

run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2],tf.float32))
model.save(saved_model_dir, signatures=concrete_func)

#def representative_dataset_gen():
#    for x, _ in train_ds.take(50):
#        yield [x]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

##POST QUANTIZATION ##
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()

with open(tflite_model_dir, 'wb') as fp:
    fp.write(tflite_model)
    
print( "TFLite Dimension:{:.2f} kB".format(os.path.getsize(tflite_model_dir)/ 2**10)  )


#Evaluation tflite performances
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
    y = y.numpy().squeeze()
    
    numerator += np.abs(y- y_pred)
    count += 1

numerator = np.mean(numerator,axis=0)
mae = numerator / count
print("MAE tflite model:", mae)


#ZIP model
import zlib
tflite_model = converter.convert()
with open(tflite_compressed_dir, 'wb') as fp:
    tflite_compressed = zlib.compress(tflite_model)
    fp.write(tflite_compressed)

print( "TFLite Dimension:{:.2f} kB".format(os.path.getsize(tflite_compressed_dir)/ 2**10)  )

