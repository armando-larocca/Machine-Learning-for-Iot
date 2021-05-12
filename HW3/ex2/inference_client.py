
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import paho.mqtt.client as PahoMQTT
import time
import tensorflow as tf
import json
import numpy as np
import argparse
import pathlib
import os



class MyInference_client:
        def __init__(self, clientID, model_path):
            self.clientID = clientID
            # create an instance of paho.mqtt.client
            self._paho_mqtt = PahoMQTT.Client(clientID,False)

            # register the callback
            self._paho_mqtt.on_connect = self.myOnConnect
            self._paho_mqtt.on_message = self.myOnMessageReceived

            self.topic1 = "Istanza/"
            self.topic2 = "pred/"+ model_path.split('.')[0][-1]
            self.messageBroker = 'mqtt.eclipseprojects.io'
            self.msg_inviati = 0

            self.interpreter = tf.lite.Interpreter(model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

        '''
        def start (self):
            #manage connection to broker
            self._paho_mqtt.connect(self.messageBroker, 1883)
            self._paho_mqtt.loop_start()
            # subscribe for a topic
            self._paho_mqtt.subscribe(self.topic1, 2)
            #self._paho_mqtt.subscribe(self.topic2, 2)

        def stop (self):
            #self._paho_mqtt.unsubscribe(self.topic)
            self._paho_mqtt.loop_stop()
            self._paho_mqtt.disconnect()
        '''
        
        def init(self):
            self._paho_mqtt.connect(self.messageBroker, 1883)
            self._paho_mqtt.subscribe(self.topic1, 2)
            self._paho_mqtt.subscribe(self.topic2, 2)

        def loop(self):
            self._paho_mqtt.loop_forever()

        def myOnConnect (self, paho_mqtt, userdata, flags, rc):
            print ("Connected to %s with result code: %d" % (self.messageBroker, rc))

        def myOnMessageReceived (self, paho_mqtt , userdata, msg):
   
            if msg.topic == self.topic1:

                print("Topic:{}; N:{}".format( msg.topic, self.msg_inviati))
                        
                str_msg = msg.payload.decode()
                dict_msg = json.loads(str_msg)
        
                mfcc = tf.convert_to_tensor(dict_msg['e'][0]['vd'])
        
                self.interpreter.set_tensor(self.input_details[0]['index'], mfcc)
                self.interpreter.invoke()
                logits = self.interpreter.get_tensor(self.output_details[0]['index'])
                #probs = tf.nn.softmax(logits)
                probs = np.array(logits)
        
                label = dict_msg['e'][1]['vd']
        
                # make out dictionay
                #to_rasp_dict = {'probs': probs.numpy().squeeze().tolist(), 'label':label}
                to_rasp_dict = {'probs': probs.squeeze().tolist(), 'label':label}
                json_to_rasp = json.dumps(to_rasp_dict)
                
                self.msg_inviati += 1
                self._paho_mqtt.publish(self.topic2, json_to_rasp, 2)
                


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model path')
    args = parser.parse_args()
    model_path = args.model
    
    test = MyInference_client("Inference"+ model_path.split('.')[0][-1], model_path)
    print(test.topic1, test.topic2)

    test.init()
    test.loop()
    
   
