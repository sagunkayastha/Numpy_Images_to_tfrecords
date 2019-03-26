""" Convert Numpy or image files to tfrecords"""


import numpy as np
import tensorflow as tf 
import os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
# FILEPATH = 'tf_data.tfrecords'


class TF_NPY:
    """ Convert Numpy or image files to tfrecords
        filename = filename for tfrecords file, default = demo
    """

    def __init__(self, filename= 'demo.tfrecords'):
        self.FILENAME = filename
        
    def _int64_feature(self,value):
        """ Helper function"""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self,value):
        """ Helper function"""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def numpy_to_tflearn(self, images, labels):

        """ Convert Numpy files to tfrecords"""

        with tf.python_io.TFRecordWriter(self.FILENAME) as writer:
            i=0
            for image, label in zip(images, labels):
                if i*10%len(images)==0:
                    perc = np.ceil((i+1)/len(images)*100 -1)
                    print(f'{int(perc)} %')
                feature = {'image':  self._bytes_feature(tf.compat.as_bytes(image.tostring())),
                        'label':  self._int64_feature(int(label))}

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                i+=1

    def folder_to_tfrecords(self,folder_path,label):

        """ Convert images files in folder tfrecords"""

        TRAIN_PATH = folder_path
        train_ids = next(os.walk(TRAIN_PATH))[2]

        with tf.python_io.TFRecordWriter(self.FILENAME) as writer:
            for i, (image, label) in enumerate(zip(train_ids, labels)):
                
                if i*10%len(train_ids)==0:
                    perc = np.ceil((i+1)/len(train_ids)*100-1 )
                    print(f'{int(perc)} %')
                image = plt.imread(TRAIN_PATH+image)
                feature = {'image':  self._bytes_feature(tf.compat.as_bytes(image.tostring())),
                            'label':  self._int64_feature(int(label))}

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            
obj = TF_NPY()       # Filename
obj.FILENAME = "test.tfrecords"


image = np.load('train_img.npy')
label = np.load('train_mask.npy')
            
# Numpy to tfrecords
obj.numpy_to_tflearn(image,label)

# Folder to tfrecords
labels = np.repeat(label,2)              # only for this example
obj.folder_to_tfrecords('images/',labels)