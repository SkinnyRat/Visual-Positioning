# From https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/4xjc7tSrb18 
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import os, math, time
import cv2, csv
import numpy as np
import tensorflow as tf
import CIFAR10

from   datetime import datetime
from   PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


HOME   = '/HOME/'                               # /HOME/DATA/ 
width  = 24
height = 24

categories = []
with open(HOME  + "DATA/LABELS", 'r') as csvfile:
    Labels = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for L in Labels:
        categories.append(L)                    # L[0]

filename = HOME + "DATA/0000.png"
#im = Image.open(filename)
#im.save(filename, format='PNG', subsampling=0, quality=100)


with tf.Session() as sess:
    input_img     = tf.image.decode_png(tf.read_file(filename), channels=3)
    tf_cast       = tf.cast(input_img, tf.float32)
    float_image   = tf.image.resize_image_with_crop_or_pad(tf_cast, height, width)
    float_image   = tf.image.per_image_standardization(float_image)
    images        = tf.expand_dims(float_image, 0)

    logits        = CIFAR10.inference(images)
    _, top_k_pred = tf.nn.top_k(logits,  k=5)

    variable_averages    = tf.train.ExponentialMovingAverage(CIFAR10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)
    ckpt  = tf.train.get_checkpoint_state(HOME+'MODEL')

    if ckpt and ckpt.model_checkpoint_path:
        print("Model path = ", ckpt.model_checkpoint_path)
        saver.restore(sess,    ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')
        exit(0)

    #init_op = tf.initialize_all_variables() 
    #sess.run(init_op) 

    _, top_indices = sess.run([_, top_k_pred])
    for key, value in enumerate(top_indices[0]):
        print ("Type %20s" % categories[value] + "\t\t" + str(_[0][key]))

