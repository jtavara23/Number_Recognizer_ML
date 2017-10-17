#Information extracted from https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/debugging/embedding.ipynb

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import scipy
import cPickle
%matplotlib inline
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
tf.__version__


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    data = dict['data']
    imgs = np.transpose(np.reshape(data,(-1,32,32,3), order='F'),axes=(0,2,1,3)) #order batch,x,y,color
    y = np.asarray(dict['labels'], dtype='uint8')
    return y, imgs
    
y, imgs = unpickle('/Users/oli/Dropbox/data/CIFAR-10/cifar-10-batches-py/test_batch')
y.shape, imgs.shape

tf.reset_default_graph()
images = tf.placeholder(tf.float32, [None, None, None, 3])
imgs_scaled = tf.image.resize_images(images, (224,224))
slim.nets.vgg.vgg_16(imgs_scaled, is_training=False)
variables_to_restore = slim.get_variables_to_restore()
print('Number of variables to restore {}'.format(len(variables_to_restore)))
init_assign_op, init_feed_dict = slim.assign_from_checkpoint('/Users/oli/Dropbox/server_sync/tf_slim_models/vgg_16.ckpt', variables_to_restore)
sess = tf.Session()
sess.run(init_assign_op, init_feed_dict)

g = tf.get_default_graph()
feed = g.get_tensor_by_name('Placeholder:0')
fetch = g.get_tensor_by_name('vgg_16/fc6/BiasAdd:0')

# Feeding 3 images through the net just for testing
feed_vals = imgs[0:3]
res = sess.run(fetch, feed_dict={feed:feed_vals})
np.shape(feed_vals), res.shape
