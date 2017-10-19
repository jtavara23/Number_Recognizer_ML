import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from funcionesAuxiliares import plot_conv_weights, plot_conv_layer,display
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

path = '/media/josuetavara/Gaston/signals/models5/'

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.import_meta_graph(path + 'model-49040.meta')
	saver.restore(sess, tf.train.latest_checkpoint(path + '.'))
	print "Modelo restaurado",tf.train.latest_checkpoint(path + '.')
	
	
	all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	#print "tensors: ",len(all_vars)
	#print sess.run(all_vars[0])[4][4]
	#print sess.run(all_vars[8])[1023]
	
	for ind in xrange(0,len(all_vars)):
		print ind,all_vars[ind]
		#print sess.run(all_vars[ind])
	
	#plot_conv_weights(sess.run(all_vars[0]),8,4,0)
	#plot_conv_weights(sess.run(all_vars[2]),8,8,0)
#"""
