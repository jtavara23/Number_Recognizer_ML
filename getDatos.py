import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from funcionesAuxiliares import plot_conv_weights, plot_conv_layer,display
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
NOMBRE_TENSOR_ENTRADA = 'inputX'

path = '/media/josuetavara/Gaston/mnist/mnistDS/'
dataset = pd.read_csv(path+'datasets/10k.csv')

imagenes = dataset.iloc[:,1:].values
imagenes = imagenes.astype(np.float)

# Normalizar, convertir de [0:255] => [0.0:1.0]
#imagenes = np.multiply(imagenes, 1.0 / 255.0)
XX_train = []
for ii in xrange(0,len(imagenes)):
    nuevo = imagenes[ii].reshape((28,28,1))
    XX_train.append(nuevo)

imagenes = np.array(XX_train)

# Mostrar la imagen 25 del dataset 	
IMAGE_TO_DISPLAY = 0
imagenconv = imagenes[IMAGE_TO_DISPLAY]
#display(imagenconv)
#plt.show()
#"""
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.import_meta_graph(path + 'CNN/modelos2/modelo-2400.meta')
	ult_pto_ctrl = tf.train.latest_checkpoint(path + 'CNN/modelos2/.')
	saver.restore(sess, ult_pto_ctrl)
	print "Modelo restaurado",ult_pto_ctrl
	
	
	all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	#print all_vars
	#print "tensors: ",len(all_vars)
	#print sess.run(all_vars[0])[4][4]
	#for ind in xrange(0,len(all_vars)):
		#print ind,all_vars[ind]
		#print sess.run(all_vars[ind])
	#feed_dictcon = {NOMBRE_TENSOR_ENTRADA+":0": [imagenconv]}
	
	#plot_conv_weights(sess.run(all_vars[0]),0)
	plot_conv_weights(sess.run(all_vars[2]),0)
	
	
	
#"""
