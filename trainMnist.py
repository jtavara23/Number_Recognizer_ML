import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
from datetime import timedelta
from matplotlib import pyplot as plt

from funcionesAuxiliares import  readData,display,activation_vector,plot_confusion_matrix,plot_example_errors,plot_conv_layer

from subprocess import check_output
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


rutaModelo = '/media/josuetavara/Gaston/mnist/mnistDS/CNN/modelos2/'
#tensorboard --logdir /media/josuetavara/Gaston/mnist/mnistDS/CNN/modelos2

#clases a clasificar
CANT_CLASES = 10

#TASA_APRENDIZAJE = 5e-4  #1,2,3ra epoca
#TASA_APRENDIZAJE = 3e-4  #4ta,5ta epoca
TASA_APRENDIZAJE = 1e-4  #7ta-10ma epoca

# entrenamiento es ejecutado seleccionando subconjuntos de imagenes
BATCH_SIZE = 400

#cada con 400 de batch, en 300 iteraciones se completa una epoca
ITERACIONES_ENTRENAMIENTO = 300*10+20

CHKP_GUARDAR_MODELO = 300 #cada 100 iteraciones
CHKP_REVISAR_PROGRESO = 2 #10 iteraciones||50 iter apatir de [600+]

def inicializar_pesos(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	w = tf.Variable(initial, name = "W")
	tf.summary.histogram("pesos", w)
	return w


def inicializar_bias(shape):
	initial = tf.constant(0.1, shape=shape)
	b = tf.Variable(initial, name = "B")
	tf.summary.histogram("biases", b)
	return b 

def conv_layer(nombre, entrada,num_inp_channels,filter_size, num_filters, use_pooling):
	with tf.name_scope(nombre):
		forma  = [filter_size, filter_size, num_inp_channels,num_filters]
		pesos  = inicializar_pesos(shape=forma)
		biases = inicializar_bias([num_filters])
		convolucion  = tf.nn.conv2d(input = entrada,
		                    filter = pesos,
		                    strides = [1,1,1,1],
		                    padding = 'SAME')

		convolucion += biases
		#puede ser despues de pooling
		convolucion1 = tf.nn.relu(convolucion)

		if use_pooling:
			convolucion2 = tf.nn.max_pool(value=convolucion1,
			                            ksize=[1, 2, 2, 1],
			                            strides=[1, 2, 2, 1],
			                            padding='SAME')
		#print nombre,": ",convolucion.get_shape(),"\n***********"
	return convolucion1,convolucion2, pesos

def flatten_layer(layer):
	layer_shape = layer.get_shape()
	#print "Layer shape: ", layer_shape
	num_features = layer_shape[1:4].num_elements()
	#print "num_features: ", num_features
	layer_flat = tf.reshape(layer, [-1, num_features])
	#print "layer_flat: : ", layer_flat

	return layer_flat, num_features

def capa_fc(nombre,entrada, num_inputs, num_outputs,use_relu=True): 
	with tf.name_scope(nombre):
		#print entrada.get_shape()," multiplica ",num_inputs,num_outputs
		pesos = inicializar_pesos(shape=[num_inputs, num_outputs])
		biases = inicializar_bias([num_outputs])
		layer = tf.matmul(entrada, pesos) + biases

		if use_relu:
			"Uso relu"
			layer = tf.nn.relu(layer)
		else:
			"No uso Relu"

	return layer,pesos


def create_cnn(sess):
	global IMAGE_SHAPE #(28, 28, 1)
	
	#Los tensores placeholder sirven como entrada al grafico computacional de TensorFlow que podemos cambiar cada vez que ejecutamos el grafico.
	#None significa que el tensor puede contener un numero arbitrario de imagenes , donde cada imagen es un vector de longitud dada
	x = tf.placeholder('float', shape=[None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]],name= NOMBRE_TENSOR_ENTRADA)
	# clases
	y_deseada = tf.placeholder('float', shape=[None, CANT_CLASES],name= NOMBRE_TENSOR_SALIDA_DESEADA)

	#las capas de convolucion esperan que las entradas sean encodificadas en tensores de 4D
	#print (x.get_shape()) # =>(?, 28, 28, 1) [?=120000]

	#------------------- Primera capa convolucional-------------------
	tam_filtro1 = 5
	num_filtro1 = 32
	nopool_conv1,capa_conv1, pesos_conv1 = conv_layer(nombre="convolucion1",entrada=x,
	                                            num_inp_channels=1,
	                                            filter_size=tam_filtro1,
	                                            num_filters=num_filtro1,
	                                            use_pooling=True)	
	#print(capa_conv1.get_shape()) # => (60000, 14, 14, 32)

	# -------------------Segunda capa convolucional-------------------
	tam_filtro2 = 5
	num_filtro2 = 64
	nopool_conv2,capa_conv2, pesos_conv2 = conv_layer(nombre="convolucion2",entrada=capa_conv1,
                                            num_inp_channels=num_filtro1,
                                            filter_size=tam_filtro2,
                                            num_filters=num_filtro2,
                                            use_pooling=True)
	#print (capa_conv2.get_shape()) # => (60000, 7, 7, 64)

	# -------------Capa totalmente conectada-------------------
	layer_flat1, num_fc_layers1 = flatten_layer(capa_conv1)#14*14*32
	layer_flat2, num_fc_layers2 = flatten_layer(capa_conv2)#7**7*64
	layer_flat = tf.concat([layer_flat1, layer_flat2], 1)
	#print "1:",layer_flat.get_shape()#(?, 9408)
	num_fc_layers = num_fc_layers1 + num_fc_layers2
	#print "2:",num_fc_layers #(9408)

	capa_fc1, pesos_fc1 = capa_fc(nombre = "FC1", entrada=layer_flat,
	                                 num_inputs = num_fc_layers,
	                                 num_outputs = 1024, use_relu=True)
	"""
	with tf.name_scope("FC1"):
		#Para esta capa se necesita un tensor de 2 dimensiones(Entradas, Salidas)
		forma = [7 * 7 * 64, 1024]
		pesos_fc1 = inicializar_pesos(shape = forma)
		biases_fc1 = inicializar_bias([1024])

		pool_conv2_flat = tf.reshape(pool_conv2, [-1, 7*7*64])
		# (60000, 7, 7, 64) => (60000, 3136)
		
		# Multiplicar matriz 'pool_conv2_flat' por matriz 'pesos_fc1' y su_entrenr bias
		fc1 = tf.matmul(pool_conv2_flat, pesos_fc1) + biases_fc1
		#print (fc1.get_shape()) # => (60000, 1024)
		
		act_fc1 = tf.nn.relu(fc1)
	"""
	#Aplicarmos dropout entre la capa FC y la capa de salida
	keep_prob = tf.placeholder('float',name=NOMBRE_PROBABILIDAD)
	fc_capa1_drop = tf.nn.dropout(capa_fc1, keep_prob)
	
	embedding_input = capa_fc1
	#------------------- Capa de Salida-------------------------
	capa_fc2, pesos_fc2 = capa_fc(nombre = "FC2", entrada=fc_capa1_drop,
                                  num_inputs = 1024,
                                  num_outputs = CANT_CLASES, use_relu=False)
	"""
	with tf.name_scope("FC2"):
		pesos_fc2 = inicializar_pesos([1024, CANT_CLASES])
		biases_fc2 = inicializar_bias([CANT_CLASES])

		fc2 = tf.matmul(h_fc1_drop, pesos_fc2) + biases_fc2
	"""
	
	y_calculada = tf.nn.softmax(capa_fc2, name = NOMBRE_TENSOR_SALIDA_CALCULADA)
	#print (y_calculada.get_shape()) # => (60000, 10)
	#[0.01, 0.04, 0.02, 0.5, 0.03 0.01, 0.05, 0.02, 0.3, 0.02] => 3
	predictor = tf.argmax(y_calculada,dimension = 1)
	tf.add_to_collection("predictor", predictor)
	
	#------------------------------------------------------------
	
	return resumen, x,y_deseada, keep_prob, internal_conv, iterac_entren, optimizador,acierto, predictor,embedding_input


if __name__ == "__main__":
	
	print "Inicio de lectura del dataset"
	path = '/media/josuetavara/Gaston/mnist/mnistDS/'
	datasetEntrenamiento = (path+'datasets/120kExtendedShuffle.p')
	datasetEvaluacion = (path+'datasets/10k.p')

	#--------------------------------PREPROCESAMIENTO DE DATOS-----------------------------------------
	print "Procesamiento de las imagenes" 
	IMAGE_SHAPE, entrenam_imagenes, entrenam_clases, entrenam_clases_flat = procesamiento(datasetEntrenamiento)
	IMAGE_SHAPE, eval_imagenes, eval_clases, eval_clases_flat = procesamiento(datasetEvaluacion)
	
	imageConvol = entrenam_imagenes[27]	
	#display(imageConvol)
	#--------------------------------CREACION DE LA RED------------------------------------------------
	print "Inicio de creacion de la red"
	tf.reset_default_graph()
	sess = tf.Session()
	resumen, x, y_deseada, keep_prob,  internal_conv, iterac_entren, optimizador, acierto, predictor, embedding_input=  create_cnn(sess)
	
	#"""
	#--------------------------------ENTRENAMIENTO DE LA RED------------------------------------------------
	
	sess.run(tf.global_variables_initializer())

	#
	entren_writer = tf.summary.FileWriter(rutaModelo+'entrenamiento11', sess.graph)
	#entren_writer.add_graph(sess.graph)
	evalua_writer = tf.summary.FileWriter(rutaModelo+'evaluacion11', sess.graph)

	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state(rutaModelo + '.')
	
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print("Sesion restaurada de: %s" % ckpt.model_checkpoint_path)

	else:
		print("No se encontro puntos de control.")
		
	
	ultima_iteracion = iterac_entren.eval(sess)
	print "Ultimo modelo en la iteracion: ", ultima_iteracion
	
	
	#Empieza el entrenamiento desde la ultima iteracion hasta el ITERACIONES_ENTRENAMIENTO dado 
	


	
	
