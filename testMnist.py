import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import tensorflow as tf
from funcionesAuxiliares import activation_vector, plot_example_errors,plot_confusion_matrix
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
BATCH_SIZE = 200
NOMBRE_TENSOR_ENTRADA = 'inputX'
NOMBRE_TENSOR_SALIDA_DESEADA = "outputYDeseada"
NOMBRE_PROBABILIDAD = 'mantener_probabilidad'

if __name__ == "__main__":
	path = '/media/josuetavara/Gaston/mnist/mnistDS/'
	dataset = pd.read_csv(path+'datasets/10ktest.csv')
	imagenes = dataset.iloc[:,1:].values
	imagenes = imagenes.astype(np.float)

	
	# Normalizar, convertir de [0:255] => [0.0:1.0]
	imagenes = np.multiply(imagenes, 1.0 / 255.0)
	#Organizar las clases de las imagenes en un solo vector
	clases_flat = dataset.iloc[:,0].values

	numero_clases = np.unique(clases_flat).shape[0]
	#print('number of labes => {0}'.format(numero_clases))
	
	# convertir tipo de clases de escalares a vectores de activacion de 1s
	# 0 => [1 0 0 0 0 0 0 0 0 0]
	# 1 => [0 1 0 0 0 0 0 0 0 0]
	# ...
	# 9 => [0 0 0 0 0 0 0 0 0 1]
	labels = activation_vector(clases_flat, numero_clases)
	labels = labels.astype(np.uint8)

	# Restauramos el ultimo punto de control
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.import_meta_graph(path + 'CNN/models/model-280.meta')
		saver.restore(sess, tf.train.latest_checkpoint(path + 'CNN/models/.'))
		print "Modelo restaurado",tf.train.latest_checkpoint(path + 'CNN/models/.')
		
		
		#Tensor predictor para clasificar la imagen
		predictor = tf.get_collection("predictor")[0]
		#cantidad de imagenes a clasificar
		cant_evaluar = (imagenes.shape[0])
		
		clases_pred = np.zeros(shape=cant_evaluar, dtype=np.int)

		
		start = 0
		print "Prediciendo clases..."
		while start < cant_evaluar:
			end = min(start + BATCH_SIZE, cant_evaluar)
			
			images_evaluar = imagenes[start:end, :]
			clases_evaluar = labels[start:end, :]

			#Introduce los datos para ser usados en un tensor
			feed_dictx = {NOMBRE_TENSOR_ENTRADA+":0": images_evaluar, NOMBRE_TENSOR_SALIDA_DESEADA+":0": clases_evaluar,NOMBRE_PROBABILIDAD+":0":1.0}

			# Calcula la clase predecida , atraves del tensor predictor
			clases_pred[start:end] = sess.run(predictor, feed_dict=feed_dictx)
			
			# Asigna el indice final del batch actual
			# como comienzo para el siguiente batch 
			start = end

		# Convenience variable for the true class-numbers of the test-set.
		clases_deseadas = clases_flat
		
		# Cree una matriz booleana
		correct = (clases_deseadas == clases_pred)
		
		# Se calcula el numero de imagenes correctamente clasificadas.
		correct_sum = correct.sum()

		# La precision de la clasificacion es el numero de imgs clasificadas correctamente
		acc = float(correct_sum) / cant_evaluar

		msg = "Acierto en el conjunto de Testing: {0:.1%} ({1} / {2})"
		print(msg.format(acc, correct_sum, cant_evaluar))

		# Muestra algunas imagenes que no fueron clasificadas correctamente
		#plot_example_errors(clases_pred=clases_pred, correct=correct,imagenes = imagenes, clases_flat=clases_flat)

		print("Mostrando Matriz de Confusion")
		plot_confusion_matrix(clases_pred, clases_deseadas,numero_clases)
		plt.show()
		
		print "Fin de evaluacion"
		outFile = open("Test_ac.csv","a")
		outFile.write(repr(tf.train.latest_checkpoint(path + 'CNN/models/.'))+"\n")
		outFile.write(msg.format(acc, correct_sum, cant_evaluar))
		outFile.write("\n-------------------------------------------------------------\n")
