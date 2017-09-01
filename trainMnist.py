import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
from datetime import timedelta
from matplotlib import pyplot as plt

from funcionesAuxiliares import  display,activation_vector,plot_confusion_matrix,plot_example_errors

from subprocess import check_output
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

NOMBRE_MODELO = 'model'
NOMBRE_TENSOR_ENTRADA = 'inputX'
NOMBRE_PROBABILIDAD = 'mantener_probabilidad'
NOMBRE_TENSOR_SALIDA_CALCULADA = 'outputYCalculada'
NOMBRE_TENSOR_SALIDA_DESEADA = "outputYDeseada"



#clases a clasificar
CANT_CLASES = 10
#TASA_APRENDIZAJE = 5e-4  #1ra epoca
#TASA_APRENDIZAJE = 3e-4  #2da epoca
TASA_APRENDIZAJE = 1e-4  #3ra epoca

# entrenamiento es ejecutado seleccionando subconjuntos de imagenes
BATCH_SIZE = 200

#cada con 200 de batch, en 300 iteraciones se completa una epoca
ITERACIONES_ENTRENAMIENTO = 1200 

CHKP_GUARDAR_MODELO = 100 #cada 100 iteraciones
CHKP_REVISAR_PROGRESO = 25 #iteraciones

DROPOUT = 0.5




def siguiente_batch_entren(batch_size,cant_imag_entrenamiento ):
    

	#cant_imag_entrenamiento = entrenam_imagenes.shape[0] = 60000
	global entrenam_imagenes
	global entrenam_clases
	global entrenam_clases_flat
	global indice_en_epoca
	global epocas_completadas

	comienzo = indice_en_epoca
	indice_en_epoca += batch_size

	# Cuando ya se han utilizado todos los datos de entrenamiento, se reordena aleatoriamente.
	if indice_en_epoca > cant_imag_entrenamiento:
		# epoca finalizada
		epocas_completadas += 1
		# barajear los datos
		perm = np.arange(cant_imag_entrenamiento)
		np.random.shuffle(perm)
		#perm = stratified_shuffle(entrenam_clases_flat, 10)
		entrenam_imagenes = entrenam_imagenes[perm]
		entrenam_clases = entrenam_clases[perm]
		entrenam_clases_flat = entrenam_clases_flat[perm]
		# comenzar nueva epoca
		comienzo = 0
		indice_en_epoca = batch_size
		assert batch_size <= cant_imag_entrenamiento
	end = indice_en_epoca
	return entrenam_imagenes[comienzo:end], entrenam_clases[comienzo:end]





def procesamiento(dataset):
	imagenes = dataset.iloc[:,1:].values
	imagenes = imagenes.astype(np.float)

	# Normalizar, convertir de [0:255] => [0.0:1.0]
	imagenes = np.multiply(imagenes, 1.0 / 255.0)
	
	#Tamanho de una imagen: 784 valores que son obtenidos de una imagen de 28 x 28
	tam_imagen = imagenes.shape[1]
	#print ('tam_imagen => {0}'.format(tam_imagen))

	#print ('anchura_imagen => {0}\naltura_imagen => {1}'.format(anchura_imagen,altura_imagen))
	
	# Mostrar la imagen 25 del dataset 	IMAGE_TO_DISPLAY = 25
	#display(imagenes[IMAGE_TO_DISPLAY])

	#Organizar las clases de las imagenes en un solo vector
	clases_flat = dataset.iloc[:,0].values.ravel()
	#Imprimir informacion de la CLASE de 'imagen-to-display' 
	#print ('label of imagen [{0}] => {1}'.format(IMAGE_TO_DISPLAY,clases_flat[IMAGE_TO_DISPLAY]))

	# convertir tipo de clases de escalares a vectores de activacion de 1s
	# 0 => [1 0 0 0 0 0 0 0 0 0]
	# 1 => [0 1 0 0 0 0 0 0 0 0]
	# ...
	# 9 => [0 0 0 0 0 0 0 0 0 1]
	clases = activation_vector(clases_flat, CANT_CLASES)
	clases = clases.astype(np.uint8)
	
	return tam_imagen, imagenes[:], clases[:], clases_flat[:]
	
		
	

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


def create_cnn(sess):
	global tam_imagen #784
	global anchura_imagen#28
	global altura_imagen#28
	
	#Los tensores placeholder sirven como entrada al grafico computacional de TensorFlow que podemos cambiar cada vez que ejecutamos el grafico.
	#None significa que el tensor puede contener un numero arbitrario de imagenes , donde cada imagen es un vector de longitud dada
	x = tf.placeholder('float', shape=[None, tam_imagen],name= NOMBRE_TENSOR_ENTRADA)
	# clases
	y_deseada = tf.placeholder('float', shape=[None, CANT_CLASES],name= NOMBRE_TENSOR_SALIDA_DESEADA)

	#las capas de convolucion esperan que las entradas sean encodificadas en tensores de 4D
	imagen = tf.reshape(x, [-1,altura_imagen, anchura_imagen,1])
	#print (imagen.get_shape()) # =>(60000,28,28,1)

	#------------------- Primera capa convolucional-------------------
	with tf.name_scope("convolucion1"):
		#Forma de los pesos del filtro
		#[tamanho_filtro,tamanho_filtro, canalEnt_img, cant_filtros]
		forma = [5, 5, 1, 32]
		
		pesos_conv1 = inicializar_pesos(shape = forma)
		biases_conv1 = inicializar_bias([32])

		#strides=[img,movx,movy ,filtro]
		convolucion1 = tf.nn.conv2d(imagen,
			                       pesos_conv1,
			                       strides=[1, 1, 1, 1],
			                       padding='SAME')

		#el valor del bias es adicionado para cada resultado de convolucion
		convolucion1 += biases_conv1
		#funcion de activacion
		act_conv1 = tf.nn.relu(convolucion1)
		#print(act_conv1.get_shape()) # => (60000, 28, 28, 32)
		
		pool_conv1 = tf.nn.max_pool(act_conv1,
		                            ksize=[1, 2, 2, 1],
		                            strides=[1, 2, 2, 1],
		                            padding='SAME')
		
		#print(pool_conv1.get_shape()) # => (60000, 14, 14, 32)

	# -------------------Segunda capa convolucional-------------------
	with tf.name_scope("convolucion2"):
		#Forma de los pesos del filtro
		#[tamanho_filtro,tamanho_filtro, canalEnt_img, cant_filtros]
		forma = [5, 5, 32, 64]
		
		pesos_conv2 = inicializar_pesos(shape = forma)		
		biases_conv2 = inicializar_bias([64])	

		#strides=[img,movx,movy ,filtro]
		convolucion2 = tf.nn.conv2d(pool_conv1,
			                       pesos_conv2,
			                       strides=[1, 1, 1, 1],
			                       padding='SAME')
		#el valor del bias es adicionado para cada resultado de convolucion
		convolucion2 += biases_conv2
		
		#funcion de activacion
		act_conv2 = tf.nn.relu(convolucion2)
		#print (act_conv2.get_shape()) # => (60000, 14,14, 64)
		
		
		pool_conv2 = tf.nn.max_pool(act_conv2,
		                            ksize=[1, 2, 2, 1],
		                            strides=[1, 2, 2, 1],
		                            padding='SAME')
		#print (pool_conv2.get_shape()) # => (60000, 7, 7, 64)

	# -------------Capa totalmente conectada-------------------
	with tf.name_scope("FC1"):
		#Para esta capa se necesita un tensor de 2 dimensiones(Entradas, Salidas)
		forma = [7 * 7 * 64, 1024]
		pesos_fc1 = inicializar_pesos(shape = forma)
		biases_fc1 = inicializar_bias([1024])

		pool_conv2_flat = tf.reshape(pool_conv2, [-1, 7*7*64])
		# (60000, 7, 7, 64) => (60000, 3136)
		
		# Multiplicar matriz 'pool_conv2_flat' por matriz 'pesos_fc1' y sumar bias
		fc1 = tf.matmul(pool_conv2_flat, pesos_fc1) + biases_fc1
		#print (fc1.get_shape()) # => (60000, 1024)
		
		act_fc1 = tf.nn.relu(fc1)
		

	#Aplicarmos dropout entre la capa FC y la capa de salida
	keep_prob = tf.placeholder('float',name=NOMBRE_PROBABILIDAD)
	h_fc1_drop = tf.nn.dropout(act_fc1, keep_prob)

	#------------------- Capa de Salida-------------------------
	with tf.name_scope("FC2"):
		pesos_fc2 = inicializar_pesos([1024, CANT_CLASES])
		biases_fc2 = inicializar_bias([CANT_CLASES])

		fc2 = tf.matmul(h_fc1_drop, pesos_fc2) + biases_fc2
		
	
	y_calculada = tf.nn.softmax(fc2, name = NOMBRE_TENSOR_SALIDA_CALCULADA)
	#print (y_calculada.get_shape()) # => (60000, 10)

	
	#[0.01, 0.04, 0.02, 0.5, 0.03 0.01, 0.05, 0.02, 0.3, 0.02] => 3
	predictor = tf.argmax(y_calculada,dimension = 1)
	tf.add_to_collection("predictor", predictor)
	
	#------------------------------------------------------------
	
	#el error que queremos minimizar va a estar en funcion de lo calculado con lo deseado(real)
	error = -tf.reduce_sum(y_deseada * tf.log(y_calculada))
	#error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_calculada, labels=y_deseada), name="error")

	with tf.name_scope("entrenamiento"):
		#Funcion de optimizacion
		iterac_entren = tf.Variable(0, name='iterac_entren', trainable=False)
		optimizador = tf.train.AdamOptimizer(TASA_APRENDIZAJE).minimize(error, global_step=iterac_entren)

	with tf.name_scope("Acierto"):
		# evaluacion
		prediccion_correcta = tf.equal(tf.argmax(y_calculada,1), tf.argmax(y_deseada,1))
		acierto = tf.reduce_mean(tf.cast(prediccion_correcta, 'float'))
		tf.summary.scalar("acierto", acierto)
	
	resumen = tf.summary.merge_all()
	
	return sess,resumen, x,y_deseada, keep_prob, iterac_entren, optimizador,acierto, predictor


if __name__ == "__main__":
	
	print "Inicio de lectura del dataset"
	path = '/media/josuetavara/Gaston/mnist/mnistDS/'
	datasetEntrenamiento = pd.read_csv(path+'datasets/60ktrain.csv')
	datasetEvaluacion = pd.read_csv(path+'datasets/10ktest.csv')

	#--------------------------------PREPROCESAMIENTO DE DATOS-----------------------------------------
	print "Procesamiento de las imagenes" 
	tam_imagen, entrenam_imagenes, entrenam_clases, entrenam_clases_flat = procesamiento(datasetEntrenamiento)
	tam_imagen, eval_imagenes, eval_clases, eval_clases_flat = procesamiento(datasetEvaluacion)
	
	# ancho y altura = 28 y 28
	anchura_imagen = altura_imagen = np.ceil(np.sqrt(tam_imagen)).astype(np.uint8)
	
	#--------------------------------CREACION DE LA RED------------------------------------------------
	print "Inicio de creacion de la red"
	tf.reset_default_graph()
	sess = tf.Session()
	sess , resumen, x,y_deseada, keep_prob, iterac_entren, optimizador,acierto,predictor =  create_cnn(sess)
	

	#--------------------------------ENTRENAMIENTO DE LA RED------------------------------------------------
	
	
	modelPath = '/media/josuetavara/Gaston/mnist/mnistDS/CNN/models/'
	#tensorboard --logdir /media/josuetavara/Gaston/mnist/mnistDS/CNN/models
	
	
	sess.run(tf.global_variables_initializer())

	#
	entren_writer = tf.summary.FileWriter(modelPath+'entrenamiento4', sess.graph)
	#entren_writer.add_graph(sess.graph)
	evalua_writer = tf.summary.FileWriter(modelPath+'evaluacion4')

	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state(modelPath + '.')
	
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print("Sesion restaurada de: %s" % ckpt.model_checkpoint_path)

	else:
		print("No se encontro puntos de control.")
		
	
	ultima_iteracion = iterac_entren.eval(sess)
	print "Ultimo modelo en la iteracion: ", ultima_iteracion
	
	
	epocas_completadas = 0
	indice_en_epoca = 0
	cant_imag_entrenamiento = entrenam_imagenes.shape[0]#60000
	cant_imag_evaluacion = eval_imagenes.shape[0]#10000
	
	clases_calc = np.zeros(shape=cant_imag_evaluacion, dtype=np.int)
	
	comienzo_time = time.time()
	
	#Desde la ultima iteracion hasta el ITERACIONES_ENTRENAMIENTO dado 
	for i in range(ultima_iteracion, ITERACIONES_ENTRENAMIENTO):
		#Obtener nuevo subconjunto(batch) de (BATCH_SIZE =100) imagenes
		batch_img_entrada, batch_img_clase = siguiente_batch_entren(BATCH_SIZE,cant_imag_entrenamiento)

		# Entrenar el batch
		[resu, _ ] = sess.run([resumen,optimizador], feed_dict={x: batch_img_entrada, y_deseada: batch_img_clase, keep_prob: DROPOUT})
		entren_writer.add_summary(resu, i)
		
		# Observar el progreso cada 'CHKP_REVISAR_PROGRESO' iteraciones
		if(i+1) % CHKP_REVISAR_PROGRESO == 0 :
			
			feed_dictx = {x: eval_imagenes, y_deseada: eval_clases,keep_prob: 1.0}
			[resu, aciertos_eval] = sess.run([resumen,acierto], feed_dict=feed_dictx)	
			
			evalua_writer.add_summary(resu, i)
			print('En la iteracion %d , Acierto de Evaluacion => %.4f '% (i+1, aciertos_eval))
		#if(i+1 == 10):
			#CHKP_REVISAR_PROGRESO *=10
		#Crear 'punto de control' cuando se llego a las CHKP_GUARDAR_MODELO iteraciones
		if (i+1) % CHKP_GUARDAR_MODELO == 0 :
			print('Guardando modelo en %d iteraciones....' %(i+1))
			saver.save(sess, modelPath+NOMBRE_MODELO, global_step=i+1,write_meta_graph=False)
	
	#Fin del proceso de entrenamiento y validacion del modelo
	end_time = time.time()
	# Cuanto tiempo tomo el entrenamiento
	time_dif = end_time - comienzo_time

	# Imprimir tiempo 
	print('Tiempo usado en %d iteraciones: %s ' % (ITERACIONES_ENTRENAMIENTO - ultima_iteracion, str(timedelta(seconds=int(round(time_dif))))))
	#tf.train.write_graph(sess.graph_def, modelPath,NOMBRE_MODELO + '.pbtxt', True) 
	
	feed_dictx = {x: eval_imagenes, y_deseada: eval_clases,keep_prob: 1.0}
	clases_calc[:] = sess.run(predictor, feed_dict=feed_dictx)
	clases_deseadas = eval_clases_flat
	# Crea una matriz booleana
	correct = (clases_deseadas == clases_calc)
		
	# Muestra algunas imagenes que no fueron clasificadas correctamente
	plot_example_errors(cls_pred=clases_calc, correct=correct, images = eval_imagenes, labels_flat=eval_clases_flat)
	plt.show()	
	
	print("Mostrando Matriz de Confusion")
	plot_confusion_matrix(clases_calc, clases_deseadas,10)
	plt.show()	
	

	
