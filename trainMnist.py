import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
from datetime import timedelta

from funcionesAuxiliares import  display,activation_vector

from subprocess import check_output
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

NOMBRE_MODELO = 'model'
NOMBRE_TENSOR_ENTRADA = 'inputX'
NOMBRE_PROBABILIDAD = 'mantener_probabilidad'
NOMBRE_TENSOR_SALIDA_CALCULADA = 'outputYCalculada'
NOMBRE_TENSOR_SALIDA_DESEADA = "outputYDeseada"


TASA_APRENDIZAJE = 5e-4 #hasta 3000 iteraciones
#TASA_APRENDIZAJE = 1e-4  #desde 3000 a (+)
ITERACIONES_ENTRENAMIENTO = 200

CHKP_GUARDAR_MODELO = 100
CHKP_REVISAR_PROGRESO = CHKP_GUARDAR_MODELO

DROPOUT = 0.5

# entrenamiento es ejecutado seleccionando subconjuntos de imagenes
BATCH_SIZE = 100

#cantidad de imagenes del conjunto de entrenamiento separadas para validar
TAM_VALIDACION = 4000

CANT_CLASES = 10



def siguiente_batch(batch_size,cant_imag_entrenamiento ):
    

	#cant_imag_entrenamiento = entrenam_imagenes.shape[0] = 56000
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



# Muestra el progreso del entrenamiento

def chequear_progreso(sess, acierto, batch_img_entrada, batch_img_clase, i):
	#Acierto del batch de entrenamiento 
	train_accuracy = acierto.eval(session=sess, feed_dict={x: batch_img_entrada, y_deseada: batch_img_clase, keep_prob: 1.0})       	
	#Acierto de las imagenes de validacion
	validation_accuracy = acierto.eval(session=sess, feed_dict={x: validac_imagenes,y_deseada: validac_clases,keep_prob: 1.0})

	train_val_File.write('%d,%.4f,%.4f\n' % (i+1,train_accuracy,validation_accuracy))
	#print('Training_accuracy => %.4f for step %d' % (train_accuracy, i+1))
	return train_accuracy,validation_accuracy


# exportar modelo para version en android
def exportar_modelo(NOMBRE_TENSOR_ENTRADAs, NOMBRE_TENSOR_SALIDA_CALCULADA):
	
	from tensorflow.python.tools import freeze_graph
	from tensorflow.python.tools import optimize_for_inference_lib

	ckpt = tf.train.get_checkpoint_state(modelPath + '.')
	freeze_graph.freeze_graph('models/' + NOMBRE_MODELO + '.pbtxt', None, False,
	    ckpt.model_checkpoint_path, NOMBRE_TENSOR_SALIDA_CALCULADA, "save/restore_all",
	    "save/Const:0", 'models/frozen_' + NOMBRE_MODELO + '.pb', True, "")

	input_graph_def = tf.GraphDef()
	with tf.gfile.Open('models/frozen_' + NOMBRE_MODELO + '.pb', "rb") as f:
	    input_graph_def.ParseFromString(f.read())

	output_graph_def = optimize_for_inference_lib.optimize_for_inference(
	        input_graph_def, NOMBRE_TENSOR_ENTRADAs, [NOMBRE_TENSOR_SALIDA_CALCULADA],
	        tf.float32.as_datatype_enum)

	with tf.gfile.FastGFile('models/opt_' + NOMBRE_MODELO + '.pb', "wb") as f:
	    f.write(output_graph_def.SerializeToString())

	print("graph saved!")




def procesamiento(datasetEntrenamiento):
	imagenes = datasetEntrenamiento.iloc[:,1:].values
	imagenes = imagenes.astype(np.float)

	# Normalizar, convertir de [0:255] => [0.0:1.0]
	imagenes = np.multiply(imagenes, 1.0 / 255.0)
	
	#Tamanho de una imagen: 784 valores que son obtenidos de una imagen de 28 x 28
	tam_imagen = imagenes.shape[1]
	#print ('tam_imagen => {0}'.format(tam_imagen))

	#print ('anchura_imagen => {0}\naltura_imagen => {1}'.format(anchura_imagen,altura_imagen))
	
	# Mostrar la imagen 25 del datasetEntrenamiento 	IMAGE_TO_DISPLAY = 25
	#display(imagenes[IMAGE_TO_DISPLAY])

	#Organizar las clases de las imagenes en un solo vector
	clases_flat = datasetEntrenamiento.iloc[:,0].values.ravel()
	#Imprimir informacion de la CLASE de 'imagen-to-display' 
	#print ('label of imagen [{0}] => {1}'.format(IMAGE_TO_DISPLAY,clases_flat[IMAGE_TO_DISPLAY]))

	# convertir tipo de clases de escalares a vectores de activacion de 1s
	# 0 => [1 0 0 0 0 0 0 0 0 0]
	# 1 => [0 1 0 0 0 0 0 0 0 0]
	# ...
	# 9 => [0 0 0 0 0 0 0 0 0 1]
	clases = activation_vector(clases_flat, CANT_CLASES)
	clases = clases.astype(np.uint8)

	# Dividir los datos en conjuntos de entrenamiento & validacion
	validac_imagenes = imagenes[:TAM_VALIDACION]
	validac_clases = clases[:TAM_VALIDACION]

	entrenam_imagenes = imagenes[TAM_VALIDACION:]
	entrenam_clases = clases[TAM_VALIDACION:]
	entrenam_clases_flat = clases_flat[TAM_VALIDACION:]
	
	return tam_imagen, entrenam_imagenes,entrenam_clases,entrenam_clases_flat, validac_imagenes,validac_clases
	

# weights initialization.
# tf.truncated_normal: Outputs random values from a truncated normal distribution.
# The generated values follow a normal distribution with specified standard deviation.
def inicializar_pesos(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# tf.constant: Creates a constant tensor. 
# The resulting tensor is populated with values of type dtype, as specified by arguments value and shape
def inicializar_bias(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def create_cnn():
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
	#print (imagen.get_shape()) # =>(56000,28,28,1)

	#------------------- Primera capa convolucional-------------------
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
	#print(act_conv1.get_shape()) # => (56000, 28, 28, 32)
	
	pool_conv1 = tf.nn.max_pool(act_conv1,
	                            ksize=[1, 2, 2, 1],
	                            strides=[1, 2, 2, 1],
	                            padding='SAME')
	
	#print(pool_conv1.get_shape()) # => (56000, 14, 14, 32)


	# -------------------Segunda capa convolucional-------------------
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
	#print (act_conv2.get_shape()) # => (56000, 14,14, 64)

	pool_conv2 = tf.nn.max_pool(act_conv2,
	                            ksize=[1, 2, 2, 1],
	                            strides=[1, 2, 2, 1],
	                            padding='SAME')
	#print (pool_conv2.get_shape()) # => (56000, 7, 7, 64)


	# -------------Capa totalmente conectada-------------------
	#Para esta capa se necesita un tensor de 2 dimensiones(Entradas, Salidas)
	forma = [7 * 7 * 64, 1024]
	pesos_fc1 = inicializar_pesos(shape = forma)
	biases_fc1 = inicializar_bias([1024])

	pool_conv2_flat = tf.reshape(pool_conv2, [-1, 7*7*64])
	# (56000, 7, 7, 64) => (56000, 3136)
	
	# Multiplicar matriz 'pool_conv2_flat' por matriz 'pesos_fc1' y sumar bias
	fc1 = tf.matmul(pool_conv2_flat, pesos_fc1) + biases_fc1
	#print (fc1.get_shape()) # => (56000, 1024)
	
	act_fc1 = tf.nn.relu(fc1)

	#Aplicarmos dropout entre la capa FC y la capa de salida
	keep_prob = tf.placeholder('float',name=NOMBRE_PROBABILIDAD)
	h_fc1_drop = tf.nn.dropout(act_fc1, keep_prob)

	#------------------- Capa de Salida-------------------------
	pesos_fc2 = inicializar_pesos([1024, CANT_CLASES])
	biases_fc2 = inicializar_bias([CANT_CLASES])

	fc2 = tf.matmul(h_fc1_drop, pesos_fc2) + biases_fc2
	
	y_calculada = tf.nn.softmax(fc2, name = NOMBRE_TENSOR_SALIDA_CALCULADA)
	#print (y_calculada.get_shape()) # => (56000, 10)

	
	#[0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
	clase_predecida = tf.argmax(y_calculada,dimension = 1)
	tf.add_to_collection("predictor", clase_predecida)
	
	#------------------------------------------------------------

	#el costo que queremos minimizar va a estar en funcion de lo calculado con lo deseado(real)
	costo = -tf.reduce_sum(y_deseada * tf.log(y_calculada))

	#Funcion de optimizacion
	iterac_entren = tf.Variable(0, name='iterac_entren', trainable=False)
	optimizador = tf.train.AdamOptimizer(TASA_APRENDIZAJE).minimize(costo, global_step=iterac_entren)
	

	# evaluacion
	prediccion_correcta = tf.equal(tf.argmax(y_calculada,1), tf.argmax(y_deseada,1))
	acierto = tf.reduce_mean(tf.cast(prediccion_correcta, 'float'))
	#acierto = tf.reduce_mean(tf.cast(correct_prediction, 'float'),name="final_result")
	
	
	return x,y_deseada, keep_prob,iterac_entren, optimizador,acierto

if __name__ == "__main__":
	
	print "Inicio de lectura del dataset"
	path = '/media/josuetavara/Gaston/mnist/mnistDS/'
	datasetEntrenamiento = pd.read_csv(path+'datasets/60ktrain.csv')
	

	#--------------------------------PREPROCESAMIENTO DE DATOS-----------------------------------------
	tam_imagen, entrenam_imagenes, entrenam_clases, entrenam_clases_flat, validac_imagenes, validac_clases = procesamiento(datasetEntrenamiento)
	
	# ancho y altura = 28 y 28
	anchura_imagen = altura_imagen = np.ceil(np.sqrt(tam_imagen)).astype(np.uint8)
	
	#--------------------------------CREACION DE LA RED------------------------------------------------
	print "Inicio de creacion de la red"
	x,y_deseada, keep_prob, iterac_entren, optimizador,acierto =  create_cnn()
	

	#--------------------------------ENTRENAMIENTO DE LA RED------------------------------------------------
	
	
	modelPath = '/media/josuetavara/Gaston/mnist/mnistDS/CNN/models/'

	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

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
	cant_imag_entrenamiento = entrenam_imagenes.shape[0]
	# Archivo para guardar los aciertos de Entrenamiento y Validacion
	train_val_File = open("TrainVal_ac.csv","a")
	
	#Inicio del proceso de entrenamiento y validacion del modelo
	comienzo_time = time.time()
	
	#Desde la ultima iteracion hasta el ITERACIONES_ENTRENAMIENTO dado 
	for i in range(ultima_iteracion, ITERACIONES_ENTRENAMIENTO):
		#Obtener nuevo subconjunto(batch) de (BATCH_SIZE =100) imagenes
		batch_img_entrada, batch_img_clase = siguiente_batch(BATCH_SIZE,cant_imag_entrenamiento)

		# Entrenar el batch
		# DROPOUT = 0.5
		sess.run(optimizador, feed_dict={x: batch_img_entrada, y_deseada: batch_img_clase, keep_prob: DROPOUT})
		
		# Observar el progreso cada 'CHKP_REVISAR_PROGRESO' iteraciones
		if(i+1) % CHKP_REVISAR_PROGRESO == 0 :
			train_accuracy, validation_accuracy = chequear_progreso(sess, acierto, batch_img_entrada, batch_img_clase, i)
			print('En la iteracion %d , Aciertos: [Entrenamiento || Validacion] => %.4f || %.4f '% (i+1, train_accuracy, validation_accuracy))
		#Crear 'punto de control' cuando se llego a las CHKP_GUARDAR_MODELO iteraciones
		if (i+1) % CHKP_GUARDAR_MODELO == 0 :
			print('En la iteracion %d , Aciertos: [Entranamiento || Validacion] => %.4f || %.4f \n'% (i+1, train_accuracy, validation_accuracy))
			print('Guardando modelo %d ....' %(i+1))
			saver.save(sess, modelPath+NOMBRE_MODELO, global_step=i+1,write_meta_graph=True)
	
	#Fin del proceso de entrenamiento y validacion del modelo
	end_time = time.time()
	# Cuanto tiempo tomo el entrenamiento
	time_dif = end_time - comienzo_time

	# Imprimir tiempo 
	print('Tiempo usado en %d iteraciones: %s ' % (ITERACIONES_ENTRENAMIENTO - ultima_iteracion, str(timedelta(seconds=int(round(time_dif))))))
	train_val_File.close()

	#exportar modelo para version android
	#tf.train.write_graph(sess.graph_def, modelPath,NOMBRE_MODELO + '.pbtxt', True)
	#exportar_modelo([NOMBRE_TENSOR_ENTRADA, NOMBRE_PROBABILIDAD], NOMBRE_TENSOR_SALIDA_CALCULADA)
	