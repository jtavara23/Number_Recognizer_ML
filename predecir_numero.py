# -*- coding: utf-8 -*-	
import numpy as np
import cv2 
import tensorflow as tf
from funcionesAuxiliares import activation_vector, plot_example_errors,plot_confusion_matrix,display
import math
import os
import sys
from matplotlib import pyplot as plt
from scipy import ndimage
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

path = '/media/josuetavara/Gaston/mnist/mnistDS/'
NOMBRE_TENSOR_ENTRADA = 'inputX'
NOMBRE_TENSOR_SALIDA_DESEADA = "outputYDeseada"
NOMBRE_PROBABILIDAD = 'mantener_probabilidad'



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    """
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    """
    #else:
        # calculate the ratio of the width and construct the
        # dimensions
    r = width / float(w)
    dim = (width, int(h * r))
	
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
def read_image(imagen):
	
	data = []
	yy, xx = imagen.shape[:2]

	for x in xrange(0,xx):
		for y in xrange(0,yy):
			data.append(imagen[x,y])
	#outFile.write(repr(data)+ "\n")
	return np.matrix(data)


def analyse_image(name):
	image = cv2.imread(name)
	
	width,height,c = image.shape
	if width > height:
		width  = int(width * float(280)/height)
		height = 280
	else:
		height  = int(height * float(280)/width)
		width = 280
	image = cv2.resize(image, (height,width))
	cv2.imwrite('imagenes/a.jpg',image) 
	image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	cv2.imwrite('imagenes/b.jpg',image) 
	image=cv2.GaussianBlur(image,(5,5),0)# 7 7| 9 9  
	cv2.imwrite('imagenes/c.jpg',image) 
	edged = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,1,11,3)#5 | 5
	cv2.imwrite('imagenes/d.jpg',edged) 


	qk = 3# 1 |5 
	#funcion de erosion y dilatacion
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (qk, qk))
	closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
	cv2.imwrite('imagenes/e.jpg',closed)


	im, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	
	lista = []
	wtam = 10
	htam = 20
	for c in cnts:
		x,y,w,h = cv2.boundingRect(c)
		#print w,h,' - ', x,y
		if w>wtam or h>htam:
			#print w,h,' - ', x,y
			lista.append([x,y,w,h])
	lista = np.array(lista) 
	lista = lista[lista[:,0].argsort()]
	return lista,closed

def drawDetected(lista,copyclone):
	for c in xrange(0,len(lista)):
		x,y,w,h = lista[c]
		cv2.rectangle(copyclone,(x,y),(x+w,y+h),200,1)
	cv2.imwrite('imagenes/f.jpg',copyclone)


def runModelo(numero_entrada):
	print "Ejecutando sesion tensorflow"
	with tf.Session() as sess:
		# Restore latest checkpoint
		sess.run(tf.global_variables_initializer())
		saver = tf.train.import_meta_graph(path + 'CNN/modelos2/modelo-3000.meta')
		ult_pto_control = tf.train.latest_checkpoint(path + 'CNN/modelos2/.')
		saver.restore(sess, ult_pto_control)
		print "Modelo restaurado",ult_pto_control

		predictor = tf.get_collection("predictor")[0]

		newsize = 28
		
		#bordersize = np.array(list(range(1,20)))
		bordersize = 4
		idx = 0
		
		import os
	
		lista,closed = analyse_image(numero_entrada)			
		
		number = ""
		for c in xrange(0,len(lista)):
			x,y,w,h = lista[c]
			#print "Evaluando ",c,"º digito"
			print ('Evaluando {0}º digito'.format(c))
			idx+=1
			final_imagenes=closed[y:y+h,x:x+w]
			results = [0] * 10
		
			new_imagenes = final_imagenes
			
			rows,cols = new_imagenes.shape
			
			compl_dif = abs(rows-cols)
			half_Sm = compl_dif/2
			half_Big = half_Sm if half_Sm*2 == compl_dif else half_Sm+1
			if rows > cols:
			    new_imagenes = np.lib.pad(new_imagenes,((0,0),(half_Sm,half_Big)),'constant', constant_values=0)
			else:
			    new_imagenes = np.lib.pad(new_imagenes,((half_Sm,half_Big),(0,0)),'constant', constant_values=0)
			
			
			new_imagenes = image_resize(new_imagenes,width=20,height=20) 
			new_imagenes = cv2.resize(new_imagenes,(20,20),cv2.INTER_AREA)#w	
			
			new_imagenes = np.lib.pad(new_imagenes,((4,4),(4,4)),'constant')
			cv2.imwrite("imagenes/z"+str(c)+"b.jpg",new_imagenes)	

			#------------FIN DE PROCESAMIENTO DE IMAGEN------------
			data = read_image(new_imagenes)
			image = data.astype(np.float)
			
			image = np.multiply(image, 1.0 / 255.0)
			
			#image = image[0, :]
			image = image.reshape((28,28,1))
			np.reshape(image, (1,28, 28))
			image = image[np.newaxis,:,:,np.newaxis]
			

			#print image.shape
			feed_dictx = {NOMBRE_TENSOR_ENTRADA+":0": image, NOMBRE_PROBABILIDAD+":0":1.0}

			# Calcula la clase usando el predictor de nuestro modelo
			label_pred = sess.run(predictor, feed_dict=feed_dictx)
			
			pred = int(label_pred)
			results[pred] += 1
			#print results
			#print "********************"
			number += str(results.index(max(results)))
		print 'Numero analisado: ', number
		print "------------------------------"
		drawDetected(lista,closed)
		return number
"""
if __name__ == "__main__":
	
	nombre = (sys.argv[1])
	numbers = ['imagenes/'+nombre]
	x,y = analyse_image(numbers[0])
	
	#runModelo(numbers)
"""	

	
