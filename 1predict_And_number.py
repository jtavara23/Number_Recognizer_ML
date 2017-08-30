import numpy as np
import cv2 
import tensorflow as tf
from helpp import dense_to_one_hot, plot_example_errors,plot_confusion_matrix,display
import math
import os
from matplotlib import pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

path = '/media/josuetavara/Gaston/mnist/mnistDS/'


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
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
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
	image = cv2.resize(image, (280,280))
	cv2.imwrite('a.jpg',image) 
	image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	cv2.imwrite('b.jpg',image) 
	image=cv2.GaussianBlur(image,(7,7),0)
	cv2.imwrite('c.jpg',image) 
	edged = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,1,11,3)
	cv2.imwrite('d.jpg',edged) 


	qk = 1#7
	#applying closing function 
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (qk, qk))
	closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
	cv2.imwrite('e.jpg',closed)


	im, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	copyclone = closed
	lista = []
	tam = 10
	for c in cnts:
		x,y,w,h = cv2.boundingRect(c)
		#print w,h,' - ', x,y
		if w>tam and h>tam and w<500:
			#print w,h,' - ', x,y
			cv2.rectangle(copyclone,(x,y),(x+w,y+h),200,1)
			lista.append([x,y,w,h])
	lista = np.array(lista) 
	lista = lista[lista[:,0].argsort()]
	cv2.imwrite('f.jpg',copyclone)	
	return lista,closed


if __name__ == "__main__":
	
	#x,y = analyse_image('prueba3.jpg')
	#"""
	print "Running session"
	with tf.Session() as sess:
		# Restore latest checkpoint
		sess.run(tf.global_variables_initializer())
		saver = tf.train.import_meta_graph(path + 'cnn/models/model-6100.meta')
		saver.restore(sess, tf.train.latest_checkpoint(path + 'cnn/models/.'))
		print "Model restored",tf.train.latest_checkpoint(path + 'cnn/models/.')

		predict_op = tf.get_collection("predict_op")[0]

		newsize = 28
		
		bordersize = np.array(list(range(1,20)))
		idx = 0
		print "Testing numbers"
		import os
		#numbers = os.listdir("/media/josuetavara/Gaston/mnist/image_processing/block_numbers")
		numbers = ['prueba3.jpg']

		for jj in xrange(0,len(numbers)):
			lista,closed = analyse_image(numbers[jj])
			number = ""
			for c in xrange(0,len(lista)):
				x,y,w,h = lista[c]
			
				idx+=1
				final_img=closed[y:y+h,x:x+w]
				results = [0] * 10
				for ii in xrange(0,len(bordersize)): 
					new_img = final_img
					border_image=cv2.copyMakeBorder(new_img, top=bordersize[ii], bottom=bordersize[ii], left=bordersize[ii], right=bordersize[ii], borderType= cv2.BORDER_CONSTANT)
					#resized_image = cv2.resize(border_image, (newsize,newsize)) 
					resized_image = image_resize(border_image,width=28,height=28) 
					resized_image = cv2.resize(resized_image,(newsize,newsize),cv2.INTER_AREA)#w	
					
					data = read_image(resized_image)
					image = data.astype(np.float)
					#print dataimages
					image = np.multiply(image, 1.0 / 255.0)
					
					image = image[0, :]
					# Create a feed-dict with these images and labels.
					feed_dictx = {"phx:0": image ,"p_keep_conv:0":1.0}

					# Calculate the predicted class using TensorFlow.
					label_pred = sess.run(predict_op, feed_dict=feed_dictx)
						
					#print label_pred
					
					pred = int(label_pred)
					#print bordersize[ii], pred
					results[pred] += 1
				print results
				print "********************"
				number += str(results.index(max(results)))
			print 'Prediction: ',numbers[jj] , number
			print "------------------------------"
		#"""