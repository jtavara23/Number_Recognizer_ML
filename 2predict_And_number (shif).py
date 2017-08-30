import numpy as np
import cv2 
import tensorflow as tf
from funcionesAuxiliares import activation_vector, plot_example_errors,plot_confusion_matrix,display
import math
import os
from matplotlib import pyplot as plt
from scipy import ndimage
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

path = '/media/josuetavara/Gaston/mnist/mnistDS/'



def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)
    #print cy,cx

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

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
	cv2.imwrite('img/a.jpg',image) 
	image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	cv2.imwrite('img/b.jpg',image) 
	image=cv2.GaussianBlur(image,(7,7),0)# 5 5 | 9 9 | 7 7 | 7 7 | 7 7
	cv2.imwrite('img/c.jpg',image) 
	edged = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,1,11,3)
	cv2.imwrite('img/d.jpg',edged) 


	qk = 3#3 |1 |1 |1 | 3
	#applying closing function 
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (qk, qk))
	closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
	cv2.imwrite('img/e.jpg',closed)


	im, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	copyclone = closed
	lista = []
	wtam = 10
	htam = 20
	for c in cnts:
		x,y,w,h = cv2.boundingRect(c)
		#print w,h,' - ', x,y
		if w>wtam or h>htam:
			#print w,h,' - ', x,y
			cv2.rectangle(copyclone,(x,y),(x+w,y+h),200,1)
			lista.append([x,y,w,h])
	lista = np.array(lista) 
	lista = lista[lista[:,0].argsort()]
	cv2.imwrite('img/f.jpg',copyclone)	
	return lista,copyclone


if __name__ == "__main__":
	
	#x,y = analyse_image('prueba7.jpg')
	#"""
	print "Running session"
	with tf.Session() as sess:
		# Restore latest checkpoint
		sess.run(tf.global_variables_initializer())
		saver = tf.train.import_meta_graph(path + 'cnn/models/model-100.meta')
		saver.restore(sess, tf.train.latest_checkpoint(path + 'cnn/models/.'))
		print "Model restored",tf.train.latest_checkpoint(path + 'cnn/models/.')

		predict_op = tf.get_collection("predict_op")[0]

		newsize = 28
		
		#bordersize = np.array(list(range(1,20)))
		bordersize = 4
		idx = 0
		print "Testing numbers"
		import os
		#numbers = os.listdir("/media/josuetavara/Gaston/mnist/image_processing/block_numbers")
		numbers = ['img/prueba5.jpg']

		for jj in xrange(0,len(numbers)):
			lista,closed = analyse_image(numbers[jj])
			number = ""
			for c in xrange(0,len(lista)):
				x,y,w,h = lista[c]
			
				idx+=1
				final_img=closed[y:y+h,x:x+w]
				results = [0] * 10
			
				new_img = final_img
				
				rows,cols = new_img.shape
				
				compl_dif = abs(rows-cols)
				half_Sm = compl_dif/2
				half_Big = half_Sm if half_Sm*2 == compl_dif else half_Sm+1
				if rows > cols:
				    new_img = np.lib.pad(new_img,((0,0),(half_Sm,half_Big)),'constant', constant_values=0)
				else:
				    new_img = np.lib.pad(new_img,((half_Sm,half_Big),(0,0)),'constant', constant_values=0)
				
				
				new_img = image_resize(new_img,width=20,height=20) 
				new_img = cv2.resize(new_img,(20,20),cv2.INTER_AREA)#w	
				
				new_img = np.lib.pad(new_img,((4,4),(4,4)),'constant')
				cv2.imwrite("img/z"+str(c)+"b.jpg",new_img)	

				#shiftx,shifty = getBestShift(new_img)
				#shifted = shift(new_img,shiftx,shifty)
				
				#new_img = shifted
				#cv2.imwrite("z"+str(c)+"img/b.jpg",new_img)	

				
				data = read_image(new_img)
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
				results[pred] += 1
				#print results
				#print "********************"
				number += str(results.index(max(results)))
			print 'Prediction: ',numbers[jj] , number
			print "------------------------------"
			#"""