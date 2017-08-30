import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import tensorflow as tf
from funcionesAuxiliares import activation_vector, plot_example_errors,plot_confusion_matrix
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
BATCH_SIZE = 100
NOMBRE_TENSOR_ENTRADA = 'inputX'
NOMBRE_TENSOR_SALIDA_DESEADA = "outputYDeseada"
NOMBRE_PROBABILIDAD = 'mantener_probabilidad'

if __name__ == "__main__":
	path = '/media/josuetavara/Gaston/mnist/mnistDS/'
	dataset = pd.read_csv(path+'datasets/10ktest.csv')
	images = dataset.iloc[:,1:].values
	images = images.astype(np.float)

	
	# convert from [0:255] => [0.0:1.0]
	images = np.multiply(images, 1.0 / 255.0)
	#print('images({0[0]},{0[1]})'.format(images.shape))
	#labels_flat = dataset.iloc[:,0].values.ravel()
	labels_flat = dataset.iloc[:,0].values
	#print('length of one image ({0})'.format(len(labels_flat)))

	labels_count = np.unique(labels_flat).shape[0]
	#print('number of labes => {0}'.format(labels_count))
	
	# convert class labels from scalars to one-hot vectors
	# 0 => [1 0 0 0 0 0 0 0 0 0]
	# 1 => [0 1 0 0 0 0 0 0 0 0]
	# ...
	# 9 => [0 0 0 0 0 0 0 0 0 1]
	labels = activation_vector(labels_flat, labels_count)
	labels = labels.astype(np.uint8)

	with tf.Session() as sess:
		# Restore latest checkpoint
		sess.run(tf.global_variables_initializer())
		saver = tf.train.import_meta_graph(path + 'CNN/models/model-100.meta')
		saver.restore(sess, tf.train.latest_checkpoint(path + 'CNN/models/.'))
		print "Modelo restaurado",tf.train.latest_checkpoint(path + 'CNN/models/.')
		
		## How to access saved operation
		#graph = tf.get_default_graph()
		predict_op = tf.get_collection("predictor")[0]
	
		num_test = (images.shape[0])
		
		# Allocate an array for the predicted classes which
		# will be calculated in batches and filled into this array.
		cls_pred = np.zeros(shape=num_test, dtype=np.int)

		# Now calculate the predicted classes for the batches.
		# We will just iterate through all the batches.
		# There might be a more clever and Pythonic way of doing this.

		# The starting index for the next batch is denoted i.
		i = 0
		print "Prediciendo clases..."
		while i < num_test:
			# The ending index for the next batch is denoted j.
			j = min(i + BATCH_SIZE, num_test)

			# Get the images from the test-set between index i and j.
			test_images = images[i:j, :]
			
			# Get the associated labels.
			test_labels = labels[i:j, :]

			# Create a feed-dict with these images and labels.
			feed_dictx = {NOMBRE_TENSOR_ENTRADA+":0": test_images, NOMBRE_TENSOR_SALIDA_DESEADA+":0": test_labels,NOMBRE_PROBABILIDAD+":0":1.0}

			# Calculate the predicted class using TensorFlow.
			cls_pred[i:j] = sess.run(predict_op, feed_dict=feed_dictx)
			
			# Set the start-index for the next batch to the
			# end-index of the current batch.
			i = j

		# Convenience variable for the true class-numbers of the test-set.
		cls_true = labels_flat
		
		# Create a boolean array whether each image is correctly classified.
		correct = (cls_true == cls_pred)
		
		# Calculate the number of correctly classified images.
		# When summing a boolean array, False means 0 and True means 1.
		correct_sum = correct.sum()

		# Classification accuracy is the number of correctly classified
		# images divided by the total number of images in the test-set.
		acc = float(correct_sum) / num_test

		# Print the accuracy.
		msg = "Acierto en el conjunto de Testing: {0:.1%} ({1} / {2})"
		print(msg.format(acc, correct_sum, num_test))

		# Plot some examples of mis-classifications, if desired.
		#plot_example_errors(cls_pred=cls_pred, correct=correct,images = images, labels_flat=labels_flat)

		print("Mostrando Matriz de Confusion")
		plot_confusion_matrix(cls_pred, cls_true,labels_count)
		plt.show()
		print "Fin de evaluacion"
		outFile = open("Test_ac.csv","a")
		outFile.write(repr(tf.train.latest_checkpoint(path + 'CNN/models/.'))+"\n")
		outFile.write(msg.format(acc, correct_sum, num_test))
		outFile.write("\n-------------------------------------------------------------\n")