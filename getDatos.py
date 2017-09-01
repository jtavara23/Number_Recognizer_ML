import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

path = '/media/josuetavara/Gaston/mnist/mnistDS/'
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.import_meta_graph(path + 'CNN/models/model-100.meta')
	saver.restore(sess, tf.train.latest_checkpoint(path + 'CNN/models/.'))
	print "Modelo restaurado",tf.train.latest_checkpoint(path + 'CNN/models/.')
	
	
	all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	#print all_vars
	print "tensors: ",len(all_vars)
	"""
	for ind in xrange(0,len(all_vars)):
		print all_vars[ind]
	"""
	
	x = all_vars[0]
	print x 
	print sess.run(x)
	x = all_vars[2]
	print x 
	print sess.run(x)
	