from TFLearn import *
import tensorflow as tf
import tflearn
from tflearn import utils
from tflearn import initializations
########################################################
# Real-time data preprocessing
Preprocessing = ImagePreprocessing()
Preprocessing.add_featurewise_zero_center()
Preprocessing.add_featurewise_stdnorm()

# Real-time data augmentation
Augmentation  = ImageAugmentation()
Augmentation.add_random_blur()
def convolution_module(net, kernel_size, filter_count, batch_norm=True, down_pool=False, up_pool=False, act_type="relu", convolution=True):
	import tflearn
	shape = tflearn.utils.get_incoming_shape(net)
	print shape
	batch_size = 20
	print batch_size
	if up_pool:
		net  	= upscore_layer(net, 
							 num_classes=filter_count, 
							 kernel_size=3, 
							 shape=[batch_size, shape[1]*2, shape[2]*2]
							 ) 
		net 	= batch_normalization(net)
		if act_type != "":
			net = activation(net, act_type)


	if convolution:
		net 	= conv_2d(net, filter_count, kernel_size)
		net 	= conv_2d(net, filter_count, kernel_size)
	
	if batch_norm:	
		net 	= batch_normalization(net)
	
	if act_type != "":
		net 	= activation(net, act_type)


	if down_pool:
		net = max_pool_2d(net, 2, strides=2)	

	return net

def get_cae():
	arch = tflearn.input_data(shape=[None, 256, 256, 20], name='input')
	
	num_filter = 16*20

	
	arch = tflearn.conv_2d(arch, num_filter*1, 3, activation='relu')
	arch = tflearn.max_pool_2d(arch, 2)
	arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.conv_2d(arch, num_filter*2, 3, activation='relu')
	arch = tflearn.max_pool_2d(arch, 2)
	arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.conv_2d(arch, num_filter*4, 3, activation='relu')
	arch = tflearn.max_pool_2d(arch, 2)
	arch = tflearn.dropout(arch, 0.75)
	
	arch = tflearn.conv_2d(arch, num_filter*8, 3, activation='relu')
	arch = tflearn.max_pool_2d(arch, 2)
	arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.conv_2d(arch, num_filter*16, 3, activation='relu')
	
	# arch = tflearn.reshape(arch, new_shape=[-1, 16*16*num_filter*16])
	# arch = tflearn.fully_connected(arch,  16*16*num_filter*16, activation='relu')
	# arch = tflearn.reshape(arch, new_shape=[-1, 16,16,num_filter*16])
	
	arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.upsample_2d(arch, 2)
	# arch = tflearn.layers.conv.upscore_layer(arch, 
						 # num_classes=num_filter*8, 
						 # kernel_size=3, 
						 # shape=[1, 32, 32]
						 # ) 
	arch = tflearn.conv_2d(arch, num_filter*8, 3, activation='relu')
	
	
	
	arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.upsample_2d(arch, 2)
	# arch = tflearn.layers.conv.upscore_layer(arch, 
							 # num_classes=num_filter*8, 
							 # kernel_size=3, 
							 # shape=[1, 64, 64]
							 # ) 
	arch = tflearn.conv_2d(arch, num_filter*4, 3, activation='relu')
	arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.upsample_2d(arch, 2)
	# arch = tflearn.layers.conv.upscore_layer(arch, 
							 # num_classes=num_filter*4, 
							 # kernel_size=3, 
							 # shape=[1, 128, 128]
							 # ) 
	arch = tflearn.conv_2d(arch, num_filter*2, 3, activation='relu')
	arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.upsample_2d(arch, 2)
	# arch = tflearn.layers.conv.upscore_layer(arch, 
							 # num_classes=num_filter*2, 
							 # kernel_size=3, 
							 # shape=[1, 256, 256]
							 # ) 
	arch = tflearn.conv_2d(arch, num_filter*1, 3, activation='relu')
	# arch = tflearn.dropout(arch, 0.75) 
	
	arch = tflearn.conv_2d(arch, 20, 3, activation='relu')
	
	return arch
########################################################
def get_model():
	"""
	Define the architecture of the network is here
	"""

	arch = get_cae()

	
	net = tflearn.regression(arch, optimizer='adam', 
						 metric='accuracy',
						 learning_rate=0.001,
                         loss='mean_square')
	# Training the network
	model = DNN(net, 
				# checkpoint_path='models',
				tensorboard_verbose=3)
	return model

