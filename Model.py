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


def get_cae():
	arch = tflearn.input_data(shape=[None, 256, 256, 20], name='input')
	
	num_filter = 10*20

	
	arch = tflearn.conv_2d(arch, num_filter*1, 4, activation='relu')
	arch = tflearn.max_pool_2d(arch, 2)
	arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.conv_2d(arch, num_filter*2, 4, activation='relu')
	arch = tflearn.max_pool_2d(arch, 2)
	arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.conv_2d(arch, num_filter*4, 4, activation='relu')
	arch = tflearn.max_pool_2d(arch, 2)
	arch = tflearn.dropout(arch, 0.75)
	
	arch = tflearn.conv_2d(arch, num_filter*8, 4, activation='relu')
	arch = tflearn.max_pool_2d(arch, 2)
	arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.conv_2d(arch, num_filter*16, 4, activation='relu')
	
	# arch = tflearn.reshape(arch, new_shape=[-1, 16*16*num_filter*16])
	# arch = tflearn.fully_connected(arch,  16*16*num_filter*16, activation='relu')
	# arch = tflearn.reshape(arch, new_shape=[-1, 16,16,num_filter*16])
	
	arch = tflearn.dropout(arch, 0.75)
	# arch = tflearn.upsample_2d(arch, 2)
	arch = tflearn.layers.conv.upscore_layer(arch, 
						 num_classes=256, 
						 kernel_size=4, 
						 shape=[1, 32, 32, num_filter*16]
						 ) 
	arch = tflearn.conv_2d(arch, num_filter*8, 4, activation='relu')
	
	
	
	arch = tflearn.dropout(arch, 0.75)
	# arch = tflearn.upsample_2d(arch, 2)
	arch = tflearn.layers.conv.upscore_layer(arch, 
							 num_classes=256, 
							 kernel_size=4, 
							 shape=[1, 64, 64, num_filter*8]
							 ) 
	arch = tflearn.conv_2d(arch, num_filter*4, 4, activation='relu')
	arch = tflearn.dropout(arch, 0.75)
	# arch = tflearn.upsample_2d(arch, 2)
	arch = tflearn.layers.conv.upscore_layer(arch, 
							 num_classes=256, 
							 kernel_size=4, 
							 shape=[1, 128, 128, num_filter*4]
							 ) 
	arch = tflearn.conv_2d(arch, num_filter*2, 4, activation='relu')
	arch = tflearn.dropout(arch, 0.75)
	# arch = tflearn.upsample_2d(arch, 2)
	arch = tflearn.layers.conv.upscore_layer(arch, 
							 num_classes=256,  
							 kernel_size=4, 
							 shape=[1, 256, 256, num_filter*2]
							 ) 
	arch = tflearn.conv_2d(arch, num_filter*1, 4, activation='relu')
	arch = tflearn.dropout(arch, 0.75) 
	
	arch = tflearn.conv_2d(arch, 20, 1, activation='relu')
	
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

