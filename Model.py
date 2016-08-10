from TFlearn import *
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
	# arch = arch/255.0
	num_filter = 10*20
	# arch = tflearn.conv_2d(arch, 40, 3, activation='relu')
	

	arch = tflearn.residual_block(arch, 1, num_filter*1)
	arch = tflearn.conv_2d(arch, num_filter*1, 3, activation='relu')
	arch = tflearn.layers.normalization.batch_normalization (arch)
	arch = tflearn.max_pool_2d(arch, 2)
	arch = tflearn.dropout(arch, 0.75)


	arch = tflearn.residual_block(arch, 1, num_filter*2)
	arch = tflearn.conv_2d(arch, num_filter*2, 3, activation='relu')
	arch = tflearn.layers.normalization.batch_normalization (arch)
	arch = tflearn.max_pool_2d(arch, 2)
	arch = tflearn.dropout(arch, 0.75)


	arch = tflearn.residual_block(arch, 1, num_filter*4)
	arch = tflearn.conv_2d(arch, num_filter*4, 3, activation='relu')
	arch = tflearn.layers.normalization.batch_normalization (arch)
	arch = tflearn.max_pool_2d(arch, 2)
	arch = tflearn.dropout(arch, 0.75)
	

	arch = tflearn.residual_block(arch, 1, num_filter*8)
	arch = tflearn.conv_2d(arch, num_filter*8, 3, activation='relu')
	arch = tflearn.layers.normalization.batch_normalization (arch)
	arch = tflearn.max_pool_2d(arch, 2)
	arch = tflearn.dropout(arch, 0.75)


	arch = tflearn.residual_block(arch, 1, num_filter*16)
	arch = tflearn.conv_2d(arch, num_filter*16, 3, activation='relu')
	arch = tflearn.layers.normalization.batch_normalization (arch)

	arch = tflearn.upsample_2d(arch, 2)
	# arch = tflearn.layers.conv.upscore_layer(arch, 
	# 					 num_classes=256, 
	# 					 kernel_size=3, 
	# 					 shape=[1, 32, 32, num_filter*8]
	# 					 ) 
	# arch = tflearn.conv_2d(arch, num_filter*8, 3, activation='relu')
	arch = tflearn.conv_2d_transpose(arch, 
									 nb_filter=num_filter*8, 
									 filter_size=3, 
									 activation='relu',
									 output_shape=[32, 32])
	arch = tflearn.layers.normalization.batch_normalization (arch)
	
	
	arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.upsample_2d(arch, 2)
	# arch = tflearn.layers.conv.upscore_layer(arch, 
	# 						 num_classes=256, 
	# 						 kernel_size=3, 
	# 						 shape=[1, 64, 64, num_filter*4]
	# 						 ) 
	# arch = tflearn.conv_2d(arch, num_filter*4, 3, activation='relu')
	arch = tflearn.conv_2d_transpose(arch, 
									 nb_filter=num_filter*4, 
									 filter_size=3, 
									 activation='relu',
									 output_shape=[64, 64])
	arch = tflearn.layers.normalization.batch_normalization (arch)

	arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.upsample_2d(arch, 2)
	# arch = tflearn.layers.conv.upscore_layer(arch, 
	# 						 num_classes=256, 
	# 						 kernel_size=3, 
	# 						 shape=[1, 128, 128, num_filter*2]
	# 						 ) 
	# arch = tflearn.conv_2d(arch, num_filter*2, 3, activation='relu')
	arch = tflearn.conv_2d_transpose(arch, 
									 nb_filter=num_filter*2, 
									 filter_size=3, 
									 activation='relu',
									 output_shape=[128, 128])
	arch = tflearn.layers.normalization.batch_normalization (arch)

	arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.upsample_2d(arch, 2)
	# arch = tflearn.layers.conv.upscore_layer(arch, 
	# 						 num_classes=256, 
	# 						 kernel_size=3, 
	# 						 shape=[1, 256, 256, num_filter*1]
	# 						 ) 
	# arch = tflearn.conv_2d(arch, num_filter*1, 3, activation='relu')
	arch = tflearn.conv_2d_transpose(arch, 
									 nb_filter=num_filter*1, 
									 filter_size=3, 
									 activation='relu',
									 output_shape=[256, 256])
	arch = tflearn.layers.normalization.batch_normalization (arch)

	arch = tflearn.dropout(arch, 0.75) 
	
	arch = tflearn.conv_2d(arch, 20, 1, activation='relu')
	# arch = tflearn.conv_2d_transpose(arch, 
	# 								 nb_filter=20, 
	# 								 filter_size=3, 
	# 								 activation='relu',
	# 								 output_shape=[256, 256])

	
	return arch
########################################################
def get_model():
	"""
	Define the architecture of the network is here
	"""

	arch = get_cae()

	net = tflearn.regression(arch, optimizer='Ftrl', 
						 metric='accuracy',
						 learning_rate=0.001,
                         loss='mean_square')
	# Training the network
	model = DNN(net, 
				# checkpoint_path='models',
				tensorboard_verbose=3)
	return model

