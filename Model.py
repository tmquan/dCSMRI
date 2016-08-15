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
	# import tensorflow as tf
	# with tf.device('/gpu:1'):
	encoder = tflearn.input_data(shape=[None, 256, 256, 20], name='input')
	# encoder = encoder/255.0
	num_filter = 8*20
	encoder = tflearn.conv_2d(encoder, 20, 7, activation='relu', regularizer='L1')
	# encoder = tflearn.conv_2d(encoder, 20, 3, activation='relu', regularizer='L1')
	

	encoder = tflearn.conv_2d(encoder, num_filter*1, 3, activation='relu', regularizer='L1')
	encoder = tflearn.residual_block(encoder, 1, num_filter*1, batch_norm=False, regularizer='L1')
	scale_0 = encoder
	# encoder = tflearn.layers.normalization.batch_normalization(encoder)
	encoder = tflearn.max_pool_2d(encoder, 2)
	# encoder = tflearn.dropout(encoder, 0.75)


	encoder = tflearn.conv_2d(encoder, num_filter*2, 3, activation='relu', regularizer='L1')
	encoder = tflearn.residual_block(encoder, 1, num_filter*2, batch_norm=False, regularizer='L1')
	scale_1 = encoder
	# encoder = tflearn.layers.normalization.batch_normalization(encoder)
	encoder = tflearn.max_pool_2d(encoder, 2)
	# encoder = tflearn.dropout(encoder, 0.75)


	encoder = tflearn.conv_2d(encoder, num_filter*4, 3, activation='relu', regularizer='L1')
	encoder = tflearn.residual_block(encoder, 1, num_filter*4, batch_norm=False, regularizer='L1')
	scale_2 = encoder
	# encoder = tflearn.layers.normalization.batch_normalization(encoder)
	encoder = tflearn.max_pool_2d(encoder, 2)
	# encoder = tflearn.dropout(encoder, 0.75)
	

	encoder = tflearn.conv_2d(encoder, num_filter*8, 3, activation='relu', regularizer='L1')
	encoder = tflearn.residual_block(encoder, 1, num_filter*8, batch_norm=False, regularizer='L1')
	scale_3 = encoder
	# encoder = tflearn.layers.normalization.batch_normalization(encoder)
	encoder = tflearn.max_pool_2d(encoder, 2)
	# encoder = tflearn.dropout(encoder, 0.75)


	
	encoder = tflearn.conv_2d(encoder, num_filter*12, 3, activation='relu', regularizer='L1')
	encoder = tflearn.residual_block(encoder, 1, num_filter*16, batch_norm=False, regularizer='L1')
	# encoder = tflearn.layers.normalization.batch_normalization(encoder)

	decoder = encoder
	# decoder = tflearn.conv_2d_transpose(decoder, 
	# 								 nb_filter=num_filter*12, 
	# 								 filter_size=3, 
	# 								 activation='relu',
	# 								 regularizer='L1',
	# 								 output_shape=[16, 16])
	



	# decoder = tflearn.upsample_2d(decoder, 2)
	decoder = tflearn.layers.conv.upscore_layer(decoder, 
						 num_classes=256, 
						 kernel_size=3, 
						 shape=[1, 32, 32, num_filter*8]
						 ) 
	decoder = tflearn.conv_2d(decoder, num_filter*8, 3, activation='relu', regularizer='L1')
	# decoder = tflearn.residual_block(decoder, 1, num_filter*8, batch_norm=False, regularizer='L1')
	# decoder = tflearn.conv_2d_transpose(decoder, 
	# 								 nb_filter=num_filter*8, 
	# 								 filter_size=3, 
	# 								 activation='relu', 
	# 								 regularizer='L1',
	# 								 output_shape=[32, 32])
	# decoder = tflearn.layers.normalization.batch_normalization(decoder)
	# decoder = decoder + scale_3
	decoder = merge([decoder, scale_3], mode='elemwise_sum', axis=3)

	
	# decoder = tflearn.dropout(decoder, 0.75)
	# decoder = tflearn.upsample_2d(decoder, 2)
	decoder = tflearn.layers.conv.upscore_layer(decoder, 
							 num_classes=256, 
							 kernel_size=3, 
							 shape=[1, 64, 64, num_filter*4]
							 ) 
	decoder = tflearn.conv_2d(decoder, num_filter*4, 3, activation='relu', regularizer='L1')
	# decoder = tflearn.residual_block(decoder, 1, num_filter*4, batch_norm=False, regularizer='L1')
	# decoder = tflearn.conv_2d_transpose(decoder, 
	# 								 nb_filter=num_filter*4, 
	# 								 filter_size=3, 
	# 								 activation='relu',
	# 								 regularizer='L1',
	# 								 output_shape=[64, 64])
	# decoder = tflearn.layers.normalization.batch_normalization(decoder)
	# decoder = decoder + scale_2
	decoder = merge([decoder, scale_2], mode='elemwise_sum', axis=3)
	# decoder = tflearn.dropout(decoder, 0.75)
	# decoder = tflearn.upsample_2d(decoder, 2)
	decoder = tflearn.layers.conv.upscore_layer(decoder, 
							 num_classes=256, 
							 kernel_size=3, 
							 shape=[1, 128, 128, num_filter*2]
							 ) 
	decoder = tflearn.conv_2d(decoder, num_filter*2, 3, activation='relu', regularizer='L1')
	# decoder = tflearn.residual_block(decoder, 1, num_filter*2, batch_norm=False, regularizer='L1')
	# decoder = tflearn.conv_2d_transpose(decoder, 
	# 								 nb_filter=num_filter*2, 
	# 								 filter_size=3, 
	# 								 activation='relu',
	# 								 regularizer='L1',
	# 								 output_shape=[128, 128])
	# decoder = tflearn.layers.normalization.batch_normalization(decoder)
	# decoder = decoder + scale_1
	decoder = merge([decoder, scale_1], mode='elemwise_sum', axis=3)
	# decoder = tflearn.dropout(decoder, 0.75)
	# decoder = tflearn.upsample_2d(decoder, 2)
	decoder = tflearn.layers.conv.upscore_layer(decoder, 
							 num_classes=256, 
							 kernel_size=3, 
							 shape=[1, 256, 256, num_filter*1]
							 ) 
	decoder = tflearn.conv_2d(decoder, num_filter*1, 3, activation='relu', regularizer='L1')
	# decoder = tflearn.residual_block(decoder, 1, num_filter*1, batch_norm=False, regularizer='L1')
	# decoder = tflearn.conv_2d_transpose(decoder, 
	# 								 nb_filter=num_filter*1, 
	# 								 filter_size=3, 
	# 								 activation='relu',
	# 								 regularizer='L1',
	# 								 output_shape=[256, 256])
	# decoder = tflearn.layers.normalization.batch_normalization(decoder)
	# decoder = decoder + scale_0
	decoder = merge([decoder, scale_0], mode='elemwise_sum', axis=3)
	# decoder = tflearn.dropout(decoder, 0.75) 
	
	decoder = tflearn.conv_2d(decoder, 20, 1, activation='relu', regularizer='L1')
	# decoder = tflearn.conv_2d_transpose(decoder, 
	# 								 nb_filter=20, 
	# 								 filter_size=3, 
	# 								 activation='relu',
	# 								 regularizer='L1',
	# 								 output_shape=[256, 256])
	
	# decoder = tf.round(decoder)
	decoder = tf.clip_by_value(decoder, 0, 255)
	
	return decoder
########################################################
def get_model():
	"""
	Define the architecture of the network is here
	"""
	# import tensorflow as tf
	# with tf.device('/gpu:0'):
	arch = get_cae()
	r2 = tflearn.metrics.R2()
	net = tflearn.regression(arch, optimizer='Adam',
						 learning_rate=0.001,
                         loss='mean_square')
	# Training the network
	model = DNN(net, 
				# checkpoint_path='models',
				tensorboard_verbose=3)
	return model

