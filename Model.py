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
	
	num_filter = 8*20
	# arch = tflearn.conv_2d(arch, 40, 3, activation='relu')
	
	# arch = tflearn.max_pool_2d(arch, 2)
	# arch = tflearn.conv_2d(arch, 81, 3, activation='relu')
	# arch = tflearn.max_pool_2d(arch, 2)

	# arch = tflearn.upsample_2d(arch, 2)
	# arch = tflearn.conv_2d(arch, 40, 3, activation='relu')
	# arch = tflearn.upsample_2d(arch, 2)
	
	# arch = tflearn.conv_2d(arch, num_filter*1, 3, activation='relu')
	arch = tflearn.residual_block(arch, 2, num_filter*1)
	arch = tflearn.max_pool_2d(arch, 2)
	# arch = tflearn.dropout(arch, 0.75)
	# arch = tflearn.conv_2d(arch, num_filter*2, 3, activation='relu')
	arch = tflearn.residual_block(arch, 2, num_filter*2)
	arch = tflearn.max_pool_2d(arch, 2)
	# arch = tflearn.dropout(arch, 0.75)
	# arch = tflearn.conv_2d(arch, num_filter*4, 3, activation='relu')
	arch = tflearn.residual_block(arch, 2, num_filter*4)
	arch = tflearn.max_pool_2d(arch, 2)
	# arch = tflearn.dropout(arch, 0.75)
	
	# arch = tflearn.conv_2d(arch, num_filter*8, 3, activation='relu')
	arch = tflearn.residual_block(arch, 2, num_filter*8)
	arch = tflearn.max_pool_2d(arch, 2)

	# arch = tflearn.dropout(arch, 0.75)
	# arch = tflearn.conv_2d(arch, num_filter*16, 3, activation='relu')
	arch = tflearn.residual_block(arch, 2, num_filter*16)
	
	# arch = tflearn.reshape(arch, new_shape=[-1, 16*16*num_filter*16])
	# arch = tflearn.fully_connected(arch,  16*16*num_filter*16, activation='relu')
	# arch = tflearn.reshape(arch, new_shape=[-1, 16,16,num_filter*16])
	
	# arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.upsample_2d(arch, 2)
	# arch = tflearn.layers.conv.upscore_layer(arch, 
						 # num_classes=num_filter*8, 
						 # kernel_size=3, 
						 # shape=[1, 32, 32]
						 # ) 
	# arch = tflearn.conv_2d(arch, num_filter*8, 3, activation='relu')
	arch = tflearn.residual_block(arch, 2, num_filter*8)
	
	
				# arch = tflearn.conv_2d(arch, num_filter*1, 3, activation='relu')
					
				# arch = tflearn.reshape(arch, new_shape=[-1, 32*32*num_filter*1])
				# arch = tflearn.fully_connected(arch,  32*32*num_filter*1, activation='relu')
				# arch = tflearn.fully_connected(arch,  32*32*num_filter*2, activation='relu')
				# arch = tflearn.fully_connected(arch,  32*32*num_filter*1, activation='relu')
				# arch = tflearn.fully_connected(arch,  20*20*num_filter*1, activation='relu')
				# arch = tflearn.fully_connected(arch,  16*16*num_filter*1, activation='relu')
				# arch = tflearn.fully_connected(arch,  12*12*num_filter*1, activation='relu')
				# arch = tflearn.fully_connected(arch,  16*16*num_filter*1, activation='relu')
				# arch = tflearn.fully_connected(arch,  20*20*num_filter*1, activation='relu')
				# arch = tflearn.fully_connected(arch,  32*32*num_filter*1, activation='relu')
				# arch = tflearn.reshape(arch, new_shape=[-1, 32,32,num_filter*1])
				
				# arch = tflearn.conv_2d(arch, num_filter*8, 3, activation='relu')
	
	# arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.upsample_2d(arch, 2)
	# arch = tflearn.layers.conv.upscore_layer(arch, 
							 # num_classes=num_filter*8, 
							 # kernel_size=3, 
							 # shape=[1, 64, 64]
							 # ) 
	# arch = tflearn.conv_2d(arch, num_filter*4, 3, activation='relu')
	arch = tflearn.residual_block(arch, 2, num_filter*4)
	# arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.upsample_2d(arch, 2)
	# arch = tflearn.layers.conv.upscore_layer(arch, 
							 # num_classes=num_filter*4, 
							 # kernel_size=3, 
							 # shape=[1, 128, 128]
							 # ) 
	# arch = tflearn.conv_2d(arch, num_filter*2, 3, activation='relu')
	arch = tflearn.residual_block(arch, 2, num_filter*2)
	# arch = tflearn.dropout(arch, 0.75)
	arch = tflearn.upsample_2d(arch, 2)
	# arch = tflearn.layers.conv.upscore_layer(arch, 
							 # num_classes=num_filter*2, 
							 # kernel_size=3, 
							 # shape=[1, 256, 256]
							 # ) 
	# arch = tflearn.conv_2d(arch, num_filter*1, 3, activation='relu')
	arch = tflearn.residual_block(arch, 2, num_filter*1)
	# arch = tflearn.dropout(arch, 0.75) 
	
	arch = tflearn.conv_2d(arch, 20, 3, activation='relu')
	
	return arch
########################################################
def get_model():
	"""
	Define the architecture of the network is here
	"""

	arch = get_cae()

	def custom_acc(prediction, target, inputs):
		acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1)), tf.float32), name='acc')
		return acc
	def custom_loss(y_pred, y_true):

		old_shape = tflearn.utils.get_incoming_shape(y_pred)
		new_shape = [old_shape[0]*old_shape[1]*old_shape[2], old_shape[3]]
		cur_shape = [old_shape[0]*old_shape[1]*old_shape[2]*old_shape[3]]
		print new_shape
		# epsilon   = tf.constant(value=0.0001, shape=old_shape)
		# y_pred = y_pred + epsilon
		y_pred = tf.reshape(y_pred, new_shape)
		y_true = tf.reshape(y_true, new_shape)
		
		# y_pred = tf.nn.log_softmax(y_pred)
		#softmax_categorical_crossentropy, categorical_crossentropy, binary_crossentropy, mean_square, hinge_loss
		#http://tflearn.org/activations/
		# return tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, tf.to_int32(y_true))
  		# return tflearn.objectives.categorical_crossentropy(y_pred, y_true)
  		# return tflearn.objectives.roc_auc_score(y_pred, y_true)
  		# return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y_true))
  		# return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, tf.to_int32(y_true)))

		with tf.name_scope('loss'):
			num_classes = y_true.get_shape()[-1]
			y_pred = tf.reshape(y_pred, new_shape)
			# shape = [y_pred.get_shape()[0], num_classes]
			epsilon = tf.constant(value=0.0001, shape=new_shape)
			y_pred = y_pred + epsilon
			y_true = tf.to_float(tf.reshape(y_true, new_shape))
			softmax = tf.nn.softmax(y_pred)
			

			cross_entropy = -tf.reduce_sum(y_true * tf.log(softmax), reduction_indices=[1])
			
			cross_entropy_mean = tf.reduce_mean(cross_entropy,
												name='xentropy_mean')
			tf.add_to_collection('losses', cross_entropy_mean)
			
			loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		return cross_entropy_mean
		# return loss
		# return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y_true))

  		# return tflearn.objectives.weak_cross_entropy_2d(y_pred, tf.to_int32(y_true), num_classes=2)
	# net = regression(arch, 
					 # optimizer='Adam', 
					 # learning_rate=0.001,
					 # # metric = tflearn.metrics.R2(),
					 # metric='Accuracy',
					 # # metric=custom_acc,
	                 # # loss='binary_crossentropy') # categorical_crossentropy, binary_crossentropy, mean_square, hinge_loss
	                 # # loss='categorical_crossentropy') # 
	                 # # loss='hinge_loss') # won't work
	                 # loss='mean_square') # won't work
	                 # # loss='L2') #softmax_categorical_crossentropy, categorical_crossentropy, binary_crossentropy, mean_square, hinge_loss
	                 # # loss='weak_cross_entropy_2d')
	                 # # loss=custom_loss)
	net = tflearn.regression(arch, optimizer='adam', 
						 metric='accuracy',
						 learning_rate=0.001,
                         loss='mean_square')
	# Training the network
	model = DNN(net, 
				# checkpoint_path='models',
				tensorboard_verbose=3)
	return model

