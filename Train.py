#!/usr/bin/env python



# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 10:12:52 2016

@author: tmquan
"""
from TFLearn 		import *
from DeepCSMRI 		import *
from Model 			import *
from Utility		import *
from GeneratePair 	import * # Import the data generation: zerofilling, full recon, mask
import logging


######################################################################################

def train():
	# X = np.load('X_train.npy')
	# y = np.load('y_train.npy')

	# y  = y.astype('float32')
	# X  = X.astype('float32')
	
	# print "X shape", X.shape
	# print "X dtype", X.dtype
	# print "Y shape", y.shape
	# print "Y dtype", y.dtype
	
	## Load the data 
	print "Load the data"
	images = np.load('images.npy')
	
	# Xz, Xf = generatePair(images)
	X, y, R = generatePair(images)
	
	print X.shape
	print y.shape
	# print R.shape
	X = np.transpose(X, (0,2,3,1))
	y = np.transpose(y, (0,2,3,1))
	# R = np.transpose(R, (0,2,3,1))
	# X = X/255.0
	# y = y/255.0
	
	np.save("X_train.npy", X)
	np.save("y_train.npy", y)
	# np.save("R_train.npy", R)
		
	# X = np.load("X_train.npy")
	# y = np.load("y_train.npy")
	# R = np.load("R_train.npy")
	
	##################################################################################
	# One hot code the full reconstruction
	# y = np.reshape(y, (-1, 400*width*width))
	# print y.shape
	# current_shape = y.flatten().shape[0]
	# print current_shape
	# one_hot_shape = [current_shape, 256]
	
	# new_y = np.zeros((current_shape, 256), dtype=np.int32)
	# new_y[np.arange(256), y] = 1
	# new_y = np.eye(256, dtype=np.int32)[y]
	# new_y = np.squeeze(new_y)
	# new_y = np.array([:,(x < y.flatten().all()) for x in range(256)], dtype=np.uint8)
	
	# print "New y"
	# print new_y.shape
	# assert(np.argmax(np.squeeze(new_y), axis=1).all()==y.all())
	
	# # Take the inverse
	# print "Run the postfix: [1, 2]/4 => [[1,1,0,0],[1,1,1,0]"
	# print "Run the postfix: [1, 2]/4 => [[1,1,0,0],[1,1,1,0]"
	# # new_y = new_y[:,::-1]
	# cum_y = np.cumsum(new_y, axis=1) #[1,1,0,0]
	# cum_y = 1-cum_y
	# # new_y = new_y + cum_y # postfix sum
	# assert(np.sum(cum_y, axis=1).all()==y.all())
	
	# y = np.squeeze(cum_y) 	# Recast the y

	
	# y = np.reshape(y, (-1, 256*tempo, width, width))
	# y = np.reshape(y, (-1, width, width, 256*tempo))
	# y = np.transpose(y, (0, 3, 1, 2)) # permute the dimension to get the channel in 1st
	##################################################################################
	# y = np.reshape(y, (-1, tempo*width*width))
	##################################################################################
	nb_iter 		= 100001
	epochs_per_iter = 1 
	batch_size 		= 1
	
	model = get_model()
	
	
	nb_folds = 4
	kfolds = KFold(len(y), nb_folds)
	for iter in range(nb_iter):
		print('-'*50)
		print('Iteration {0}/{1}'.format(iter, nb_iter))  
		print('-'*50) 
		
		X, y, R = generatePair(images)
	
		print X.shape
		print y.shape
		# print R.shape
		X = np.transpose(X, (0,2,3,1))
		y = np.transpose(y, (0,2,3,1))
		
		# Shuffle the data
		print('Shuffle data...')
		seed = np.random.randint(1, 10e6)
		np.random.seed(seed)
		np.random.shuffle(X)
		np.random.seed(seed)
		np.random.shuffle(y)
		
		f = 0
		for train, valid in kfolds:
			print('='*50)
			print('Fold', f+1)
			f += 1
			
			# Extract train, validation set
			X_train = X[train]
			X_valid = X[valid]
			y_train = y[train]
			y_valid = y[valid]
			
			print "X_train", X_train.shape
			print "y_train", y_train.shape
			
			print "X_valid", X_valid.shape
			print "y_valid", y_valid.shape
			
			
						
			model.fit(X, y, 
				run_id="fully_convolutional_neural_network", 
				n_epoch=10, 
				validation_set=0.2,
				shuffle=True,
				show_metric=True,
				snapshot_step=160, 
				snapshot_epoch=False,
				batch_size=batch_size)
		if iter%100==0:
			fname = 'model_%05d.tfl' %(iter)
			model.save(fname)
if __name__ == '__main__':
	train()
