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

	## Load the data 
	print "Load the data"
	images = np.load('images.npy')
	images = images[0:100,:,:,:]

	
	##################################################################################
	
	##################################################################################
	nb_iter 		= 100001
	epochs_per_iter = 1 
	batch_size 		= 1
	
	model = get_model()
	
	
	nb_folds = 5
	kfolds = KFold(len(images), nb_folds)
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
			
			
						
			model.fit(X_train, y_train, 
				run_id="direct_model", 
				n_epoch=1, 
				validation_set=(X_valid, y_valid),
				shuffle=False,
				show_metric=True,
				snapshot_step=80, 
				snapshot_epoch=False,
				batch_size=batch_size)
		del X_train, X_valid, y_train, y_valid
		del X, y, R
		if iter%100==0:
			fname = 'model_pretrained_%05d.tfl' %(iter)
			model.save(fname)
if __name__ == '__main__':
	train()
