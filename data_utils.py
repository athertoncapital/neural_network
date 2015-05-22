import cPickle
import numpy as np
import os
import pdb

def unpickle(file_path):
	"""
	Unpickles a cifar batch file.

	Parameters
	----------
	file_path : string
		path to the cifar10 batch file, which is a Python "pickled" object produced with cPickle. 

	Returns
	-------
	dict
		a dictioniary with the following elements:
		
		data (string) -> numpy.ndarray
			a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. 
			The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. 
			The image is stored in row-major order, so that the first 32 entries of the array are the red channel values 
			of the first row of the image.
			
		labels (string) -> list
			a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image 
			in the array data.
	"""
	with open(file_path, 'rb') as f:
		datadict = cPickle.load(f)
	return datadict

def load_cifar_batches(root_path):
	"""
	Load all cifar batches into a dictionary.

	Parameters
	----------
	root_path
		path to directory where all 6 cifar10 batch files are stored

	Returns
	-------
	dict
		a dictionary with the following elements: 

		training_data (string) -> numpy.ndarray
			a 50000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.			

		training_labels (string) -> numpy.ndarray
			a numpy array of 50000 vectors that each represent labels from 0 to 9. The number at index i indicates the label 
			of the ith image in the training data.

		test_data (string) -> numpy.ndarray
			a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.		
		
		test_labels (string) -> numpy.ndarray
			a numpy array of 50000 vectors that each represent labels from 0 to 9. The number at index i indicates the label 
			of the ith image in the test data.
 	"""
 	dataset = {}

 	# training data (5 batch files)
 	training_data = []
 	training_labels = []
 	for i in range(1, 2): # range(1, 6) for all 5 datasets, use just 1 for development
 		batch_file_name = 'data_batch_' + str(i)
 		file_path = os.path.join(root_path, batch_file_name)
 		
 		datadict = unpickle(file_path)
 	
 		training_data.extend(datadict['data'])
 		training_labels.extend(datadict['labels'])

 	dataset['training_data'] = np.array(training_data)
 	dataset['training_labels'] = np.array([vectorize_label(label) for label in training_labels])

 	# test data (1 batch file)
 	file_path = os.path.join(root_path, 'test_batch')

 	datadict = unpickle(file_path)
 	dataset['test_data'] = datadict['data']
 	dataset['test_labels'] = np.array(datadict['labels'])

 	return dataset

@np.vectorize
def vectorize_label(n):
	"""
	Constructs a 10 x 1 vector of length 10 based on the input, so that the (n-1)-th element in the vector is 1 and all the other
	elements are 0.

	Parameters
	----------
	int : label
		The label of the class.

	Returns
	-------
	numpy.ndarray
		The vectorized representation of the label.	
	"""
	v = np.zeros(10)
	v[n] = 1
	return v
	  	

  