import math, bigfloat, pdb
import numpy as np
from data_utils import load_cifar_batches

class Network():
	count = 0

	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes

		# initialize random weights
		self.w1 = 0.01 * np.random.randn(sizes[1], sizes[0]) # hidden layer weights 
		self.w2 = 0.01 * np.random.randn(sizes[2], sizes[1]) # output layer weights

	def train(self, training_data, training_labels):
		count = 0

		for i, instance in enumerate(training_data):
			if not i % 100:
				print i

			# FEEDFORWARD
			# reshape array in order to easily calculate dot product
			x = instance.reshape(3072, 1)
			
			y = Network.sigmoid(np.dot(self.w1, x))
			z = np.dot(self.w2, y)
			p = self.softmax(z)
			
			if p.argmax() == training_labels[i].argmax():
				count += 1

			# loss = self.cross_entropy_loss(p, training_labels[i])

			# BACKPROPAGATION
			# output layer (z) gradient
			dloss_dz = p - training_labels[i]
			dloss_dw2 = np.dot(dloss_dz, y.T)

			# hidden layer (y) gradient
			# dloss_dy = np.dot(self.w2.T, dloss_dz)
			# dy_da = self.sigmoid_derivative(y)
			# dy_dw1 = np.dot(dy_da, x.T)
			# dloss_dw1 = np.dot(dloss_dy.T, dy_dw1)

			# # perform parameter update
			# self.w1 -= dloss_dw1
			self.w2 -= dloss_dw2

		return float(count) / len(training_labels)

	def cross_entropy_loss(self, probability_distribution, target_vector):
		"""
		Cross entropy loss function.
		"""
		return -np.sum((target_vector * np.log(probability_distribution)))

	def softmax(self, z):
		"""
		Takes a vector of arbitrary real-valued scores and squashes it to a vector of values 
		between zero and one that sum to one. This vector represents a probability distribution
		over mutually exclusive alternatives.

		Parameters
		----------
		z : numpy.ndarray
			a vector with real values that represent the scores of each class. 

		Returns
		-------
		numpy.ndarray
			 a vector of values between zero and one that sum to one.
		"""
		f = z - np.max(z)
		return np.exp(f) / np.sum(np.exp(f))

	@staticmethod
	@np.vectorize
	def sigmoid(x):
	    """ Numerically-stable sigmoid function. """
	    if x >= 0:
	        z = np.exp(-x)
	        return 1 / (1 + z)
	    else:
	        # if x is less than zero then z will be small, denom can't be
	        # zero because it's 1+z.
	        z = np.exp(x)
	        return z / (1 + z)

	# @staticmethod
	# def sigmoid(x):
	# 	""" 
	# 	A function that takes a real-valued number and "squashes" it into range between 0 and 1, so
	# 	that large negative numbers become 0 and large positive numbers become 1.  

	# 	Parameters
	# 	----------
	# 	x : numpy.ndarray
	# 		an array of real values, each of them is given as an argument to the sigmoid function.

	# 	Returns
	# 	-------
	# 	numpy.ndarray
	# 		an array of real values that are "squashed" into range between 0 and 1 by the function
	# 	""" 
	# 	return 1.0 / (1.0 + np.exp(-x))

	def sigmoid_derivative(self, y):
		return self.sigmoid(y) * (1 - self.sigmoid(y))

root_path = '/home/sten/projects/neural_network/cifar_10_batches'
d = load_cifar_batches(root_path)

nn = Network([3072, 2000, 10])
x = nn.train(d['training_data'], d['training_labels'])
print "Accuracy: " + str(x)