import math, pdb
import numpy as np
from data_utils import load_cifar_batches

class Network():

	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes

		# initialize random weights
		self.w1 = 0.001 * np.random.randn(sizes[1], sizes[0]) # hidden layer weights 
		self.w2 = 0.001 * np.random.randn(sizes[2], sizes[1]) # output layer weights

		# initialize random biases
		self.bias1 = np.random.randn(sizes[1], 1) # hidden layer biases 
		self.bias2 = np.random.randn(sizes[2], 1) # output layer biases

	def train(self, training_data, training_labels): 
		for i, instance in enumerate(training_data):
			print i
			# reshape array in order to easily calculate dot product
			instance = instance.reshape(3072, 1)

			# feedforward
			hidden_layer_input = np.dot(self.w1, instance) + self.bias1

			hidden_layer = Network.sigmoid(hidden_layer_input)
			output_layer = np.dot(self.w2, hidden_layer) + self.bias2

			# softmax probability distribution
			probability_distribution = Network.softmax(output_layer)

			# loss
			loss = self.cross_entropy_loss(probability_distribution, training_labels[i])

			# compute gradient for all weights from hidden layer to output layer
			output_layer_gradient = probability_distribution - training_labels[i]
			w2_gradient = np.dot(output_layer_gradient, hidden_layer.T)

			self.w2 -= w2_gradient * 0.001

			# compute gradient for all weights from input layer to hidden layer
			dloss_dz = output_layer_gradient
			dz_dy = self.w2

			dy_da = Network.sigmoid_derivative(hidden_layer_input)

			da_dw1 = instance

			dloss_dz__dz_dy = np.dot(dloss_dz.T, dz_dy)
			dloss_dz__dz_dy__dy_da = np.dot(dloss_dz__dz_dy.T, dy_da) 
			w1_gradient = np.dot(dloss_dz__dz_dy__dy_da, da_dw1.T)  

			self.w1 -= w1_gradient * 0.001

		return loss

	def cross_entropy_loss(self, probability_distribution, target_vector):
		"""
		Cross entropy loss function.
		"""
		return -np.sum((target_vector * np.log(probability_distribution)))

	@staticmethod
	def softmax(z):
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
		return np.exp(z) / np.sum(np.exp(z))

	@staticmethod
	@np.vectorize
	def sigmoid(x):
		""" 
		A function that takes a real-valued number and "squashes" it into range between 0 and 1, so
		that large negative numbers become 0 and large positive numbers become 1.  

		Parameters
		----------
		x : numpy.ndarray
			an array of real values, each of them is given as an argument to the sigmoid function.

		Returns
		-------
		numpy.ndarray
			an array of real values that are "squashed" into range between 0 and 1 by the function
		""" 
		return 1.0 / (1.0 + np.exp(-x))

	@staticmethod
	def sigmoid_derivative(x):
		return np.dot((1-Network.sigmoid(x)).T, Network.sigmoid(x))

root_path = '/home/sten/projects/neural_network/cifar_10_batches'
d = load_cifar_batches(root_path)

nn = Network([3072, 2000, 10])
x = nn.train(d['training_data'], d['training_labels'])