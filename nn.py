import pdb
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_cifar_batches

class Network():
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes

		# initialize random weights
		self.w1 = 0.001 * np.random.randn(sizes[1], sizes[0]) # hidden layer weights 
		self.w2 = 0.001 * np.random.randn(sizes[2], sizes[1]) # output layer weights

		# initialize biases 
		self.b1 = 0.001 * np.random.randn(sizes[1], 1)
		self.b2 = 0.001 * np.random.randn(sizes[2], 1)

	def predict(self, test_data):
		predictions = []

		for x in test_data:
			x = x.reshape(3072, 1)
			y = Network.sigmoid(np.dot(self.w1, x) + self.b1)
			z = np.dot(self.w2, y) + self.b2

			predictions.append(z.argmax())

		return np.array(predictions)

	def train(self, training_data, training_labels, batch_size=10, epochs=1, learning_rate=0.1):
		data_mini_batches = [training_data[n:n+batch_size] for n in range(0, len(training_data), batch_size)]
		label_mini_batches = [training_labels[n:n+batch_size] for n in range(0, len(training_labels), batch_size)]
		
		# calculate batches in 1/10 epoch for plotting 
		batches_in_tenth_epoch = len(training_data) / 10 / batch_size
		tenth_epoch_losses = []
		tenth_epoch_loss = 0

		total_instances = len(training_data) * epochs

		for epoch in range(epochs):
			for i, batch in enumerate(data_mini_batches):
				# print how many percentage till finished training 
				percentage = round(float(i * batch_size + epoch * len(training_data)) / total_instances * 100, 2) 
				print str(percentage) + "%" 

				# FEEDFORWARD
				# reshape array in order to easily calculate dot product
				x = batch.T
				y = Network.sigmoid(np.dot(self.w1, x) + self.b1)
				z = np.dot(self.w2, y) + self.b2
				p = np.apply_along_axis(self.softmax, 0, z)

				# calculate loss
				loss = np.zeros(batch_size)
				for j, (a, b) in enumerate(zip(p.T, label_mini_batches[i])):
					loss[j] = self.cross_entropy_loss(a, b)

				batch_loss = sum(loss) / batch_size
				tenth_epoch_loss += batch_loss

				if not epoch and not i:
					tenth_epoch_losses.append(tenth_epoch_loss)
					tenth_epoch_loss = 0
				elif not i % batches_in_tenth_epoch:
					tenth_epoch_losses.append(tenth_epoch_loss / batches_in_tenth_epoch)
					tenth_epoch_loss = 0

				# BACKPROPAGATION
				# output layer (z) gradient
				dloss_dz = p - label_mini_batches[i].T
				dloss_dw2 = np.dot(dloss_dz, y.T)

				# hidden layer (y) gradient
				dloss_dy = np.dot(self.w2.T, dloss_dz)
				dy_da = self.sigmoid_derivative(y)
				dloss_da = dloss_dy * dy_da
				dloss_dw1 = np.dot(dloss_da, x.T)

				# perform weight update
				self.w1 -= dloss_dw1 * learning_rate
				self.w2 -= dloss_dw2 * learning_rate

				# perform bias update
				self.b1 -= np.sum(dloss_da, axis=1).reshape(self.sizes[1], 1) * learning_rate
				self.b2 -= np.sum(dloss_dz, axis=1).reshape(self.sizes[2], 1) * learning_rate

		# plot epoch losses 
		plt.plot(range(len(tenth_epoch_losses)), tenth_epoch_losses)
		plt.xlabel("1/10 epoch")
		plt.ylabel("Loss")
		plt.show()

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
			an array of real values that are "squashed" into range between 0 and 1 by the function.
		""" 
		if x >= 0:
			z = np.exp(-x)
			return 1 / (1 + z)
		else:
			# if x is less than zero then z will be small, denominator can't be zero because it's 1+z
			z = np.exp(x)
			return z / (1 + z)

	def sigmoid_derivative(self, y):
		return self.sigmoid(y) * (1 - self.sigmoid(y))

	@staticmethod
	def test_accuracy(x, y):
		count = 0
		for i, prediction in enumerate(x):
			if (prediction == y[i]):
				count += 1

		return float(count) / x.shape[0]

root_path = '/home/sten/projects/neural_network/cifar_10_batches'
d = load_cifar_batches(root_path)

nn = Network([3072, 1541, 10])
nn.train(d['training_data'], d['training_labels'])

a = nn.predict(d['test_data'])
b = d['test_labels']
print "Test accuracy: " + str(Network.test_accuracy(a, b))