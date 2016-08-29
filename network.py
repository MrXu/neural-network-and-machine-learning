import numpy as np
import random

class Network(object):
	"""
	docstring for Network
	sizes: number of neurons in respective layers
	sizes=[4,3,2]: first layer 4 nodes, second layer 3 nodes, third layer 2 nodes
	"""
	def __init__(self, sizes):
		self.num_layers =len(sizes)
		self.sizes = sizes

		# intialize bias and weights randomly
		# first layer does not need bias
		'''
		[
			array([[-0.77373517],[-0.17805081],[ 0.52246022]]), 
			array([[ 1.22777508],[ 0.27331014]])
		]

		'''
		self.biases = [np.random.randn(y,1) for y in sizes[1:]] 

		# weights[0]: weights connecting first layer and second layer, 
		# np.random.randn(y,x) generates matrix of y rows(number of nodes in back layer) x columns(number of nodes in front layer)
		'''
		[
			array([[-0.39626801,  0.35554345,  0.30420648,  0.21139611],[ 0.34649084,  1.76863201, -0.97616125,  0.45150293],[ 0.96022086, -0.89792873,  0.59252072,  1.82106298]]), 
			array([[-0.92155612,  0.00737503, -1.52180064], [-0.26195647,  0.65326624,  0.07236324]])
		]
		'''
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
		print self.biases
		print self.weights


	# input a is an (n,1) numpy ndarray
	def feedforward(self, a):
		for bias, weight in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a)+b)
		return a


	# stochastic gradient descent
	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
		"""
		Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.

        epochs: number of epochs to train
        eta: learning rate
        """
        if test_data:
        	n_test = len(test_data)
        n = len(training_data)

        for j in xrange(epochs):
        	random.shuffle(training_data)
        	mini_batches = [ training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size) ]
    		for mini_batch in mini_batches:
    			self.update_mini_batch(mini_batch, eta)
    		if test_data:
    			print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)

    		else:
    			print "Epoch {0} complete".format(j)


    



# generate sigmoid
def sigmoid(z):
	'''
	 when the input z is a vector or Numpy array, 
	 Numpy automatically applies the function sigmoid elementwise, that is, in vectorized form.
	'''
	return 1.0/(1.0+np.exp(-z))



if __name__ == "__main__":
	sizes = [4,3,2]
	network = Network(sizes)
		