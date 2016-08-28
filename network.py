import numpy as np

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
		self.biases = [np.random.randn(y,1) for y in sizes[1:]] 
		# weights[0]: weights connecting first layer and second layer, 
		# np.random.randn(y,x) generates matrix of y rows(number of nodes in back layer) x columns(number of nodes in front layer)
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
		print self.biases
		print self.weights



if __name__ == "__main__":
	sizes = [4,3,2]
	network = Network(sizes)
		