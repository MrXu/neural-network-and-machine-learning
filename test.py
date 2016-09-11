__author__ = 'xuwei'
from src import mnist_loader
from src import network

def test():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)



if __name__ == "__main__":
    test()




