import numpy as np

class NeuralNetworks():
    def __init__(self):
        self.X = np.array(
            [
                [0, 1],
                [0, 1],
                [1, 0],
                [1, 0]
            ]
        )

        self.y = np.array([[0, 0, 1, 1]]).T

        self.traning_epoches = 100

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_to_derivative(self, sigmoid_output):
        return sigmoid_output * (1 - sigmoid_output)

    def train(self):
        np.random.seed(1)
        # initialize weights randomly with mean 0
        synapse_0 = 2 * np.random.random((2, 1)) - 1

        for iter in xrange(self.traning_epoches):
            layer_0 = self.X
            layer_1 = self.sigmoid(np.dot(layer_0, synapse_0))
            #print(layer_0)
            #print(synapse_0)
            #print(layer_1)
            layer_1_error = layer_1 - self.y
            layer_1_delta = layer_1_error * self.sigmoid_to_derivative(layer_1)
            synapse_0_derivative = np.dot(layer_0.T, layer_1_delta)
            synapse_0 -= synapse_0_derivative
            print('layer_error:', layer_1_error)
            print('layer_1_delta:', layer_1_delta)
            print('synapsse_0_derivative:', synapse_0_derivative)
            print('synapse_0', synapse_0)
        print(layer_1)
        print(synapse_0)
        return synapse_0

    def test(self, x, synapse):
        pass

def test():
    nn = NeuralNetworks()
    nn.train()

test()
