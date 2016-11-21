#coding:utf-8
#KERAS_BACKEND=tensorflow python -c "from keras import backend; print(backend._BACKEND)"
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import copy

class LSTM():
    def __init__(self):

        self.binary_dim = 8
        self.largest_number = pow(2, self.binary_dim)
        self.int2binary = self.generate_data()
        self.alpha = 0.1
        self.input = 2
        self.hidden = 16
        self.output = 1
        self.training_epoches = 10

        self.weights, self.weights_update = self.init_weights()

    def generate_data(self):
        int2binary = {}
        """
        a = np.array([[2], [7], [23]], dtype=np.uint8)
        b = np.unpackbits(a, axis=1)
        b:
            array([[0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1, 1, 1],
                   [0, 0, 0, 1, 0, 1, 1, 1]], dtype=uint8)
        """
        binary = np.unpackbits(np.array([range(self.largest_number)], dtype=np.uint8).T, axis=1)

        for i in range(self.largest_number):
            int2binary[i] = binary[i]
        return int2binary

        #print(int2binary)
        #print(binary, np.shape(binary))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_to_derivative(self, sigmoid_output):
        return sigmoid_output * (1 - sigmoid_output)

    def init_weights(self):
        weights = {
            'synapse_0' : 2 * np.random.random((self.input, self.hidden)) - 1,
            'synapse_1' : 2 * np.random.random((self.hidden, self.output)) - 1,
            'synapse_h' : 2 * np.random.random((self.hidden, self.hidden)) - 1,
        }
        weights_update = {
            'synapse_0_update' : np.zeros_like(weights['synapse_0']),
            'synapse_1_update' : np.zeros_like(weights['synapse_1']),
            'synapse_h_update' : np.zeros_like(weights['synapse_h'])
        }

        return weights, weights_update

    def train(self):
        for i in range(self.training_epoches):
            a_int = np.random.randint(self.largest_number/2)
            a = self.int2binary[a_int]
            b_int = np.random.randint(self.largest_number/2)
            b = self.int2binary[b_int]
            c_int = a_int + b_int
            c = self.int2binary[c_int]
            #print(a)
            #print(b)
            #print(c)
            #print('\n')
            d = np.zeros_like(c)

            loss = 0.

            layer_2_deltas = list()
            layer_1_values = list()
            layer_1_values.append(np.zeros(self.hidden))

            for position in range(self.binary_dim):
                X = np.array([[a[self.binary_dim -  position - 1], b[self.binary_dim - position - 1]]])
                y = np.array([[c[self.binary_dim - position - 1]]]).T
                print('x: ', X)
                print('y: ', y)
                layer_1 = self.sigmoid(np.dot(X, self.weights['synapse_0']) + np.dot(layer_1_values[-1], self.weights['synapse_h']))
                layer_2 = self.sigmoid(np.dot(layer_1, self.weights['synapse_1']))

                layer_2_error = y - layer_2
                layer_2_deltas.append((layer_2_error) * self.sigmoid_to_derivative(layer_2))
                loss += np.abs(layer_2_error[0])

                d[self.binary_dim - position - 1] = np.round(layer_2[0][0])
                layer_1_values.append(copy.deepcopy(layer_1))
            future_layer_1_delta = np.zeros(self.hidden)



def _test():
    lstm = LSTM()
    lstm.generate_data()
    lstm.train()

_test()



