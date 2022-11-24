from typing import List
import pandas as pd
import numpy as np
import random 

class Network:

    def __init__(self, input: int, output: int, hidden_layers: List[int], data=[]):
        self.size = {"input": input, "hidden_layers": hidden_layers, "output": output}
        self.weights = [None]*(len(hidden_layers)+1)
        self.weights[0] = np.random.rand(input,hidden_layers[0]) - 0.5
        self.weights[-1] = np.random.rand(hidden_layers[-1], output) - 0.5
        for i in range(1, len(hidden_layers)):
            self.weights[i] = np.random.rand(hidden_layers[i-1], hidden_layers[i]) - 0.5
        self.bias = [None]*(len(hidden_layers)+1)
        self.bias[-1] = np.random.rand(output) - 0.5
        for i in range(0, len(hidden_layers)):
            self.bias[i] = np.random.rand(hidden_layers[i]) - 0.5
        self.data = data.copy()
        self.unused_data = data.copy()

    def set_data(self, data):
        self.data = data.copy()
        self.unused_data = data.copy()

    def copy(self):
        copy = Network(self.size["input"], self.size["output"], self.size["hidden_layers"])
        copy.weights = [self.weights[i].copy() for i in range(len(self.weights))]
        copy.bias = [self.bias[i].copy() for i in range(len(self.bias))]
        copy.data = [self.data[i].copy() for i in range(len(self.data))]
        copy.unused_data = [self.unused_data[i].copy() for i in range(len(self.unused_data))]
        return copy

    def predict(self, x):
        n = len(self.weights)
        for i in range(n):
            x = x@self.weights[i]+self.bias[i]
            if i != n-1:
                x = ReLU(x)
            else:
                x = normalize(x)
        prediction = -1
        best_prediction = 0
        for i in range(len(x)):
            if x[i] > best_prediction:
                best_prediction = x[i]
                prediction = i
        return prediction

    def test(self, test_data):
        correct = 0
        n = test_data.shape[0]
        for i in range(n):
            one_test = test_data[i, :]
            if one_test[0] == self.predict(one_test[1:]):
                correct+=1
        return correct/n

def ReLU(x):
    return np.maximum(x,0)

def normalize(x):
    x1 = x.copy()
    s = np.sum(x1)
    for i in range(len(x1)):
        x1[i] = x1[i]/s
    return x1

def new_test_data(size):
    test_data = np.array(pd.read_csv('mnist_test.csv'))
    np.random.shuffle(test_data)
    return test_data[:size,:]

def next_gen(prev_gen):
    next_gen = prev_gen.copy()
    np.random.shuffle(next_gen)
    for i in range(len(prev_gen)//2):
        old_net1 = next_gen[i]
        old_net2 = next_gen[-i]
        new_net1 = old_net1.copy()
        new_net2 = old_net2.copy()
        new_net1.weights[-1] = old_net2.weights[-1].copy()
        new_net1.bias[-1] = old_net2.bias[-1].copy()
        new_net2.weights[-1] = old_net1.weights[-1].copy()
        new_net2.bias[-1] = old_net1.bias[-1].copy()
        next_gen.append(new_net1)
        next_gen.append(new_net2)
    return next_gen

def mutation(gen, rate):
    n = len(gen[0].size["hidden_layers"])+1
    gen.sort(key=(lambda x: x.test(test_data)), reverse=True)
    for i in range(len(gen)//10, len(gen)):
        one_net = gen[i]
        for layer in range(n):
            nw, mw = one_net.weights[layer].shape
            for li in range(nw):
                for col in range(mw):
                    if random.random()<rate:
                        one_net.weights[layer][li][col] = random.random() - 0.5
            nb = len(one_net.bias[layer])
            for li in range(nb):
                if random.random()<rate:
                    one_net.bias[layer][li] = random.random() - 0.5
    return gen

gen = [Network(784, 10, [16,16]) for _ in range(100)]
acc = 0
i = 0
while acc<70:
    test_data = new_test_data(1000)
    gen.sort(key=(lambda x: x.test(test_data)), reverse=True)
    gen = next_gen(gen[:len(gen)//2])
    gen = mutation(gen, 1/1000)
    gen.sort(key=(lambda x: x.test(test_data)), reverse=True)
    acc = gen[0].test(test_data)
    i+=1
    print(f"Gen. {i} : acc. {acc}")