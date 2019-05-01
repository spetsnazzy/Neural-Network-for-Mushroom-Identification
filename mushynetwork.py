# Mycological data gathered from the UCI
# online Machine Learning Repository
# Data sourced from The Audubon Society
# Field Guide to North American Mushrooms (1981)
# All data includes samples corresponding to 23
# species of gilled mushrooms in the Agaricus and
# Lepiota Family. This network identifies whether
# a given sample is edible or poisonous based
# on 22 attributes.
# An output value approximately equal to 1
# represents a poisonous mushroom, while an
# output value approximately equal to 0 represents
# an edible mushroom.

import csv
import numpy as np
import random
################################################
# Creates Mushroom Data

with open('mushrooms.csv') as file:
    mushy_data = list(csv.reader(file))

#################################################
# Sigmoid and Derivative definitions for
# normalizing values and backprop
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def enumerator(input_data):
    for each in input_data:
        input_data[input_data == 'a'] = 0
        input_data[input_data == 'b'] = 1 / 26
        input_data[input_data == 'c'] = 2 / 26
        input_data[input_data == 'd'] = 3 / 26
        input_data[input_data == 'e'] = 4 / 26
        input_data[input_data == 'f'] = 5 / 26
        input_data[input_data == 'g'] = 6 / 26
        input_data[input_data == 'h'] = 7 / 26
        input_data[input_data == 'i'] = 8 / 26
        input_data[input_data == 'j'] = 9 / 26
        input_data[input_data == 'k'] = 10 / 26
        input_data[input_data == 'l'] = 11 / 26
        input_data[input_data == 'm'] = 12 / 26
        input_data[input_data == 'n'] = 13 / 26
        input_data[input_data == 'o'] = 14 / 26
        input_data[input_data == 'p'] = 15 / 26
        input_data[input_data == 'q'] = 16 / 26
        input_data[input_data == 'r'] = 17 / 26
        input_data[input_data == 's'] = 18 / 26
        input_data[input_data == 't'] = 19 / 26
        input_data[input_data == 'u'] = 20 / 26
        input_data[input_data == 'v'] = 21 / 26
        input_data[input_data == 'w'] = 22 / 26
        input_data[input_data == 'x'] = 23 / 26
        input_data[input_data == 'y'] = 24 / 26
        input_data[input_data == 'z'] = 25 / 26
        input_data[input_data == '?'] = 26 / 26

    return input_data

#################################################

# Taking the mushroom data and splitting it into
# input data & output data for training
mushy_matrix = np.array(mushy_data)

# Initialize the training inputs
training_inputs = np.delete(mushy_matrix, 0, 1)
training_inputs = np.delete(training_inputs, 0, 0)
training_inputs = training_inputs[:25]

# Initialize the known outputs for the corresponding training inputs
training_outputs = mushy_matrix[:, [0]]
training_outputs = np.delete(training_outputs, 0, 0)
training_outputs = training_outputs[:25]

# Initialize testing Data
test_data = np.delete(mushy_matrix, 0, 1)
test_data = np.delete(test_data, 0, 0)
# Test data is 2 rows off. (A row index of 21 will refer to row 23)
test_data = test_data[21]
enumerator(test_data)
print("Test data")
print(test_data)

# Changes output field to numerical inputs
for each in training_outputs:
    training_outputs[training_outputs == 'p'] = 1
    training_outputs[training_outputs == 'e'] = 0

enumerator(training_inputs)

training_inputs = training_inputs.astype(np.float)
training_outputs = training_outputs.astype(np.float)
test_data = test_data.astype(np.float)

# Choosing a random seed for the weights for consistency
# np.random.seed(1)

# Initialize random synaptic weights between -1 and 1
synaptic_weights = 2 * np.random.random((22,1)) - 1
print("Random synaptic weights: ")
print(synaptic_weights)

for iteration in range(250000):

    input_layer = training_inputs

    # Forward Pass
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # Calcs how far off the output was
    error = training_outputs - outputs

    # How we should change each weight to get a better output
    adjustments = error * sigmoid_derivative(outputs)

    # Changes the weights
    synaptic_weights += np.dot(input_layer.T, adjustments)

print("Synaptic weights after training: ")
print(synaptic_weights)

print("Outputs after training: ")
print(outputs)

print("Desired Outputs: ")
print(training_outputs)

# Running the testing data through the network
input_layer = test_data
outputs = sigmoid(np.dot(input_layer, synaptic_weights))

# Transforms test data output into readable format
print("\nTest Data: ")
for each in outputs:
    if(each >= .5):
        print("poisonous")
    else:
        print("edible")
