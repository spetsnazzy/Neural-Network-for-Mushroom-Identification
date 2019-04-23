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

#################################################

# Taking the mushroom data and splitting it into
# input data & output data for training
mushy_matrix = np.array(mushy_data)

# Initialize the training inputs
training_inputs = np.delete(mushy_matrix, 0, 1)
training_inputs = np.delete(training_inputs, 0, 0)
training_inputs = training_inputs[:10]

# Initialize the known outputs for the corresponding training inputs
training_outputs = mushy_matrix[:, [0]]
training_outputs = np.delete(training_outputs, 0, 0)
training_outputs = training_outputs[:10]

# Changes input fields to numerical inputs
for each in training_outputs:
    training_outputs[training_outputs == 'p'] = 1
    training_outputs[training_outputs == 'e'] = 0

for each in training_inputs:
    training_inputs[training_inputs == 'a'] = 0
    training_inputs[training_inputs == 'b'] = 1 / 26
    training_inputs[training_inputs == 'c'] = 2 / 26
    training_inputs[training_inputs == 'd'] = 3 / 26
    training_inputs[training_inputs == 'e'] = 4 / 26
    training_inputs[training_inputs == 'f'] = 5 / 26
    training_inputs[training_inputs == 'g'] = 6 / 26
    training_inputs[training_inputs == 'h'] = 7 / 26
    training_inputs[training_inputs == 'i'] = 8 / 26
    training_inputs[training_inputs == 'j'] = 9 / 26
    training_inputs[training_inputs == 'k'] = 10 / 26
    training_inputs[training_inputs == 'l'] = 11 / 26
    training_inputs[training_inputs == 'm'] = 12 / 26
    training_inputs[training_inputs == 'n'] = 13 / 26
    training_inputs[training_inputs == 'o'] = 14 / 26
    training_inputs[training_inputs == 'p'] = 15 / 26
    training_inputs[training_inputs == 'q'] = 16 / 26
    training_inputs[training_inputs == 'r'] = 17 / 26
    training_inputs[training_inputs == 's'] = 18 / 26
    training_inputs[training_inputs == 't'] = 19 / 26
    training_inputs[training_inputs == 'u'] = 20 / 26
    training_inputs[training_inputs == 'v'] = 21 / 26
    training_inputs[training_inputs == 'w'] = 22 / 26
    training_inputs[training_inputs == 'x'] = 23 / 26
    training_inputs[training_inputs == 'y'] = 24 / 26
    training_inputs[training_inputs == 'z'] = 25 / 26
    training_inputs[training_inputs == '?'] = 26 / 26

training_inputs = training_inputs.astype(np.float)
training_outputs = training_outputs.astype(np.float)

# Choosing a random seed for the weights for consistency
# np.random.seed(1)

# Initialize random synaptic weights between -1 and 1
synaptic_weights = 2 * np.random.random((22,1)) - 1
print("Random synaptic weights: ")
print(synaptic_weights)

for iteration in range(200000):

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
