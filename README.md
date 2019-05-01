# Neural Network for Mushroom Identification
Predicts whether or not a given mushroom is poisonous.

## Overview
This Neural Network takes, as an input, 22 attributes of a mushroom in the Agaricus
or Lepiota family. It then predicts whether the given mushroom is edible or 
poisonous. 

The data set was provided by the UCI Machine Learning Repository. You can find the
specific CSV file I worked with on Kaggle at this link:  https://www.kaggle.com/uciml/mushroom-classification
All data was sourced from The Audubon Society Field Guide to North American Mushrooms. (1981) 

## Description
This is a very simple neural network. There are no hidden layers and the error is defined
as the raw difference between the expected output and predicted output.

For efficency's sake and testing purposes, I only used 10 rows of data for training, but there 
are 8125 rows of data included in the CSV file, so more data can be used, or different data,
with very simple alterations. (The network becomes inaccurate around 100 rows of training data.)

## How to Read the Output
The program will output 'outputs after training.' After 200,000 iteratios, any 
value X where X = K * e^-1 can be taken as a 1, where K is any decimal number. 
Any value of X where X = K * e^(n < -1) can be taken as 0, where k is any decimal
number and n is any integer. Values where X = K * e^-1 almost always see K = 9.99...,
so hopefully the binary values of the predicted outputs are obvious. 

In any case, a value approximately equal to 1 represents a poisonous mushroom, and a
value approximately close to 0 represents an edible mushroom.

The newly added testing data will print the word poisonous or edible depending on its prediction. Please note that when setting a range for test data, if you enter rows 25 through 35 for example as [25:35], the test data will actually be rows 27 through 37. 


Please enjoy!
