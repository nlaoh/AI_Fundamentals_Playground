from random import seed
from random import randrange
from math import exp
from csv import reader

import functions

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.arraysetops import unique

# 2. Load data/dataset¶
# Recall the steps to machine learning or supervised learning. Any machine learning algorithm needs input data to build a model. Thus,load a CSV file.
def load_csv(filename, skip = False):
    pass

# 3. Preprocess the data
# The first step is to extract the raw data from the dataset, both for X and y.
def extract_only_x_data(dataset):
    pass

def extract_only_y_data(dataset):
    pass

def flower_name_column_to_number(dataset, column):
    pass

# 4. Provide the supportive functions for training¶
# Define sigmoid.
def sigmoid(z):
    pass

# Define loss function
def loss(y, y_hat):
    pass

def gradients(X, y, y_hat):
    
    # X Input.
    # y true/target value.
    # y_hat predictions.
    # w weights.
    # b bias.
    
    # number of training examples.
    numner_of_examples = X.shape[0]
    
    # Gradient of loss weights.
    dw = (1/numner_of_examples)*np.dot(X.T, (y_hat - y))
    
    # Gradient of loss bias.
    db = (1/numner_of_examples)*np.sum((y_hat - y)) 
    
    return dw, db

# 5. Train the Logistic Regression
# Define the default weights and bias Train the data in a training loop Update the weights and bias by making predictions and getting the gradients of loss Output the weights, bias and losses
def train(X, y, batch_size, epochs, learning_rate):
    pass

def predict(X, w, b):
    pass

def accuracy(y, y_hat):
    pass

# 6. Output the results
# Output the plot
def plot_decision_boundary(X, w, b, xl, xr, yl, yr):
    
    # X Inputs
    # w weights
    # b bias
    
    fig = plt.figure(figsize=(10,8))
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "g^")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
    plt.xlim([xl, xr])
    plt.ylim([yl, yr])
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Decision Boundary')
    
    # The Line is y=mx+c
    # So, Equate mx+c = w.X + b
    # Solving we find m and c
    x1 = [min(X[:,0]), max(X[:,0])]
    
    if(w[1] != 0):
        m = -w[0]/w[1]
        c = -b/w[1]
        x2 = m*x1 + c
        plt.plot(x1, x2, 'y-')
    
    plt.show()

# 7. Evaluate the algorithm
# a. Example dataset
# In order to plot the model, let's take an example dataset
dataset = [[0.27810836,0.2550537003,0],
    [0.1465489372,0.2362125076,0],
    [0.3396561688,0.4400293529,0],
    [0.138807019,0.1850220317,0],
    [0.306407232,0.3005305973,0],
    [0.7627531214,0.2759262235,1],
    [0.5332441248,0.2088626775,1],
    [0.6922596716,0.177106367,1],
    [0.8675418651,-0.242068655,1],
    [0.7673756466,0.3508563011,1]]

X_train_data = functions.extract_only_x_data(dataset)
y_train_data = functions.extract_only_y_data(dataset)

X = np.array(X_train_data)
y = np.array(y_train_data)


# Training 
w, b, l = functions.train(X, y, batch_size=100, epochs=1000, learning_rate=0.01)
# Plotting Decision Boundary
plot_decision_boundary(X, w, b, -1, 1, -1, 1)

functions.accuracy(y, y_hat=functions.predict(X, w, b))

"""
This is a different dataset
"""

filename = 'iris.csv'
dataset = functions.load_csv(filename, skip=True)



functions.flower_name_column_to_number(dataset, 2)

X_train_data = functions.extract_only_x_data(dataset)
y_train_data = functions.extract_only_y_data(dataset)

X = np.array(X_train_data)
y = np.array(y_train_data)


# Training 
w, b, l = functions.train(X, y, batch_size=100, epochs=1000, learning_rate=0.01)
# Plotting Decision Boundary
plot_decision_boundary(X, w, b, 0, 6, 0, 2)

functions.accuracy(y, y_hat=functions.predict(X, w, b))