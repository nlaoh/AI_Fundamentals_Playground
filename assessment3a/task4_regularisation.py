# Step 1
# Import the needed libraries.
from math import sqrt
from matplotlib import pyplot as plot
from random import seed
from random import randrange
from csv import reader
import numpy as np

# Step 2
# Load a CSV file
def load_csv(filename, skip = False):
    ###
    ### YOUR CODE HERE
    ###
    dataset = list()

    with open(filename, 'r') as file:
        csv_reader = reader(file)
        if skip:
            next(csv_reader, None)

        for row in csv_reader:
            dataset.append(row)

    return dataset

# Step 3¶
# Convert string column to float
def string_column_to_float(dataset, column):
    for row in dataset:
        # The strip() function remove white space
        # then convert the data into a decimal number (float)
        # and overwrite the original data
        row[column] = float(row[column].strip())

# Step 4
# Make Prediction
def predict(X, b, W) : # X is dataset? b 
    ###
    ### YOUR CODE HERE
    ###
    # yhat = X.dot(W) + b # Linear line equation
    yhat = x * W + b
    return yhat


# Step 5
# Update the weights with the gradients and L2 penality
def update_weights(X, Y, b, W, no_of_training_examples, learning_rate, l2_penality):
###
### YOUR CODE HERE
###
    Y_pred = predict(X, b, W)

    dW = (-(2 * (X.T).dot(Y - Y_pred)) + (2 * l2_penality * W)) / no_of_training_examples 
    # 2 is from the derivative, 2 moves down from the square
    db = -2 * np.sum(Y - Y_pred) / no_of_training_examples

    W = W - learning_rate * dW
    b = b - learning_rate * db

    return W, b

# Step 6
# Linear Regression with L2 (Ridge) Regularisation
def ridge_regression(X, Y, iterations = 1000, learning_rate = 0.01, l2_penality = 1):
###
### YOUR CODE HERE
###
#     num_training_examples, num_features = X.shape
#     W = np.zeros(num_features) # Initialise weights as zeros
    num_training_examples = len(X) 
    W = 0
    b = 0 # Initialise bias as zero

    for _ in range(iterations):
        W, b = update_weights(X, Y, b, W, num_training_examples, learning_rate, l2_penality)

    return W, b

# Step 7
# Split the data into training and test sets
def train_test_split(dataset, split):
    ###
    ### YOUR CODE HERE
    ###
    X_list = list()
    Y_list = list()
    
    for row in dataset:
        X_list.append(list())
        X_list[-1].append(row[0])
        # print("X_list has this row: {}", row) #Testing
    
    for row in dataset:
        Y_list.append(list())
        Y_list[-1].append(row[1])
        # print("Y_list has this row: {}", row) #Testing
    
    # Create training and testing sets
    training_size = split * len(dataset)
    X_train = list()
    Y_train = list()
    X_test = list(X_list)
    Y_test = list(Y_list)
    
    while len(X_train) < training_size:
        index = randrange(len(X_test))
        X_train.append(X_test.pop(index))
        Y_train.append(Y_test.pop(index))
    
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

# Step 8
# Perform regression algorithm on dataset
def evaluate_ridge_regression(dataset, split):
    
    # Spilt the data in training and test sets
    # And split further in X_train, Y_train, X_test, Y_test    

    X_train, Y_train, X_test, Y_test = train_test_split(dataset, split)

    # Train the model
        
    W, b = ridge_regression(X_train, Y_train, iterations = 10000, l2_penality = 0.001)
    
    # Make a prediction with the model
    
    yhat = predict(X_test, b, W)
    
    print(W)
        
    print( "Predicted values ", np.round( yhat[:3], 2 ) )
    print( "Real values      ", Y_test[:3] )
    print( "Trained W        ", np.round( W[0], 2 ) )    
    print( "Trained b        ", round( b, 2 ) )
    
    visualise(X_test, Y_test, yhat)

# Step 9
# Visualise the results
def visualise(X_test, Y_test, yhat):
    plot.scatter( X_test, Y_test, color = 'blue' )    
    plot.plot( X_test, yhat, color = 'orange' )    
    plot.title( 'Fertility Rate vs Worker Percentage' )    
    plot.xlabel('Fertility Rate')
    plot.ylabel('Worker Percentage')
    plot.show()

# Step 10
# Seed the random value
seed(1)

# Step 11
# Load and prepare data
filename = 'fertility_rate-worker_percent.csv'
dataset = load_csv(filename, skip=True)

for i in range(len(dataset[0])):
    string_column_to_float(dataset, i)

# Step 12
# Evaluate regression algorithm on training dataset
split = 0.66

evaluate_ridge_regression(dataset, split)