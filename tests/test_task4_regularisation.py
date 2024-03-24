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
    yhat = X.dot(W) + b # Linear line equation
    return yhat


# Step 5
# Update the weights with the gradients and L2 penality
def update_weights(X, Y, b, W, no_of_training_examples, learning_rate, l2_penality):
    ###
    ### YOUR CODE HERE
    ###
    # dW = (-2 / no_of_training_examples) * X.T.dot(Y - (X.dot(W) + b)) + 2 * l2_penality * W
    # db = (-2 / no_of_training_examples) * np.sum(Y - (X.dot(W) + b))
    # W -= learning_rate * dW
    # b -= learning_rate * db
    Y_pred = predict(X, b, W)

    dW = (-(2 * (X.T).dot(Y - Y_pred)) + (2 * l2_penality * W)) / no_of_training_examples 
    # 2 is from the derivative, 2 moves down from the square
    db = -2 * np.sum(Y - Y_pred) / no_of_training_examples

    W -= learning_rate * dW
    b -= learning_rate * db

    return W, b

# Step 6
# Linear Regression with L2 (Ridge) Regularisation
def ridge_regression(X, Y, iterations = 1000, learning_rate = 0.01, l2_penality = 1):
    ###
    ### YOUR CODE HERE
    ### 
    num_training_examples, num_features = X.shape
    W = np.zeros(num_features) # Initialise weights as zeros
    b = 0 # Initialise bias as zero

    for _ in range(iterations):
        W, b = update_weights(X, Y, b, W, num_training_examples, learning_rate, l2_penality)

    return W, b
    # num_features = X.shape[1]
    # W = np.zeros(num_features)  # Initialize weights as zeros
    # b = 0  # Initialize bias as zero

    # # no_of_training_examples = X.shape[0]

    # for _ in range(iterations):
    #     W, b = update_weights(X, Y, b, W, num_training_examples, learning_rate, l2_penality)
    #     # predictions = X.dot(W) + b
    #     # dW = (-2 / no_of_training_examples) * X.T.dot(Y - predictions) + 2 * l2_penalty * W
    #     # db = (-2 / no_of_training_examples) * np.sum(Y - predictions)

    #     # W -= learning_rate * dW  # Update weights
    #     # b -= learning_rate * db  # Update bias
    # return b, W

# Step 7
# Split the data into training and test sets
def train_test_split(dataset, split):
    ###
    ### YOUR CODE HERE
    ###
    # dataset = np.array(dataset)
    # split_index = int(len(dataset) * split)

    # train_data, test_data = dataset[:split_index], dataset[split_index:]

    # X_train, Y_train = train_data[:, :-1], train_data[:, -1]
    # X_test, Y_test = test_data[:, :-1], test_data[:, -1]

    # return X_train, Y_train, X_test, Y_test

    train = list()
    train_size = split * len(dataset)
    test = list(dataset)
    
    while len(train) < train_size:
        index = randrange(len(test))
        train.append(test.pop(index))

    temp_X_train = list()
    for i in range(0, len(train)): # Each row add a list
        temp_X_train.append(list())
        for j in range(0, len(train[0]) - 1): # Each column except right-most column, extract x values
            temp_X_train[i].append(train[i][j])
    X_train = np.array(temp_X_train)

    temp_Y_train = list()
    for row in train:
        temp_Y_train.append(row[len(train[0]) - 1]) # Right-most column, extract as y values
    Y_train = np.array(temp_Y_train)

    temp_X_test = list()
    for i in range(0, len(test)): # Each row add a list
        temp_X_test.append(list())
        for j in range(0, len(test[0]) - 1): # Each column except right-most column, extract x values
            temp_X_test[i].append(train[i][j])
    X_test = np.array(temp_X_test)

    temp_Y_test = list()
    for row in test:
        temp_Y_test.append(row[len(test[0]) - 1]) # Right-most column, extract as y values
    Y_test = np.array(temp_X_test)

    return X_train, Y_train, X_test, Y_test

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
    print( "Trained W        ", round( W[0], 2 ) )    
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