# 1. Import the needed libraries
from math import sqrt
from matplotlib import pyplot as plot
from random import seed
from random import randrange
from csv import reader

import functions
import numpy as np

# 2. Load data/dataset
# Recall the steps to machine learning or supervised learning. 
# Any machine learning algorithm needs input data to build a model. Thus,load a CSV file.
def load_csv(filename, skip = False):
    dataset = list()
    # Opens the file in read only mode
    
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        
        # Skip the header, if needed
        if skip:
            next(csv_reader, None)
        
        for row in csv_reader:
            dataset.append(row)
            
    return dataset

# 3. Pre-process data: Convert string column to float¶
# Let's pre-process the data. Currently, the rows in the dataset are in string format. 
# So, we're converting the rows from string to float format. This will convert the string column into decimal number (float) and overwrite the data. 
# Note that the function strip() will remove the white spaces from the data.
def string_column_to_float(dataset, column):
    for row in dataset:
        # The strip() function remove white space
        # then convert the data into a decimal number (float)
        # and overwrite the original data
        row[column] = float(row[column].strip())

# 4. Make Prediction¶
# Input: X (np array), b (number), W (number)
# Output: (number) 
def predict(X, b, W) :    
    yhat = X.dot(W) + b # Linear line equation
    return yhat

# 5. Update the weights with the gradients and L2 penality
# Input: X (np array), Y (np array), b (number), W (number), no_of_training_examples (number), learning_rate (number),
#        l2_penality (number)
# Output: b (number), W (number)
def update_weights(X, Y, b, W, no_of_training_examples, learning_rate, l2_penality):
    Y_pred = predict(X, b, W)

    dW = (-(2 * (X.T).dot(Y - Y_pred)) + (2 * l2_penality * W)) / no_of_training_examples 
    # 2 is from the derivative, 2 moves down from the square
    db = -2 * np.sum(Y - Y_pred) / no_of_training_examples

    W -= learning_rate * dW
    b -= learning_rate * db

    return W, b

# 6. Linear Regression with L2 (Ridge) Regularisation¶
# Input: X (np array), Y (np array), iterations (number), learning_rate (number), l2_penality (number)
# Output: b (number), W (number)
def ridge_regression(X, Y, iterations = 1000, learning_rate = 0.01, l2_penality = 1):
    num_training_examples, num_features = X.shape
    W = np.zeros(num_features)
    b = 0

    for _ in range(iterations):
        W, b = update_weights(X, Y, b, W, num_training_examples, learning_rate, l2_penality)

    return W, b

# 7. Split the dataset into training and test sets¶
# Input: dataset (csv file instance), split (number)
# Output: X_train (np array), Y_train (np array), X_test (np array), Y_test (np array)
def train_test_split(dataset, split):
#     dataset = np.array(dataset)
#     split_index = int(len(dataset) * split)

#     train_data, test_data = dataset[:split_index], dataset[split_index:]

#     X_train, Y_train = train_data[:, :-1], train_data[:, -1]
#     X_test, Y_test = test_data[:, :-1], test_data[:, -1]

#     return X_train, Y_train, X_test, Y_test
    train = list()
    train_size = split * len(dataset)
    test = list(dataset)
    
    while len(train) < train_size:
        index = randrange(len(test))
        train.append(test.pop(index))
    train_x = list()
    train_y = list()
    for row in train:
        train_x.append(row[0])
        train_y.append(row[1])
    X_train = np.array(train_x)
    Y_train = np.array(train_y)
    
    test_x = list()
    test_y = list()
    for row in test:
        test_x.append(row[0])
        test_y.append(row[1])
    X_test = np.array(test_x)
    Y_test = np.array(test_y)

#     temp_X_train = list()
#     for i in range(0, len(train)): # Each row add a list
#         temp_X_train.append(list())
#         for j in range(0, len(train[0]) - 1): # Each column except right-most column, extract x values
#             temp_X_train[i].append(train[i][j])
#     X_train = np.array(temp_X_train)

#     temp_Y_train = list()
#     for row in train:
#         temp_Y_train.append(row[len(train[0]) - 1]) # Right-most column, extract as y values
#     Y_train = np.array(temp_Y_train)

#     temp_X_test = list()
#     for i in range(0, len(test)): # Each row add a list
#         temp_X_test.append(list())
#         for j in range(0, len(test[0]) - 1): # Each column except right-most column, extract x values
#             temp_X_test[i].append(train[i][j])
#     X_test = np.array(temp_X_test)

#     temp_Y_test = list()
#     for row in test:
#         temp_Y_test.append(row[len(test[0]) - 1]) # Right-most column, extract as y values
#     Y_test = np.array(temp_X_test)

    return X_train, Y_train, X_test, Y_test


# 8. Perform regression algorithm on dataset
def evaluate_ridge_regression(dataset, split):
    
    X_train, Y_train, X_test, Y_test = functions.train_test_split(dataset, split)
    
    # Train the model
    
    b, W = functions.ridge_regression(X_train, Y_train, iterations = 10000, l2_penality = 0.01)
    
    # Make a prediction with the model
    
    yhat = functions.predict(X_test, b, W)
        
    print( "Predicted values ", np.round( yhat[:3], 2 ) )
    print( "Real values      ", Y_test[:3] ) 
    print( "Trained W        ", round( W[0], 2 ) )    
    print( "Trained b        ", round( b, 2 ) )
    
    visualise(X_test, Y_test, yhat)
# 9. Visualise the results
def visualise(X_test, Y_test, yhat):
    plot.scatter( X_test, Y_test, color = 'blue' )    
    plot.plot( X_test, yhat, color = 'orange' )    
    plot.title( 'Salary vs Experience' )    
    plot.xlabel( 'Years of Experience' )    
    plot.ylabel( 'Salary' )    
    plot.show()
# 10. Seed the random value¶
seed(1)
# 11. Load and prepare data¶
filename = 'salary_data.csv'
dataset = load_csv(filename, True) 

for i in range(len(dataset[0])):
    string_column_to_float(dataset, i)
# 12. Evaluate algorithm
split = 0.66

evaluate_ridge_regression(dataset, split)