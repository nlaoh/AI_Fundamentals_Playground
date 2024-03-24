# Step 1
# Import the libraries.
from random import seed
from random import randrange
from csv import reader
from math import exp
from csv import reader

# Step 2¶
# Import extra libraries, only needed for displaying the classification graph
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.arraysetops import unique

# Step 3
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

# Step 4
# Convert string diagnosis to number

# Assign the diagnosis of mailgnant (M) to 0 and assign the diagnosis of benign (B) to 1
def diagnosis_column_to_number(dataset, column):
    ###
    ### YOUR CODE HERE
    ###
    for i in range(0, len(dataset)): # Each row
        for j in range(0, len(dataset[0]) - 1): # Each column except right-most column, convert x values into integers
            dataset[i][j] = float(dataset[i][j].strip())
    for row in dataset:
        if row[column] == "M": # assign malignant values in dataset to 0
            row[column] = 0
        elif row[column] == "B": # assign benign values in dataset to 1
            row[column] = 1

# Step 5
# Extract only the x data
def extract_only_x_data(dataset):
    data = list()
    ###
    ### YOUR CODE HERE
    ###
    for i in range(0, len(dataset)): # Each row add a list
        data.append(list())
        for j in range(0, len(dataset[0]) - 1): # Each column except right-most column, extract x values
            data[i].append(dataset[i][j])

    return data

# Step 6
# Extract only the y data
def extract_only_y_data(dataset):
    data = list()
    ###
    ### YOUR CODE HERE
    ###
    for row in dataset:
        data.append(row[len(dataset[0]) - 1]) # Right-most column, extract as y values

    return data

# Step 7
# Define sigmoid function
def sigmoid(z):
###
### YOUR CODE HERE
###
    x = z
    y = 1 / (1 + np.exp(x * -1))
    # Return the value of the implemented sigmoid function, do not simply return z
    return y

# Step 8
# Define loss function
def loss(y, y_hat):
###
### YOUR CODE HERE
###
    loss = -np.log(1-y_hat)

    return loss

# Step 9
# Define gradients function
def gradients(X, y, y_hat):
    
    # X Input.
    # y true/target value.
    # y_hat predictions.
    # w weights.
    # b bias.
    
    # number of training examples.
    number_of_examples = X.shape[0]
    
    # Gradient of loss weights.
    dw = (1/number_of_examples)*np.dot(X.T, (y_hat - y)) # mean multiplication of y_hat - y?
    
    # Gradient of loss bias.
    db = (1/number_of_examples)*np.sum((y_hat - y)) # mean of sum of y_hat - y?
    
    return dw, db

# Step 10
# Train the dataset
def train(X, y, batch_size, epochs, learning_rate):
    
    # X Input.
    # y true/target value.
    # batch_size Batch Size.
    # epochs Number of iterations.
    # learning_rate Learning rate.
        
    # number of training examples
    # number of features 
    number_of_examples, number_of_features = X.shape # number of examples is number of rows, number of features is number of x variables
    
    # Initializing weights and bias to zeros.
    weights = np.zeros((number_of_features,1))
    bias = 0
    
    # Reshaping y.
    y = y.reshape(number_of_examples,1)
    # Empty list to store losses.
    losses = []
    
    # Training loop.
    for epoch in range(epochs): # epoch is a constant passed into the function, this is because the weights are random and need time to be optimised, so it is up to the user how many times the algorithm is run.
        for i in range((number_of_examples-1)//batch_size + 1):
            
            # Defining batches. SGD.
            start_i = i * batch_size
            end_i = start_i + batch_size
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]
            
            # Calculating hypothesis/prediction.
            y_hat = sigmoid(np.dot(xb, weights) + bias)
            print(y_hat)
            
            # Getting the gradients of loss w.r.t parameters.
            dw, db = gradients(xb, yb, y_hat)
            
            # Updating the parameters.
            ###
            ### YOUR CODE HERE
            ###
            weights -= learning_rate * dw
            bias -= learning_rate * db

        
        # Calculating loss and appending it in the list.
        l = loss(y, sigmoid(np.dot(X, weights) + bias)) # This is the actual equation formed by the model. So an example would be the equation for a line using b0, b1 in linear regression.
        losses.append(l)
        
    # returning weights, bias and losses(List).
    return weights, bias, losses

# Step 11
# Make the prediction
def predict(X, w, b):
    
    # X Input.
    
    # Calculating presictions/y_hat.
    preds = sigmoid(np.dot(X, w) + b)
    
    # Empty List to store predictions.
    pred_class = []
    
    # DELETE the following two lines and replace it with your own code
    # Otherwise leaving this code will pollute your predictions 
    # for i in preds:
    #     pred_class.append(0)
        
    # if y_hat >= 0.5 round up to 1
    # if y_hat < 0.5 round down to 0
    
    ###
    ### YOUR CODE HERE
    ###
    for prediction in preds:
        if prediction >= 0.5:
            pred_class.append(1)
        if prediction < 0.5:
            pred_class.append(0)

    return np.array(pred_class)

# Step 12
# Obtain the accuracy
def accuracy(y, y_hat):
    accuracy = 0
    ###
    ### YOUR CODE HERE
    ###
    accuracy = np.sum(y == y_hat) / len(y) # Create an array of true and false values using y == y_hat. Then sum up number of true values and divide by number of values for mean decision accuracy.
    return accuracy

# Step 13
# Output the plot
def plot_decision_boundary(X, w, b):
    
    # X Inputs
    # w weights
    # b bias
    
    fig = plt.figure(figsize=(10,8))
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "g^")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
    plt.xlim([-2, 2])
    plt.ylim([0, 2.2])
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

# Step 14
# Evaluate algorithm
filename = 'breast_cancer_data-1.csv'
# Each element is a row in the csv
dataset = load_csv(filename, skip=True)

diagnosis_column_to_number(dataset, 2)

X_train_data = extract_only_x_data(dataset)
y_train_data = extract_only_y_data(dataset)

X = np.array(X_train_data)
y = np.array(y_train_data)


# Training 
w, b, l = train(X, y, batch_size=100, epochs=1000, learning_rate=0.01)
# Plotting Decision Boundary
plot_decision_boundary(X, w, b)

accuracy(y, y_hat=predict(X, w, b))