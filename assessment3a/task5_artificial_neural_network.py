# Step 1
# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
from csv import reader

# Step 2
# Load the big heart CSV file
def load_csv(filename, skip = False):
    dataset = list()

    with open(filename, 'r') as file:
        csv_reader = reader(file)
        if skip:
            next(csv_reader, None)

        for row in csv_reader:
            dataset.append(row)

    return dataset

# Step 3
# extract only x data
def extract_only_x_data(dataset):
    data = list()

    for i in range(len(dataset)):
        data.append(list())

        for j in range(len(dataset[i]) - 1):
            data[-1].append(float(dataset[i][j]))

    return data

# Step 4
# extract only y data
def extract_only_y_data(dataset):
    data = list()

    for i in range(len(dataset)):
        data.append(int(dataset[i][-1]))

    return data
# Step 5
# Defining the Config that contains the inputs for the network, the outputs for the network as well as the parameters for gradient descent
class Config:
    #Specify inpulayer dimensionality,  output layer dimensionality, learning rate for gradient descent, regularization strength

    ###
    ### YOUR CODE HERE
    ###
    input_dimension = 2
    output_dimension = 1
    learning_rate = 0.04
    regularisation_strength = 0.0005

    

# Step 6
# Load the moons.csv data, extract the data and convert it into num py arrays
def generate_data():
    filename = 'moons.csv'

    dataset = load_csv(filename)                            # load the data from a csv file

    x_data = extract_only_x_data(dataset)                   # extract the input data (2 inputs)
    y_data = extract_only_y_data(dataset)                   # extract the output data (1 output)
    
    X = np.array(x_data)                                    # convert the data into np array
    y = np.array(y_data)                                    # convert the data into np array

    return X, y                                             # returns both input (X) data and output (y) data

# Step 7
# Assign the X and y with the generated data
X, y = generate_data()

# Step 8
# Draw the scatter plot
plt.figure(figsize=(10,7))                              # Defines the size of the plot
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral)  # Draws a scatter plot

# Step 9
# Define a activation function of your choice.
def activation_function(x):
    ###
    ### YOUR CODE HERE
    ###
    # Return the value of the implemented activation function, do not simply return x
    y = 1 / (1 + np.exp(x * -1))
    return y

# Step 10
# Define a lost function
def loss(a):
    ###
    ### YOUR CODE HERE
    ###
    # Return the value of the implemented loss function, do not simply return a
    loss = (-1.0 / (len(a))) * np.sum((a - activation_function(a)) ** 2)
    return loss

# Step 11
# Define a weight regularization function
def weight_regularization(Wx, dWx):
    ###
    ### YOUR CODE HERE
    ###
    # Return the value of the implemented weight regularization function, do not simply return dWx
    #TODO:
    return dWx

# Step 12
# Define forward propagation function
def forward_propagation(X, W1, b1, W2, b2):
    ###
    ### YOUR CODE HERE
    ###
    z1 = np.dot(X, W1) + b1
    a1 = activation_function(z1)
    z2 = np.dot(a1, W2) + b2
    return np.exp(z2)

# Step 13
# Define backward propagation function
def backward_propagation(X, y, W1, b1, W2, b2, exp_scores, number_of_examples):
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
    delta3 = probs
    delta3[range(number_of_examples), y] -= 1

    z2 = np.dot(X, W1) + b1
    a1 = np.tanh(z2)

    dW2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)

    delta2 = delta3.dot(W2) * (1 - np.power(a1, 2))
    
    dW1 = np.dot(X.T, delta2)
    
    # Add regularization terms (b1 and b2 don't have regularization terms)
    db1 = np.sum(delta2, axis=0)
    ###
    ### YOUR CODE HERE
    ###
    dW1 = dW1 * (1 - Config.regularisation_strength)
    dW2 = dW2 * (1 - Config.regularisation_strength)
    return dW1, dW2, db1, db2

# Step 14
# Define the loss function, used to evaluate how well the model is doing
def calculate_loss(model, X, y):
   
    number_of_examples = len(X)
    
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    
    exp_scores = forward_propagation(X, W1, b1, W2, b2)
    
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Calculating the loss
    
    correct_probs = loss(probs[range(number_of_examples), y])
    
    data_loss = np.sum(correct_probs)
    
    data_loss = data_loss + Config.regularisation_strength / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    
    return 1. / number_of_examples * data_loss

# Step 15
# Make a prediction
def predict(model, X):
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    
    exp_scores = forward_propagation(X, W1, b1, W2, b2)
    
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    return np.argmax(probs, axis=1)

# Step 16
# Build the neural network using batch gradient descent using the backpropagation derivates
def build_model(X, y, number_of_nodes_within_hidden_layer, passes=20000, print_loss=False):

    # Initialize the parameters to random values. We need to learn these
    
    
    number_of_examples = len(X)
    np.random.seed(0)
    
    # Two weights are needed as the network has two inputs
    # Likewise, two biases are needed as the network has two inputs

    ###
    ### YOUR CODE HERE
    ###
    W1 = np.random.randn(Config.input_dimension, number_of_nodes_within_hidden_layer) / np.sqrt(Config.input_dimension)

    b1 = np.zeros((1, number_of_nodes_within_hidden_layer))

    W2 = np.random.randn(number_of_nodes_within_hidden_layer, Config.output_dimension) / np.sqrt(Config.output_dimension)

    b2 = np.zeros((1, number_of_nodes_within_hidden_layer))
    
    model = {}
    
    for i in range(0, passes):
        # Forward Propgation
        exp_scores = forward_propagation(X, W1, b1, W2, b2)
        
        # Back Propgation
        dW1, dW2, db1, db2 = backward_propagation(X, y, W1, b1, W2, b2, exp_scores, number_of_examples);
        
        # Gradient descent parameter update

        ###
        ### YOUR CODE HERE
        ###
        W1 = W1 - dW1 * Config.learning_rate
        b1 = b1 - db1 * Config.learning_rate

        W2 = W2 - dW2 * Config.learning_rate
        b2 = b2 - db2 * Config.learning_rate
    
        
        # Assign new parameters to the model
        
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        result = calculate_loss(model, X, y)
        
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, result))
            
            if result > 2.0:
                print("Loss is too high")
                break
            
    return model

# Step 17
# Train the model
# Note: 10,000 passes is used in the auto graders
model = build_model(X, y, 3, print_loss=True, passes=10000)

# Step 18
# Plot Decision Boundary
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.title("Artificial Neural Network")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

# Step 19
# Visualize
def visualize(X, y, model):
    plot_decision_boundary(lambda x:predict(model, x), X, y)

# Step 20
# Call the visualize function
try:
    visualize(X, y, model)
except:
    print("Can not visualize the graph, the data or the model is inconsistent")