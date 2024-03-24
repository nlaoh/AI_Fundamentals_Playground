# Step 1
# Import the needed libraries.
from math import sqrt
from matplotlib import pyplot as plot
from random import seed
from random import randrange
from csv import reader

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

# Step 3
# Convert string column to float
def string_column_to_float(dataset, column):
 
    for row in dataset:
        # The strip() function remove white space
        # then convert the data into a decimal number (float)
        # and overwrite the original data
        
        ###
        ### YOUR CODE HERE
        ###
        row[column] = float(row[column].strip())

# Step 4
# Calculate the mean value of a list of numbers
def mean(values):
 
    # Sum all the values and then divide number of values
    ###
    ### YOUR CODE HERE
    ###
    mean_results = sum(values) / float(len(values))
    return mean_results

# Step 5
# Calculate least squares between x and y
def leastSquares(dataset):

    x = list()
    y = list()
    
    for row in dataset:
        x.append(row[0])
        
    for row in dataset:
        y.append(row[1])

    b0 = 0 # w0, y-intercept
    b1 = 0 # w1, gradient
    
    ###
    ### YOUR CODE HERE
    ###
    mean_x = mean(x)
    mean_y = mean(y)
    variance = 0.0
    covariance = 0.0
    
    #Calculate variance
    for i in range(len(x)):
        variance = variance + (x[i] - mean_x) ** 2

    # Calculate covariance   
    for i in range(len(x)):
        covariance = covariance + ((x[i] - mean_x) * (y[i] - mean_y))
        
    # Estimate coefficients
    b1 = covariance / variance
    b0 = mean_y - b1 * mean_x 
    # print("variance is {}".format(variance))
    # print("covariance is {}".format(covariance))
    # print("b0 is {}".format(b0))
    # print("b1 is {}".format(b1))
    return [b0, b1]

# Step 6
# Calculate root mean squared error
def root_mean_square_error(actual, predicted): 
    sum_error = 0.0
    rmse = 0.0
    ###
    ### YOUR CODE HERE
    ###
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error = sum_error + (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    rmse = sqrt(mean_error)
    ###
    ### YOUR CODE HERE
    ###
    return rmse

# Step 7
# Make Predictions
def simple_linear_regression(train, test):

    predictions = list()
    b0, b1 = leastSquares(train)
    
    # Calculate the prediction (yhat)
    ###
    ### YOUR CODE HERE
    ###
    for row in test:
        yhat = b0 + b1 * row[0]
        predictions.append(yhat)
    
    return predictions


# Step 8
# Split the data into training and test sets
def train_test_split(dataset, split):
    train = list()
    train_size = split * len(dataset)
    test = list(dataset)
    
    while len(train) < train_size:
        index = randrange(len(test))
        train.append(test.pop(index))
        
    return train, test

# Step 9
# Evaluate regression algorithm on training dataset
def evaluate_simple_linear_regression(dataset, split=0):

    test_set = list()
    train, test = train_test_split(dataset, split)
    
    for row in dataset:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    
    ###
    ### YOUR CODE HERE
    ###
    if(split == 0):
        predicted = simple_linear_regression(dataset, test_set)
    else:
        predicted = simple_linear_regression(train, test_set)

    if(split == 0):
        actual = [row[-1] for row in dataset]
    else:
        actual = [row[-1] for row in test]
    
    rmse = root_mean_square_error(actual, predicted)
    
    return rmse

# Step 10
# Visualise the dataset
def visualise_dataset(dataset):
    test_set = list()
    
    for row in dataset:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    
    sizes, prices = [], []
    for i in range(len(dataset)):
        sizes.append(dataset[i][0])
        prices.append(dataset[i][1])
        
    plot.figure()
    plot.plot(sizes, prices, 'x')
    plot.plot(test_set, simple_linear_regression(dataset, test_set))
    plot.xlabel('Size')
    plot.ylabel('Price')
    plot.grid()
    plot.tight_layout()
    plot.show()

# Step 11
# Seed the random value
seed(1)

# Step 12
# Load and prepare data
filename = 'fertility_rate-worker_percent.csv'
dataset = load_csv(filename, skip=True)

for i in range(len(dataset[0])):
    string_column_to_float(dataset, i)

# Step 13
# Evaluate algorithm
split = 0.6
rmse = evaluate_simple_linear_regression(dataset, split)

print('Root Mean Square Error: %.3f' % rmse)
visualise_dataset(dataset)