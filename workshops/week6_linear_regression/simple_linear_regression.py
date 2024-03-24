from math import sqrt
from matplotlib import pyplot as plot
from random import seed
from random import randrange
from csv import reader

def load_csv(filename):
    dataset = list()
    # Opens the file in read only mode
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        
        for row in csv_reader:
            dataset.append(row)
            
    return dataset

def string_column_to_float(dataset, column):
    for row in dataset:
        # The strip() function remove white space
        # then convert the data into a decimal number (float)
        # and overwrite the original data
        row[column] = float(row[column].strip())

def mean(values):
    # Sum all the values and then divide number of values
    mean_results = sum(values) / float(len(values))
    return mean_results

def variance(values, mean):
    # Create the differences list
    differences = list()
   
    # Loop through the values, take the difference and square
    for x in values:
        differences.append((x - mean) ** 2)
    
    # Sum all the values and then divide number of values
    variance_results = sum(differences)
    
    return variance_results

# Calculate mean and variance of x and y
dataset = [[0,1], [1,3], [2,2], [3,3], [4,5]]

x = list()
y = list()

for row in dataset:
    x.append(row[0])
    
for row in dataset:
    y.append(row[1])
    
mean_x = mean(x)
mean_y = mean(y)

variance_x = variance(x, mean_x)
variance_y = variance(y, mean_y)

print('x stats: mean=%.3f variance=%.3f' % (mean_x, variance_x))
print('y stats: mean=%.3f variance=%.3f' % (mean_y, variance_y))

def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    
    # Calculate the relationship between the groups of data     
    for i in range(len(x)):
        covar = covar + ((x[i] - mean_x) * (y[i] - mean_y))
        
    return covar

# Calculate covariance
dataset = [[1,1], [2,3], [3,2], [4,3], [5,5]]

x = list()
y = list()

for row in dataset:
    x.append(row[0])
    
for row in dataset:
    y.append(row[1])

mean_x = mean(x)
mean_y = mean(y)

covariance_variable = covariance(x, mean_x,y, mean_y)

print('Covariance=%.3f' % (covariance_variable))

def coefficients(dataset):
    x = list()
    y = list()
    
    for row in dataset:
        x.append(row[0])
        
    for row in dataset:
        y.append(row[1])
    
    x_mean = mean(x)
    y_mean = mean(y)
    
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    
    b0 = y_mean - b1 * x_mean
    
    return [b0, b1]

b0, b1 = coefficients(dataset)

print('Coefficients: b0=%.3f, b1=%.3f' % (b0, b1))

def root_mean_square_error(actual, predicted):
    sum_error = 0.0
    
    # Loops through the difference between the prediction
    # and the actual output
    # Then update the sum error
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error = sum_error + (prediction_error ** 2)
    
    # Take the average
    mean_error = sum_error / float(len(actual))
    
    return sqrt(mean_error)

def simple_linear_regression(train, test):
    predictions = list()
    b0, b1 = coefficients(train)
    
    # Calculate the prediction (yhat)
    for row in test:
        yhat = b0 + b1 * row[0]
        predictions.append(yhat)
        
    return predictions

def train_test_split(dataset, split):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    
    return train, dataset_copy

def evaluate_simple_linear_regression(dataset, split=0):
    test_set = list()
    train, test = train_test_split(dataset, split)
    
    for row in dataset:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
        
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

rmse = evaluate_simple_linear_regression(dataset)

print('Root Mean Square Error: %.3f' % rmse)
visualise_dataset(dataset)

seed(1)

filename = 'insurance.csv'
dataset = load_csv(filename)

for i in range(len(dataset[0])):
    string_column_to_float(dataset, i)

split = 0.6
rmse = evaluate_simple_linear_regression(dataset, split)

print('Root Mean Square Error: %.3f' % rmse)
visualise_dataset(dataset)