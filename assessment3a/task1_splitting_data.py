#!/usr/bin/env python3
# Step 1
# Import the libraries.
from random import seed
from random import randrange
from csv import reader

from tabulate import tabulate

# Step 2
# Load the big heart CSV file
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
# Print the file's content
# Note: This function can be called for other parts in the assignment
def print_the_dataset(dataset, contents = True, length = True):
    if(contents):
        print(tabulate(dataset))
        
    if(length):
        print(len(dataset))

# Step 4
# Split the big heart dataset into training and test data
def train_test_split(dataset, split):
    # Create an empty list for the training set
    ###
    ### YOUR CODE HERE
    ###
    training_set = list()

    # Define the size of the training set
    ###
    ### YOUR CODE HERE
    ###
    training_size = split * len(dataset)
    
    # Copy the original dataset to 
    ###
    ### YOUR CODE HERE
    ###
    test_set = list(dataset) # Will pop by split amount in next step
    
    #Loops only to the size of the training set
    ###
    ### YOUR CODE HERE
    ###
    while len(training_set) < training_size: 
        # Populate the training set, by moving the data points from the
        # dataset/test set to the training set
        index = randrange(len(test_set)) # Random index from test set
        ###
        ### YOUR CODE HERE
        ###
        training_set.append(test_set.pop(index)) # Pop random index from test set into training set
        
    # Return both the training set and test set 
    ###
    ### YOUR CODE HERE
    ###
    return training_set, test_set

# Step 5
# Seed the random value
seed(1)

# Step 6
# Load and prepare data
filename = 'big_heart.csv'

dataset = load_csv(filename, skip = True)
print_the_dataset(dataset)
training, test = train_test_split(dataset, 0.8)

print(len(training))

print(len(test))