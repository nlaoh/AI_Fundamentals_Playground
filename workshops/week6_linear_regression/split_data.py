from random import seed
from random import randrange
from csv import reader

from tabulate import tabulate

# LOAD CSV FILE
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

# PRINT THE FILE's CONTENT
def print_the_dataset(dataset, contents = True, length = True):
    if(contents):
        print(tabulate(dataset))
        
    if(length):
        print(len(dataset))

# SPLIT THE DATA SET INTO TRAINING AND TEST DATA
def train_test_split(dataset, split):
    # Create an empty list for the training set
    training = list()
    
    # Define the size of the training set
    train_size = split * len(dataset)
    
    # Copy the original dataset to 
    test = list(dataset)
    
    #Loops only to the size of the training set
    while len(training) < train_size:
        # Obtain a random index from the dataset/test set
        index = randrange(len(test))
        
        # Populate the training set, by moving the data points from the
        # dataset/test set to the training set
        training.append(test.pop(index))
        
    # Return both the training set and test set    
    return training, test

# SEED THE RANDOM VALUE
seed(1)

# LOAD AND PREPARE DATA
filename = 'small_heart.csv'
dataset = load_csv(filename, skip = True)
print_the_dataset(dataset)
training, test = train_test_split(dataset, 0.8)

print(len(training))

print(len(test))