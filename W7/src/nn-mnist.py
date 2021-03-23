# path tools
import sys,os
sys.path.append(os.path.join(".."))
# neural network with numpy
from W7.utils.neuralnetwork import NeuralNetwork

import argparse
import numpy as np

# Pandas to print the classification report 
import pandas as pd

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import datasets


class nn_mnist:
    def __init__(self):
         self.mnist = fetch_openml('mnist_784', version=1, return_X_y=False) # This loads the entire set of the nnist handwritten digits data

    def data_preparation(self, split_val):
        X = np.array(self.mnist.data.astype("float"))    
        y = np.array(self.mnist.target)
        X = (X - X.min())/(X.max() - X.min())
        
        X_train, X_test, y_train, y_test = train_test_split(X, # The input features
                                                        y, # The labels
                                                        random_state=9, # Random state - producing same results every time
                                                        test_size=split_val) # I have selected a test / train split of 20 % / 80 %

        
        # Convert label to binary with LabelBinarizer()
        # Now they only contain 1 and 0's in a matrix
        y_train = LabelBinarizer().fit_transform(y_train)  
        y_test = LabelBinarizer().fit_transform(y_test)

        return X_train, X_test, y_train, y_test

    def nn_classifier(self, X_train, y_train, layer_val, output_val):
        nn = NeuralNetwork([X_train.shape[1], layer_val, output_val])
        print("[INFO] {}".format(nn))
        nn.fit(X_train, y_train, epochs=100) 
        return nn

    def evalutation(self, nn, X_test, y_test):
        y_pred = nn.predict(X_test)
        report = metrics.classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), output_dict = True) # Using .argmax(axis=1) to output correct labels in the right format
        return report # The function returns a print of the report
    
    def print_report(self, report):
        #sys.stdout.write(pd.DataFrame(report))
        print(pd.DataFrame(report))


    

def main(): # Now I'm defining the main function 
    
    # I try to make it possible executing arguments from the terminal
    # Add description
    ap = argparse.ArgumentParser(description = "[INFO] creating neural network classifier") # Defining an argument parse
    ap.add_argument("-s","--split_value", 
                    required=False, # As I have provided a value it is not required as I have provided a default split value of 80 % / 20 %
                    type = float, # Int type
                    default = .20, # Setting default to 20 %
                    help = "Test size of dataset")

    ap.add_argument("-l","--layer_value", 
                required=False, # As I have provided a value it is not required as I have provided a default 
                type = int, # Int type
                default = 15, # Setting default to 15
                help = "Value in hidden layer")

    ap.add_argument("-o","--output_value", 
                required=False, # As I have provided a value it is not required as I have provided a default 
                type = int, # Int type
                default = 10, # Setting default to 10
                help = "Value of output layer")
    args = vars(ap.parse_args()) # Adding them together    
    
    
    # Assigning the arguments passed in the terminal to a variable I can use in the script
    split_val = args["split_value"] # Defining split value 
    layer_val = args["layer_value"] # Defining layer value
    output_val = args["output_value"] # Defining layer value

    nn_mnist_class = nn_mnist()
    
   # Using my data preparation function to make data in the right format, split data and make a  matrix with dummy variables
    X_train, X_test, y_train, y_test = nn_mnist_class.data_preparation(split_val = split_val)
    
   # Here I'm defining the nn classifier. 20 epochs. 
    nn = nn_mnist_class.nn_classifier(X_train, y_train, layer_val = layer_val, output_val = output_val)
    
    # Evaluation function to print the resulting f1 scores and accuracies
    report = nn_mnist_class.evalutation(nn, X_test, y_test)
    
    nn_mnist_class.print_report(report)
       
if __name__ == '__main__':
    main()
