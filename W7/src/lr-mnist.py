import os
import sys
#sys.path.append(os.path.join(".."))
import argparse
# Import teaching utils
import numpy as np
# ERROR when running classifier_utils because it does not exist
# import utils.classifier_utils as clf_util
import pandas as pd
# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class lr_mnist:
    def __init__(self):
         self.mnist = fetch_openml('mnist_784', version=1, return_X_y=False) # This loads the entire set of the nnist handwritten digits data
            
    def data_preparation(self, split_val):
        X = np.array(self.mnist.data)    
        y = np.array(self.mnist.target)

        X = (X - X.min())/(X.max() - X.min())

        # Splitting data up
        X_train, X_test, y_train, y_test = train_test_split(X, # The input features
                                                        y, # The labels
                                                        random_state=9, # Random state - producing same results every time
                                                        test_size=split_val) # I have selected a test / train split of 20 % / 80 %


        return X_train, X_test, y_train, y_test
    
    # Determining the classifier - here I'm selecting the multinomial logistic regression
    def lr_classifier(self, X_train, y_train):
        clf = LogisticRegression(penalty='none', 
                             tol=0.1, 
                             solver='saga',
                             multi_class='multinomial').fit(X_train, y_train)
        return clf

    
    def evalutation(self, clf, X_test, y_test):
        y_pred = clf.predict(X_test)
        report = metrics.classification_report(y_test, y_pred, output_dict = True)
        return report
        #print(pd.DataFrame(report))# The function returns a print of the report
    
    def print_report(self, report):
        #sys.stdout.write(pd.DataFrame(report))
        print(pd.DataFrame(report))

    

def main(): # Now I'm defining the main function 
    # I try to make it possible executing arguments from the terminal
    # Add description
    ap = argparse.ArgumentParser(description = "[INFO] creating logistic regression classifier") # Defining an argument parse
    ap.add_argument("-s","--split_value", 
                    required=False, # As I have provided a value it is not required as I have provided a default split value of 80 % / 20 %
                    type = float, # Int type
                    default = .20, # Setting default to 20 %
                    help = "Test size of dataset")
    args = vars(ap.parse_args()) # Adding them together
                    
                         
    # Assigning the arguments passed in the terminal to a variable I can use in the script
    split_val = args["split_value"] # Defining split value 

    lr_mnist_class = lr_mnist()
    
   # Using my data preparation function to make data in the right format, split data and make a  matrix with dummy variables
    X_train, X_test, y_train, y_test = lr_mnist_class.data_preparation(split_val = split_val)
    
   # Here I'm defining the lr classifier
    lr = lr_mnist_class.lr_classifier(X_train, y_train)
    
    # Evaluation function to print the resulting f1 scores and accuracies
    report = lr_mnist_class.evalutation(lr, X_test, y_test)
    
    # Printing the report specified above
    lr_mnist_class.print_report(report)
    
if __name__ == '__main__':
    main()