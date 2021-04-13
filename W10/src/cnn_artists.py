# data tools
import os
import sys
import pandas as pd
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)

from tensorflow.keras.utils import plot_model 
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


class cnn_artists():
    """This is a class for performing a Convolutional Neural Network classification on a Kaggle dataset with impressionist painters (data can be found here https://www.kaggle.com/delayedkarma/impressionist-classifier-data)
    """
    def __init__(self, args):
        self.args = args
        
       # Save folders   
        self.trainfolder = self.args.folders[0] # 0 element is the trainfolder
        self.valfolder = self.args.folders[1] # 1 element is the validation folder
    
    def data_preparation(self, folder, save_labels = None):
        # Empty list for labels and for images
        if save_labels == True:
            labelNames = []
        X = []
        y = []
        # Definition of label binarizer
        lb = LabelBinarizer()

        # Train data
        for subfolder in Path(folder).glob("*"):
            artist = os.path.basename(subfolder) # Only keeping last part og path
            if save_labels == True: # If true it save one instance of each artist
                labelNames.append(artist) # Appends to list above
            # Take the current subfolder
            for pic in Path(subfolder).glob("*.jpg"): # Taking all elements in the folder
                pic_array = cv2.imread(str(pic)) # Reading image
                compressed = cv2.resize(pic_array, (self.args.resize,self.args.resize), interpolation = cv2.INTER_AREA) # Resizing image
                X.append(compressed)
                y.append(artist)
        
        if save_labels == True:
            self.labelNames=labelNames
        # Making it arrays 
        X = np.array(X) 
        y = np.array(y)

        # Normalization
        X = X.astype("float") / 255.

        # Label binarization
        # integers to one-hot vectors - one-hot encoding
        y = lb.fit_transform(y)

        return X, y


    def model_preparation(self, trainX, trainY, testX, testY):
        # Define model
        # initialise model
        self.model = Sequential()

        # define CONV => RELU layer
        self.model.add(Conv2D(32, (3, 3), # 32 = neurons, (3,3) = kernel size
                         padding="same", # Adding a layer of 0 like in slides
                         input_shape=(self.args.resize, self.args.resize, 3)))

        self.model.add(Activation("relu"))

        # softmax classifier
        self.model.add(Flatten())
        self.model.add(Dense(10))
        self.model.add(Activation("softmax"))

            # Compile model
        opt = SGD(lr =.01) # Learning rate 0.001 --> 0.01 are the usual values
        self.model.compile(loss="categorical_crossentropy", # Loss function also used in backpropagation networks
                      optimizer=opt, # Specifying opt
                      metrics=["accuracy"])
        
        self.H = self.model.fit(trainX, trainY, 
              validation_data=(testX, testY), 
              batch_size=32,
              epochs=self.args.epochs,
              verbose=1)
        return self.model.summary()

    def model_evaluation(self, testX, testY, batches = 32):
        predictions = self.model.predict(testX, batch_size = batches)
        # Comparing predictions to our test labels
        report = pd.DataFrame(classification_report(testY.argmax(axis=1), # y true
                                predictions.argmax(axis=1), # y pred
                                target_names=self.labelNames, # labels
                                       output_dict=True)) #
        
        report.to_csv("../output/classification_report.csv")
        print("report saved in output as classification_report.csv")
    def plot_history(self): # Plot of a model as it learns
        # visualize performance
        plt.style.use("fivethirtyeight")
        plt.figure()
        plt.plot(np.arange(0, self.args.epochs), self.H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.args.epochs), self.H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.args.epochs), self.H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, self.args.epochs), self.H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig("../output/loss_accuracy_curve.png")
        plt.show()
        print("figure of learning curves are saved as ../output/loss_accuracy_curve.png")
    
def main(): 
    ap = argparse.ArgumentParser(description="[INFO] class made to run CNN on Kaggle data set with impressionist artists") 
    
    # Adding a folder argument - I don't know how I can split without actually pasting the folder names 
    ap.add_argument('-f',
                    '--folders',
                    help='str, folders for train and validation data',
                    nargs=2) # Specifies the number of elements there can be in the argument
    # Adding an argument for epochs
    ap.add_argument('-resize',
                    '--resize',
                    help="int resize value of paintings",
                    default=200) # Default is 25
    ap.add_argument('-e',
                    '--epochs',
                    default=25) # Default is 25
 
    args = ap.parse_args('--folders ../data/training ../data/validation'.split()) # I know that it is wrong to paste the actual paths here, but I cannot find a way to make 2 in 1 arguments without doing it. Hope I'll find out :) 
    
    # cnn_artists is imported
    cnn_artists_class = cnn_artists(args)
    
    # Train and test data is defined 
    # Train
    trainX, trainY = cnn_artists_class.data_preparation(cnn_artists_class.trainfolder, 
                                   save_labels = True)
    # Test
    testX, testY = cnn_artists_class.data_preparation(cnn_artists_class.valfolder, 
                                   save_labels = False)
   # Model is defined and compiled
    cnn_artists_class.model_preparation(trainX, trainY, testX, testY)
    
    # Model is evaluated
    cnn_artists_class.model_evaluation(testX, testY)
    # Learning and accuracy curves are saved in a plot
    cnn_artists_class.plot_history()
    
if __name__=="__main__":
    main() 