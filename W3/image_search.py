# Importing modules 
import os # Path organization
import sys # For including utils
sys.path.append(os.path.join("..")) # For including utils
from pathlib import Path
import cv2 # Handling images
import numpy as np # Numerical operations
from utils.imutils import jimshow # Showing image
from utils.imutils import jimshow_channel # Showing channels
import matplotlib.pyplot as plt # Plotting
import pandas as pd
import glob


# defining functino that both normalizes and makes hist 
def calc_norm_hist(image, bins = 8, end_val = 256):
    # using cv2 to create the histogram before it is normalized
    hist = cv2.calcHist([image], [0,1,2], None, [bins, bins, bins], [0, end_val, 0, end_val, 0, end_val]) 
    # using cv2 again, defining the mininmum and the maximum, and method (MINMAX)
    hist_norm = cv2.normalize(hist, hist, 0,255, cv2.NORM_MINMAX) 
    return hist_norm # returning the histogram

def calc_target_comp(path = "flower_data", target_file = "image_1164.jpg"):
    # reading file by joining the path folder name and the specific target filename
    target = cv2.imread(os.path.join(path, target_file)) 
    # the function defined above is applied
    target_hist = calc_norm_hist(target)
    # filepath is defined 
    filepath = os.path.join(path)
    # empty list is created where distance and filename can be appended
    chi_sqs = []

    for filename in glob.glob(filepath+"/*.jpg"): # takes every file that exists with *.jpg extension and performs the following action on them:
        if filename == filepath + f"/{target_file}": # if the filename matches the one of our target file then "pass" or "skip" this round
            pass
        else:
            image = cv2.imread(str(filename)) # read the image
            hist = calc_norm_hist(image) # generate a histogram
            chi = round(cv2.compareHist(hist, target_hist, cv2.HISTCMP_CHISQR),2) # calculating distance metric - here chi-square 
            chi_sqs.append((os.path.basename(filename), chi)) # appending information to our list 

    df = pd.DataFrame(chi_sqs, columns =["filename", "distance"]) # using pandas, i'm making a dataframe
    df.sort_values("distance", inplace=True) # arranging the df in ascending order so that the lowest value comes first
    print(f"file with lowest distance to target file is {print(df.filename.loc[0])}") # using .loc we can index the df and get the zero row and print this using formatted strings

    outpath = os.path.join("img_distance_info.csv") # defining place to save csv
    df.to_csv(outpath)
    
def main():
    calc_target_comp()

# Define behaviour when called from command line
if __name__=="__main__":
    main()
