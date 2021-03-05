''' Assignment description
The purpose of this assignment is to use computer vision to extract specific features from images. In particular, we're going to see if we can find text. We are not interested in finding whole words right now; we'll look at how to find whole words in a coming class. For now, we only want to find language-like objects, such as letters and punctuation.
'''

# Packages 
import os # Path operations
import sys # Path operations
sys.path.append(os.path.join("..", "..", ".."))  # We need to include the home directory in our path, so we can read in our own module.
import cv2 # Image operations
import numpy as np # Numeric operatoins
from utils.imutils import jimshow # Showing image
from utils.imutils import jimshow_channel # Showing image channels
from pathlib import Path
import argparse # To make arguments from the terminal


# Trying with init and argument parser

class canny_calculations:       

    def __init__(self, data_folder, image_name): # Always starting with "self"
        self.data_folder = data_folder # Assigning data folder to a self
        self.image_name = image_name
    
    def read_img(self):
        root_dir = Path.cwd() # Path pastes the parent folder
        fname = os.path.join(root_dir, self.data_folder, self.image_name) # Joining both root_dir, the data folder (defined above) and the image name also defined above
        image = cv2.imread(fname) # reading the image 
        return image
    
         # We'll need to save the images a couple of times - so I have made a fucntion for this as well
    def write_img(self, image, filepath, filename): 
        outfile = os.path.join(filepath, filename) # Joining filepath and filename into one path
        cv2.imwrite(outfile, image) # Writing the file  
    
    def canny_contour_calc(self, image):
        start_x = int(image.shape[1]/3) # It seems like the letter is in the center of the image so I'm taking x and y coordinates 1/3 - 2/3 of the picture
        end_x = int(2*(image.shape[1]/3)) # Reversed x and y - therefore I'm taking [1]

        start_y = int(image.shape[0]/3) # Reversed x and y - therefore I'm taking [0]
        end_y = int(2*(image.shape[0]/3))

        # Now I'll see whether these coordinates gets an amount of text that I can analyze
        image_ROI = cv2.rectangle(image.copy(), # Copy to avoid deconstructing the original image
                      (start_x, start_y), # Start on x and y axis
                      (end_x, end_y), # End on x and y
                      (0, 255, 0), # Color - in this case green
                      3) # Thickness of 3

        # Cropping the picture 
        cropped = image[start_y:end_y, start_x:end_x]


      # Now I'll see how we can extract edges with canny and find the letters. I'm starting by turning the picture into greyscale
        grey_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey_image, (5,5), 0) # image, 5x5 kernel, 0 = amount of variation from the mean you can take into account 

        # Defining the canny edge detection
        canny = cv2.Canny(blurred, 115, 150) # Min and max threshold - I tried to find them manually and found these thresholds to be best. 
        # Defining the contours in the image
        (contours,_) = cv2.findContours(canny.copy(), # using np function to make a copy rather than destroying the image itself
                         cv2.RETR_EXTERNAL,  # contour retrieval mode,
                         cv2.CHAIN_APPROX_SIMPLE) # contour approximation

        cropped_contours = cv2.drawContours(cropped.copy(), # image, contours, fill, color, thickness
                                 contours,
                                 -1, # whihch contours to draw. -1 will draw contour for every contour that it finds
                                 (0,255,0), # contour color
                                 2)
        return image_ROI, cropped, cropped_contours # Returning both the image with region of interest, the cropped picture and the image with canny edge detection 


    
def main(): # Now I'm defining the main function where I try to make it possible executing arguments from the terminal
    # add description
    ap = argparse.ArgumentParser(description = "[INFO] creating canny edge detection") # Defining an argument parse
    # add argument
    ap.add_argument("-d", "--data_folder",  # Argument 1
                    required=False, 
                    type = str,
                    default = "letter_data", 
                    help = "str of data_folder") 
    
    ap.add_argument("-i",  # Argument 2
                    "--image_name", 
                    required=False, 
                    type = str,
                    default = "We_Hold_These_Truths_at_Jefferson_Memorial_IMG_4729.JPG",
                    help = "str of image filename") 

    args = vars(ap.parse_args()) # Adding them together
   
    image_operator = canny_calculations(data_folder = args["data_folder"], image_name = args["image_name"]) # Defining what they corresponds to in the canny class and functions

    image_original = image_operator.read_img()
    
    (image_ROI, cropped, cropped_contours) = image_operator.canny_contour_calc(image_original)
   
    # Saving all images
    image_operator.write_img(image = image_ROI, 
              filepath = ".", 
              filename = "image_with_ROI.jpg") # Saving ROI image
    
    image_operator.write_img(image = cropped, 
              filepath = ".", 
              filename = "image_cropped.jpg") # Saving cropped image
    
    image_operator.write_img(image = cropped_contours, 
              filepath = ".", 
              filename = "image_letters.jpg") # saving cropped image with contours around letters
    
   
    
if __name__ == "__main__":
    main()
    
    
    
    