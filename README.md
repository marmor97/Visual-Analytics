# Visual-Analytics

## Week 5
To use the files in this repository, please clone this repo to the place you work with the following command:

``git clone https://github.com/marmor97/Visual-Analytics``

``cd Visual-Analytics/W5``

``bash ./create_venv_w5.sh``

To execute the script calculating contours of the image, open your terminal and type:

``python3 edge_detection_v2.py``

If you want to look at the notebook, you can find it under the filename ``edge_detection.ipynb``

## Week 7
To use the scripts in this repository for week 7, please clone this repo to the place you work with the following command:

``git clone https://github.com/marmor97/Visual-Analytics``

``cd Visual-Analytics/W7``

``bash ./create_venv_w7.sh``

After creating the virtual environment, please activate it by writing:

``source marmor_visw7/bin/activate``

When you have activated the environment you can run the two scripts. To run the logistic regression, please type:

``python3 src/lr-mnist.py``

And to run the neural network:

``python3 src/nn-mnist.py``

You can modify the train test split in both by adding the argument -split (e.g. --split 0.5 to get a 50%/50% split) and you can change the amount of hidden layers and output layers with --output and --layer.

You should be able to see the entire classification report in your terminal.

## Week 10
This script classifies paintings impressionist artists with a convolutional neural network (CNN). This includes preprocessing where paintings are compressed into sizes of 200 and normalization of values. To use the scripts in this repository for week 10, please clone this repo to the place you work with the following command:

``git clone https://github.com/marmor97/Visual-Analytics/W10``

``bash ./create_venv_w10.sh``

After creating the virtual environment, please activate it by writing:

``source marmor_visw10/bin/activate``

When you have activated the environment you can run the CNN classifier script. To do so, please type:
``cd src``
``python3 cnn_artists.py``

If not you wish to run the CNN (takes 10-15 min), you can find the accuracy/loss curves and classification report in the folder ``output``. 
