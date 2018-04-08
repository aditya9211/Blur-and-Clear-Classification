"""
This file contains constants required in the training
and testing stage of data processing also for training
Neural Networks Model

"""
# -*- coding: utf-8 -*-
import os

# Stores current exceuting path
HOME_FOLDER_PATH = os.getcwd()

# if not passes default model path
MODEL_PATH = HOME_FOLDER_PATH + '/model/result.pkl'
DATA_PATH = HOME_FOLDER_PATH + '/data'

# Path where splitted data get stored
TRAIN_DATA_PATH = DATA_PATH + '/train_images.npy'
TRAIN_LABEL_PATH = DATA_PATH + '/train_labels.npy'
TEST_DATA_PATH = DATA_PATH + '/test_images.npy'
TEST_LABEL_PATH = DATA_PATH + '/test_labels.npy'

# Path for saving plot of cost vs iterations
PLOT_PATH = DATA_PATH + '/loss_decay.png'


# Median Filter size
RADIUS = 3

# width and height of resized image
WIDTH = 100
HEIGHT = 100

# Splitting ratio for training & testing
SPLIT_RATIO = 0.2

# Seed to get same results on re-run
SEED = 10

# Size of data to be fed at each epochs
BATCH_SIZE = 10

# Hidden Layer neurons size
NEURONS_SIZE = 300

# Iteration in NN Model
MAX_ITER = 50

# Logging steps to show summary
LOGGING_STEPS = min(1, MAX_ITER)

# Learning rate in NN Model
ALPHA = 0.001

# Regularization Term used in cost
LAMBDA = 0.0007

# Default Activation Function
ACT = 'sig'