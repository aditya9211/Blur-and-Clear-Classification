"""
Script to test the NN Model
on test data which splitted
in train.py

"""
# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
from sklearn.externals import joblib
from utils import (model_score, path_validation)
from config import (MODEL_PATH, TEST_DATA_PATH,
                    TEST_LABEL_PATH)

def get_data():
    """
    Validating the Path and Extracting the Test Set
    and trained theta parameters

    @ Returns:
    ----------
    params: dict
        Conatins the trained theta weights
        used to calculate output of each layer 
        by mutilpying it to each layer
        act(input x theta1)
        act(hidden x theta2) = output
    test_images: np.array
        Contains the test images
    test_labels: np.array
        Contains the labels(1/0)
        corresponding to selected images

    """
    if not path_validation(MODEL_PATH, read_access=True):
        exit(0) 
    if not path_validation(TEST_DATA_PATH, read_access=True):
        exit(0) 
    if not path_validation(TEST_LABEL_PATH, read_access=True):
        exit(0) 

    params = joblib.load(MODEL_PATH)
    test_images = np.load(TEST_DATA_PATH)
    test_labels = np.load(TEST_LABEL_PATH)

    # Addition of bias in test set
    test_images = np.insert(test_images, 0, 1, axis=1)

    return params, test_images, test_labels


def main():
    """
    Function to load the data
    and calculate the test set
    accuracy 

    """
    # LOading the Test images & labels
    params, test_images, test_labels = get_data()

    # Accuracy on Test Data
    accuracy = model_score(params, test_images, test_labels, act='sig')
    print ('\nAccuracy : ' + str(accuracy) + ' %\n')


if __name__ == "__main__":
    main()
