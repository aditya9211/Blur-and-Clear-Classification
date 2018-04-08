"""
Script for predicting label as
good or bad given the image
It preprocess the image same as it 
was pre-processed in training 

"""
import argparse
from sklearn.externals import joblib
import numpy as np
import scipy.misc as ms
import scipy.ndimage as nd
from utils import (resize, validate, path_validation)
from config import (MODEL_PATH, WIDTH, HEIGHT,
                    RADIUS)

def predict_preprocess(IMAGE_PATH):
    """
    Preprocess the image given path of the image/images
    to apply median filter and resizing the image
    same as done in training the network

    @ Parameters:
    -------------
    IMAGE_PATH: str
        Path of the images

    @ Returns:
    ----------
    img: np.array
        filtered and pre-processed combined
        images arrays 
    
    """
    # Reading images in grayscale mode
    img = ms.imread(IMAGE_PATH, mode='L')
    # APllying median filter to remove noise
    img = nd.median_filter(img, RADIUS)
    # To make it 2D
    img = img[np.newaxis, :]
    # Resizing the images to that of train
    img = resize(img, width=WIDTH, height=HEIGHT)
    # Addition of bias term
    img = np.insert(img, 0, 1, axis=1)

    return img


def main():
    """
    Parse the argument and check validaton
    of passed image and trained model path
    Predict the label of images passed after 
    pre-process the images same as done in 
    training part
    
    """
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-img", "--image_path", required=True, 
                        help="path to image")
    args = vars(ap.parse_args())

    IMAGE_PATH = args["image_path"]

    # Path Validation of image and Model
    if not path_validation(IMAGE_PATH, read_access=True):
        exit(0)
    if not path_validation(MODEL_PATH, read_access=True):
        exit(0)

    # Preprocessed the images
    img = predict_preprocess(IMAGE_PATH)

    # Load the trained NN model
    params = joblib.load(MODEL_PATH)

    # Find the label predicted by the model
    predicted_label = validate(params, img)

    for label in predicted_label:
        if label:
            print("Good Image\n")
        else:
            print("Bad Image\n")


if __name__ == "__main__":
    main()


