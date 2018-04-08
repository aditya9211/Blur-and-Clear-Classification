"""
Files which contains
helper function which
pre-process or common functions
in train.py and test.py

"""
# -*- coding: utf-8 -*-
import numpy as np
import scipy.misc as ms
import os

def path_validation(PATH, read_access=False):
    """
    Validate the given path, that if it exists(file)
    if "write_access" file path is passed it creates new dir
    if not present.
    and if "read_access" file path is passed it validates
    if not present returns false

    @ Parameters:
    -------------
    PATH: str
        Contains the path of the
        validation check file.
    read_access: bool
        If True, path specified is going to
        read so it validates the path.
        else, path specified is going to
        write some data into that PATH.

    @ Returns:
    ----------
    : bool
        True, means path file exists or
        it is in write_access mode.
        whereas False, means path doesn't exists
        and it was in read_access mode.

    """
    # Path Validation
    if read_access:
        if not os.path.exists(PATH):
            print('File Doesn\'t Exists ' + PATH + '....\n')
            print('Terminating Processs ......\n')
            return False
    else:
        if not os.path.exists(PATH):
            if not os.path.exists(os.path.dirname(PATH)):
                print('Path Doesn\'t Exists ' + PATH + '....\n')
                print('Creating Dir  ' + os.path.dirname(PATH) + '....\n')
                os.makedirs(os.path.dirname(PATH))

    return True

def resize(imgs, width=100, height=100):
    """
    Resize the given grayscale image
    with orig_reso dimension to
    w x h dimensions and then
    scale the image to be in [0.1]

    @ Parameters:
    -------------
    imgs: np.array
        Grayscale images comprising of
        both clear and blur images
    width: int
        Width of the returned resized images
    height: int
        Height of the returned resized images

    @ Returns:
    ----------
    resized_img: np.array
        Resized w x h dimension grayscale images

    """
    print('\nResizing the images .........\n')
    orig_reso = (imgs.shape[1], imgs.shape[2])
    resized_img = []
    for i in range(imgs.shape[0]):
        resized_img.append(ms.imresize(imgs[i,...].reshape(orig_reso[0],orig_reso[1]),
                            (width,height),interp='cubic').flatten())

    # Converting list of images to np.array 
    resized_img = np.asarray(resized_img)

    # Scaling the images values to be in [0,1]
    resized_img = resized_img/255.

    return resized_img

def sigmoid(X): 
    """
    Sigmoid activation Function
    Calcultes the sigmoid of the passed
    values

    @ Parameters:
    -------------
    X: np.array
        array whose sigmoid needs to
        be calculated        

    @ Returns:
    ----------
    x_sig: np.array
        sigmoid calculated values 
    
    """

    x_sig = 1.0 / (1.0 + np.exp(-X))

    return x_sig


def h(theta,X,func='sig'):
    """
    Used to calculate the value of
    different Activaion Function
    namely 'sigmoid', 'tanh', 
    'relu', 'softmax' using
    raw X value and connecting theta values

    @ Parameters:
    -------------
    theta: np.array
        Contains the theta value 
        corresponding to adjacent layers
    X: np.array
        Value of layer before
        applying activation to it

    @ Returns:
    ----------
    activation values 

    """

    # Value of layer next to X
    a = theta.dot(X.T)

    # tanh activation function
    if(func== 'tanh'):
        return np.tanh(a)
    # Identity activation function
    if func == 'none':
        return a
    # Softplus activation function
    if func == 'softplus':
        return np.log(1 + np.exp(a))
    # Leaky-ReLu activation function
    if func == 'relu':
        return np.maximum(0.01*a, a)
    # Softmax activation function
    if func == 'softmax':
        a1 = np.exp(a)
        a1 = a1 / np.sum(a1, axis = 0, keepdims = True)
        return a1
    
    # Sigmoid activation function
    return sigmoid(a)


def validate(params, img, act = 'sig'):
    """
    Outputs the index of classes,labels 
    corresponds to trained NN Model weigts.
    Doing forward propagation using input images.

    @ Parameters:
    -------------
    params: dict
        Conatins the trained theta weights
        used to calculate output of each layer 
        by mutilpying it to each layer
        act(input x theta1)
        act(hidden x theta2) = output
    img: np.array
        Contains the images whose labels need
        to be predicted
    act: str
        Activation function which is applied to 
        the neurons in forward propagation

    @ Returns:
    ----------
    predicted_class: np.array
        Gives the index of label which has 
        max output probability in the array
        of predicted final output layer  

    """

    # Forward Propagation
    hidden_activation = h(params['Theta1'],img,act)
    hidden_activation = np.insert(hidden_activation, 0, 1, axis=0)
    output_activation = h(params['Theta2'],hidden_activation.T,'softmax')

    # Maximum of prob. is the predicted output class
    predicted_class = np.argmax(output_activation,axis=0) 

    return predicted_class


def model_score(params, images, labels, act='sig'):
    """
    Calculates the Accuracy of the Model on given 
    images and actual labels of the dataset

    @ Parameters:
    -------------
    params: dict
        Conatins the trained theta weights
        used to calculate output of each layer 
        by mutilpying it to each layer
        act(input x theta1)
        act(hidden x theta2) = output
    images: np.array
        Contains the images whose labels need
        to be predicted
    labels: np.array
        Contains the labels(1/0)
        corresponding to selected images
    act: str
        Activation function which is applied to 
        the neurons in forward propagation

    @ Returns:
    ----------
    accuracy: float
        Accuracy of the model when provided with
        images and its actual labels

    """
    
    # Calculating the predicted labels
    pred_labels = validate(params, images, act)
    labels = labels.flatten()
    accuracy = np.mean(pred_labels == labels)*100.0

    return accuracy
