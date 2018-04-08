"""
Script for pre-processing the data by
resizing, median filtering the images.
And finally training the Neural Network model
for the task of classifying blur and clear images.

"""
# Loading required Libraries
from __future__ import print_function
from config import *
from utils import (h, sigmoid, validate, resize,
                    model_score, path_validation)
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import scipy.misc as ms
import scipy.ndimage as nd
import argparse

def data_preprocess(GOOD_IMG_PATH, BAD_IMG_PATH, radius=3):
    """
    Extracts the images from the given paths
    then pre-process them by applying
    median filter to filter out the 
    noise present in the images and 
    finally concatenate the good and bad 
    preprocessed images to 
    one input images

    @ Parameters:
    -------------
    GOOD_IMG_PATH: str
        Path of the folder containing
        good images
    BAD_IMG_PATH: str
        Path of the folder containing
        bad images
    radius: int
        Radius of the median filter 
        applied to the image

    @ Returns:
    ----------
    combined_img: np.array
        filtered and pre-processed combined
        images arrays of both good and clear 
        iamges
    labels: np.array
        labels containing 1, if images is good
        and 0, if image is bad

    """
    print ('Pre-Processsing the Data...........\n')
    # Reading the Good Images 
    good_img = []
    for filename in os.listdir(GOOD_IMG_PATH):
        good_img.append(ms.imread(GOOD_IMG_PATH+filename, mode='L'))
    good_img = np.asarray(good_img)

    # Reading the Bad Images 
    bad_img = []
    for filename in os.listdir(BAD_IMG_PATH):
        bad_img.append(ms.imread(BAD_IMG_PATH+filename, mode='L'))
    bad_img = np.asarray(bad_img)

    # Concatenate the array of Good & Bad images
    combined_img = np.concatenate((good_img, bad_img))  
    labels = np.concatenate((np.ones(good_img.shape[0]), 
                            np.zeros(bad_img.shape[0])))
 
    # Filtering the combined images to Reduce the Noise present
    combined_img = nd.median_filter(combined_img, radius)

    return combined_img, labels


def save_data(train_images, train_labels,
            test_images, test_labels):
    """
    Checking the existence of path
    if not exists then creates one
    and save the train & test data
    """

    if path_validation(TRAIN_DATA_PATH):
        print ('Train Data Path Success .....')
    if path_validation(TRAIN_LABEL_PATH):
        print ('Train Label Path Success .....')
    if path_validation(TEST_DATA_PATH):
        print ('Test Data Path Success .....')
    if path_validation(TEST_LABEL_PATH):
        print ('Test Label Path Success .....')

    print('\nSaving the splitting results......\n')
    np.save(TRAIN_DATA_PATH,train_images)
    np.save(TRAIN_LABEL_PATH,train_labels)
    np.save(TEST_DATA_PATH,test_images)
    np.save(TEST_LABEL_PATH,test_labels)


def NN_Model(neurons, good_initializer=False):
    """
    Intializing an 2 Layer Neural Network Model
    with random value of weights.

    @ Parameters:
    -------------
    neurons: tuple
        Dimension(no of neuron) in 
        input layer -> hidden layer-> output layer
        in the same order(input, hidden, output)
    good_initializer: bool
        If set to True, intialize the network with good
        set of intial weight to parameters
        extracted from saome paper
        else, intialized the parameters to default 
        normal random values from [-1,1]
    
    @ Returns:
    ----------
    param: dict
        parameters theta1 and theta2
        with randomly intialized values

    """

	# Good Weight Initialization Cited from Paper
    print('\nIntializing the Model...........\n')
    if good_initializer:
        weight = np.sqrt(6.0/(neurons[0] + neurons[-1]))
    else:
        weight = 1.0

    # Intializing the theta parameters
    theta1 = 2.0*np.random.random((neurons[1],neurons[0]))*weight - 1.0*weight
    theta2 = 2.0*np.random.random((neurons[-1],neurons[1]+1))*weight - 1.0*weight

    # Store the paramaters to dictionary
    param = {'Theta1':theta1, 'Theta2':theta2}

    return param


def derivative(a,func='sig'):
    """
    Derivative f'(x) of correspnding Activation Function
    which is applied to neurons used in calculatng
    backpropgation

    @ Parameters:
    -------------
    a: np.array
        Activated neurons
    func: str
        Activation function whose
        derivative needs to be calculated

    @ Returns:
    ----------
    derivative of activated neurons

    """

    # tanh derivative function
    if func == 'tanh':
        return (1 - a*a)
    # Identity derivative function
    if func == 'none':
        return 1
    # Softplus derivative function
    if func == 'softplus':
        return 1.0/(1 + np.exp(-a))
    # Noisy ReLU , Noise is added to it.
    # Derivative function
    if func == 'relu':          
        a[a >= 0.00] = 1.00
        a[a < 0.00] = 0.01
        return a

    # Sigmoid derivative function
    return a*(1-a)


## Diagram to show the Weight and Input matrix Multiplication

#==============================================================================
#                                        --- Total examples -----
#  [theta0 theta1 t2 t3 t4 ........ ]  x0 x10
#              Total features          x1 x11 ... .. . .. .. ..   
#                    .                 x2 x12
#                    .                 x3 .
#                    .                 x4
#                    .                 x5
#                    .                 ..
#                    .                 .
#                    .                 .
#                    .                 . 
#                    .                 . .
#                    .                 xn x1n .........
#==============================================================================
#==============================================================================


def cost(act_val, target, theta, lambdaa):
    """
    Cost Function  with L2 regularization 
    Not penalizing the Bias terms in weight values
    using softmax/max-likelihood loss function

    @ Parameters:
    -------------
    act_val: np.array
        Activated value of last layer neurons
    target: np.array
        Value of target class
    theta: dict
        Conatins the trained theta weights
        used to calculate the error
    lambdaa: float
        Intensity of regularization to be
        applied to final results 

    @ Returns:
    ----------
    overall cost of that epochs
    or NN network trained

    """

    # Regularization computation ignoring
    # bias '0' term in it
    reg = (lambdaa/2.0)*(np.sum(theta['Theta1'][1:,:]**2)
						+ np.sum(theta['Theta2'][1:,:]**2))
    reg = reg/float(target.shape[0])

    # MAx-Likelihood Calcaulatoin as in 
    # like Logistics Regression
    first = (-1.0) * ( target*np.log(act_val) + (1-target)*np.log(1 - act_val))

    # Actual cost = Total cost - regualrized
    return (np.mean(first) + reg)


def back_propagate(theta1, theta2, train_images,
                    train_labels, nclass, alpha=0.001, lambdaa=0.0007,
                    max_iter=50, act='sig', batch_size=32, logging=1):
    """
    Method of updating the weights in NN Model
    by taking gradients of theta using cost function
    thata = theta - f('theta)

    Mini-batch gradient descent, applied to get the 
    gradient of the theta.
    Here updation of weights use momentum factor(gamma)
    so as to approach global minima faster
    Core of ANN, BackProp..

    @ Parameters:
    -------------
    test_images: np.array
        Contains the test_images whose labels need
        to be predicted
    test_labels: np.array
        Contains the labels(1/0)
        corresponding to selected images
    theta1: np.array
        Contains the trained theta weights
        corresponding to input->hidden layer
    theta2: np.array
        Contains the trained theta weights
        corresponding to hidden->output layer
    train_images: np.array
        Contains the train_images used to learn
        the weights of networks
    train_labels: np.array
        Contains the labels(1/0)
        corresponding to train_images
    nclass: int
        No of unique class present in the 
        training dataset
    alpha: float
        Learning rate, rate at which each gradient
        update take place
    lambdaa: float
        Regularization term which penalizes
        the cost function
    max_iter: int
        No of epochs to be performed on 
        data to learn the weights
    act: str
        Activation function which is applied to 
        the neurons in forward propagation
    batch_size: int
        No of images,labels to be fetched from
        overall data at each iterations for
        updation of weights   
    logging: int
        Steps at which logs are displayed
        or recorded

    @ Returns:
    ----------
    parameters: dict
        trained theta1,theta2 
        and per epoch Loss values

    """

    # Used to store theta1 & theta2 
    parameters = {}
    # Momentum Factor
    gamma = 0.9 
    # Intial dtheta values used for
    # momentum 
    dtheta1 , dtheta2 = 0.0, 0.0
    # One-Hot labelling the labels of data
    one_hot = output_encoding(train_labels, nclass)

    # Used to store best theta1 and theta2 values
    # whose error was least in whole epochs
    best_theta1, best_theta2 = (np.zeros((theta1.shape[0],theta1.shape[1])),
                                np.zeros((theta2.shape[0],theta2.shape[1])))

    # Store the value of cost in each epochs 
    cost_list = []

    # Global Min Error term
    err = 100.0

    for epoch in np.arange(0,max_iter):
        # Used to print results of result summary
        k = 0
        print
        print ('\nOverall Min. Error rate : ' + str(err))
        print
		# Softmax in Final Layer 
        for batchX , batchY in get_batch(train_images,one_hot,batch_size):
            m, n = batchX.shape
            a2 = h(theta1,batchX,act)
            a2 = np.insert(a2, 0, 1, axis=0)
            a3 = h(theta2,a2.T,func='softmax')
            eps = alpha/float(m)

			# Error in Hidden and Output Layer
            delta3 = (a3 - batchY)*derivative(a3,'none')
            delta2 = ((theta2.T).dot(delta3))*derivative(a2,act)

			# Gradient of Theta Matrices
            ktheta1 = np.dot(delta2[1:,:],batchX)
            ktheta2 = np.dot(delta3,a2.T)

			# Momemtum Part to Accelerate the Learning Rate
            dtheta1 = eps*(ktheta1 + lambdaa*theta1) + gamma*dtheta1
            dtheta2 = eps*(ktheta2 + lambdaa*theta2) + gamma*dtheta2
            theta1 = theta1 - dtheta1
            theta2 = theta2 - dtheta2

			# Cost Per Batch iteration
            cost_epoch = cost(a3,batchY, {'Theta1':theta1, 'Theta2':theta2}, lambdaa)
            cost_list.append(cost_epoch)
            
			# Summary of Back Prop
            if (k % LOGGING_STEPS  == 0):
                accuracy = model_score({'Theta1':theta1, 'Theta2':theta2}, 
                                        train_images, train_labels, act)
                error = 100.0 - accuracy

				# Error Updation if LEss Error is Discovered
                if(error < err):
                    err = error
                    # Store the best theta of least error
                    best_theta1 = theta1
                    best_theta2 = theta2
                
				# Info of Learning of NN
                print ("Epoch " + str(epoch+1) + " in " + str(k+1) + " iter"+ " | "
                        "Train Error rate: " + str(error) + "%" + " | Batch loss: " 
                        + str(cost_epoch))
 		    
            k = k + 1
            
    parameters = {'Theta1':best_theta1, 'Theta2':best_theta2, 'Loss':cost_list}
    return parameters
        

def get_batch(img, labels, batch_size):
    """
    Extracting data in batches of given batch_size
    in each epoch in Training

    @ Parameters:
    -------------
    img: np.array
        Contains the images
    labels: np.array
        Contains the labels(1/0)
        corresponding to selected images
    batch_size: int
        No of images,labels to be fetched from
        overall data at each iterations for
        updation of weights

    @ Returns:
    ----------
    corresonding batches of images & labels 

    """
    for i in np.arange(0, img.shape[0], batch_size):
        yield(img[i:i+batch_size,:],labels[:,i:i+batch_size])

        
def output_encoding(labels, nclass):
    """
    Convert the labels to classes dimension
    same as one_hot_encoding()
    Make entry correpsonds to each class as 1(one)
    and rest all as zero; thus provides
    each label vector corresponds to each images

    @ Parameters:
    -------------
    labels: np.array
        Labels(1/0) or class
        corresponding to selected images
    nclass: int
        No of unique class present in the 
        training dataset

    @ Returns:
    ----------
    one_hot: np.array
        One Hot vector with size of nclass having
        1 at index corresponds to class

    """
    one_hot = np.zeros((nclass,labels.shape[0]))
    for  c in np.arange(0,nclass):
        pos = np.where(labels==c)
        one_hot[c][pos] = 1 
    return one_hot


def show_plot(cost, PLOT_PATH):
    """
    Plot the Cost vs Iteration Curve

    @ Parameters:
    -------------
    cost: np.array
        Contains the cost calculated 
        in every iterations
    PLOT_PATH: str
        Path where the cost vs iteratiom
        curve get saved

    @ Returns:
    ----------
        Gives the plot showing error rate
        behaviour wrt each epochs

    """

    plt.plot(np.arange(0,len(cost)) , cost)
    plt.title("Cost Vs Iteration Curve ")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show() 
    fig = plt.gcf()
    if(path_validation(PLOT_PATH)):
        fig.savefig(PLOT_PATH)
    fig.clf()


def main():
    """
    Pre-process the data with filtering, resizing 
    and trained the Neural Networks with 
    resulting pre-processed data using backpropagation

    """

    ## Input Layer -> 10001 U
    ## 1 Hidden Layers -> 300 HU 
    ## 1 Output Layer -> 2 Neurons
     
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-path1", "--good_path", required=True, 
                        help="path to good images directory")
    ap.add_argument("-path2", "--bad_path", required=True, 
                        help="path to bad images directory")
    args = vars(ap.parse_args())

    # Taking Absolute Path neede for reading images
    GOOD_IMG_PATH = os.path.abspath(args["good_path"]) + str('/')
    BAD_IMG_PATH = os.path.abspath(args["bad_path"]) + str('/')

    # Path Validation
    if not path_validation(GOOD_IMG_PATH, read_access=True):
        exit(0)
    if not path_validation(BAD_IMG_PATH, read_access=True):
        exit(0)

    # Model Path Vaildation
    if path_validation(MODEL_PATH):
        print ('\nModel Path Success .....\n')


    # Getting the Same Result in Shuffle in each Run.
    np.random.seed(SEED)

    # Convert the Good & Bad Images to Cumulative numpy array 
    imgs, labels = data_preprocess(GOOD_IMG_PATH, BAD_IMG_PATH, radius=RADIUS)
    			
    # Resizing the feature space for easier to handle
    imgs = resize(imgs, width=WIDTH, height=HEIGHT)

    # Splitting the Data for Training and Testing Purpose
    print('\nSplitting of Data......\n')
    train_images, test_images, train_labels, test_labels = train_test_split(imgs, labels, 
                                                        test_size=SPLIT_RATIO, random_state = SEED) 

    # Saving the splitted data to disk
    save_data(train_images, train_labels, test_images, test_labels)

    # No of unique class in data
    nclass = np.unique(labels).shape[0]
    			
    # Addition of Bias in Train/Test Images
    train_images = np.insert(train_images, 0, 1, axis=1) 
    test_images = np.insert(test_images, 0, 1, axis=1)

    # May Used for Cal No Of Neuron as hyper-parameters to Good value
    no_of_neurons = train_images.shape[0]/(2*(train_images.shape[1]+10))


    # Intializing the Model
    theta = NN_Model([train_images.shape[1],NEURONS_SIZE,nclass])

    print ("BAckPROP .................\n")
    params = back_propagate(theta['Theta1'], theta['Theta2'], train_images, train_labels,
                            nclass, alpha=ALPHA, lambdaa=LAMBDA, max_iter=MAX_ITER, act=ACT, 
                            batch_size=BATCH_SIZE, logging=LOGGING_STEPS)

    # Accuracy Score on Train set
    accuracy = model_score(params, train_images, train_labels, act=ACT) 
    print('\nAccuracy on Train Data: ', accuracy)

    # Accuracy Score on test set
    accuracy = model_score(params, test_images, test_labels, act=ACT) 
    print('\nAccuracy on Test Data: ', accuracy)

    # Storing the Results in tmp directory 
    print ('\nSaving Results...............\n')
    joblib.dump(params, MODEL_PATH)

    # Plotting the Curve
    show_plot(params['Loss'], PLOT_PATH)


if __name__ == "__main__":
    main()