# -*- coding: utf-8 -*-
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import scipy.misc as ms
import scipy.ndimage as nd
import argparse

def resize(X, orig_reso, w=100, h=100):
	## Resize the Image to Default 100 x 100 pixels Image
    r,c = w,h
    X_new = np.zeros((X.shape[0],r*c))
    for i in range(X.shape[0]):
        X_new[i,:] = ms.imresize(X[i,:].reshape(orig_reso[0],orig_reso[1]),(r,c),interp='cubic').flatten()
    return X_new

def NN_Model(neuron,initialize=False):
	## Good Weight Initialization Cited from Paper
    if initialize:
        r1 = np.sqrt(6.0/(neuron[0] + neuron[-1]))
    else:
        r1 = 1.0
    theta1 = 2.0*np.random.random((neuron[1],neuron[0]))*r1 - 1*r1
    theta2 = 2.0*np.random.random((neuron[-1],neuron[1]+1))*r1 - 1*r1
    return {'Theta1':theta1, 'Theta2':theta2}
    
def data_preprocess(path1, path2, orig_reso):
 print 'Pre-Processsing the Data...........'
 path1 = path1 + str("/*.jpg")
 path2 = path2 + str("/*.jpg")
 ## Contents counts in a Dir
 files = glob.glob(path1)
 good_count = len(files)

 files = glob.glob(path2)
 bad_count = len(files)
	
 ## Converting the Images in flatten numpy array format 
 i=0
 X2 = np.zeros((good_count,orig_reso[0]*orig_reso[1]))
 for filename in glob.glob(path1):
     inp_image = ms.imread(filename, mode='L')
     X2[i] = inp_image.flatten()
     i = i + 1 
 i=0
 X1 = np.zeros((bad_count,orig_reso[0]*orig_reso[1]))
 for filename in glob.glob(path2):
     inp_image = ms.imread(filename, mode='L')
     X1[i] = inp_image.flatten()
     i = i + 1 
  
 ## Concatenate the Array of Images that are Good and Bad
 X = np.concatenate((X1, X2))  
 y = np.concatenate((np.zeros(bad_count),np.ones(good_count)))
 
 ## Filtering the Image to Reduce the Noise in Image
 X = nd.median_filter(X,3)
 return X, y

def sigmoid(X): ## Sigmoid activation Function
    return 1.0 / (1.0 + np.exp(-X))

## Different Activaion Function
def h(theta,X,func='sig'):
    a = theta.dot(X.T)
    if(func== 'tanh'):
        return np.tanh(a)
    if func == 'none':
        return a
    if func == 'softplus':
        return np.log(1 + np.exp(a))
    if func == 'relu':
        return np.maximum(0.01*a, a)
    
    if func == 'softmax':
        a1 = np.exp(a)
        a1 = a1 / np.sum(a1, axis = 0, keepdims = True)
        return a1
    
    return sigmoid(a)

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

## Cost Function Implementing L2 Norm, Not
	## penalizing the Bias terms in weight values
def cost(a4, y_new, theta, lambdaa):
    reg = (lambdaa/2.0)*(np.sum(theta['Theta1'][1:,:]**2)
						+ np.sum(theta['Theta2'][1:,:]**2))
    reg = reg/float(y_new.shape[0])
    first = (-1.0) * ( y_new*a4 + (1-y_new)*np.log(1 - a4))
    return (np.mean(first) + reg)

## Derivative of Correspnding Activation Function
def derivative(a,func='sig'):
    if func == 'tanh':
        return (1 - a*a)
    if func == 'none':
        return 1
    if func == 'softplus':
        return 1.0/(1 + np.exp(-a))
    if func == 'relu': 			## Noisy ReLU , Noise is added to it.
        a[a >= 0.00] = 1.00
        a[a < 0.00] = 0.01
        return a
    
    return a*(1-a)

## Core of ANN, BackProp..
def back_propagate(X1, y1, theta1, theta2, X, y, alpha, lambdaa, nclass,  max_iter
                   , act, batch_size=32):
    parameters = {}
    gamma = 0.9 ## Momentum Factor
    dtheta1 , dtheta2 = 0.0, 0.0
    y_new = output_encoding(y, nclass) ## Convert the value of labels to dimension of classes
    theta1_up, theta2_up = np.zeros((theta1.shape[0],theta1.shape[1])), np.zeros((theta2.shape[0],theta2.shape[1]))
    cost_new = []
    err = 100.0
    for j in np.arange(0,max_iter):
        k = 0
        print
        print 'Overall Min. Error rate : ' + str(err)
        print
		
		## Softmax in Final Layer 
        for batchX , batchY in get_batch(X,y_new,batch_size):
            m, n = batchX.shape
            a2 = h(theta1,batchX,act)
            a2 = np.insert(a2, 0, 1, axis=0)
            a3 = h(theta2,a2.T,func='softmax')
            eps = alpha/float(m)

			## Error in Hidden and Output Layer
            delta3 = (a3 - batchY)*derivative(a3,'none')
            delta2 = ((theta2.T).dot(delta3))*derivative(a2,act)

			## Gradient of Theta Matrices
            ktheta1 = np.dot(delta2[1:,:],batchX)
            ktheta2 = np.dot(delta3,a2.T)

			## Momemtum Part to Accelerate the Learning Rate
            dtheta1 = eps*(ktheta1 + lambdaa*theta1) + gamma*dtheta1
            dtheta2 = eps*(ktheta2 + lambdaa*theta2) + gamma*dtheta2
            theta1 = theta1 - dtheta1
            theta2 = theta2 - dtheta2

			## Cost Per Iteration
            cost_new.append(cost(a3,batchY, {'Theta1':theta1, 'Theta2':theta2}, lambdaa))
            
			## Summary of Back Prop
            if (k % 1  == 0):
                pred_y = validate(theta1, theta2, X1, act)
                y1 = y1.flatten()
                error = 100.0 - np.mean(pred_y == y1)*100.0

				## Error Updation if LEss Error is Discovered
                if(error < err):
                    err = error
                    theta1_up = theta1
                    theta2_up = theta2
                
				## Info of Learning of NN
                if k == 0:
                    print "Epoch " + str(j+1) + " in " + str(k+1) + " iterations"+ " Error rate :  " + str(error) + "%" + " loss: " + str(cost(a3,batchY, {'Theta1':theta1, 'Theta2':theta2}, lambdaa)) 
                else:
                    print "Epoch " + str(j+1) + " in " + str(k+1) + " iterations"+ " Error rate :  " + str(error) + "%" + " loss: " + str(cost(a3,batchY, {'Theta1':theta1, 'Theta2':theta2}, lambdaa))
 		    
            k = k + 1
            
    parameters = {'Theta1':theta1_up, 'Theta2':theta2_up, 'Loss':cost_new}
    return parameters
        
        
## Extracting the Batch per Epoch in Training  
def get_batch(X, y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        yield(X[i:i+batch_size,:],y[:,i:i+batch_size])
        
## Find the Result of Model
def validate(theta1, theta2, X, act = 'sig'):
    aa1 = h(theta1,X,act)
    aa1 = np.insert(aa1, 0, 1, axis=0)
    aa2 = h(theta2,aa1.T,'softmax')
    accu_matrix = np.argmax(aa2,axis=0) 
    return accu_matrix

## Plot the Cost vs Iteration Curve
def show_plot(cost):
    plt.plot(np.arange(0,len(cost)) , cost)
    plt.xlabel('Iterations.......')
    plt.ylabel('Loss.............')
    plt.show() 
    
## Convert the labels to classes dimension
## same as one_hot_encoding()
def output_encoding(y, nclass):
    y_new = np.zeros((nclass,y.shape[0]))
    for  c in np.arange(0,nclass):
        pos = np.where(y==c)
        y_new[c][pos] = 1 
    return y_new
   
## Input Layer -> 10001 U
## 1 Hidden Layers -> 300 HU 
## 1 Output Layer -> 10 Neurons

## Getting the Same Result in Shuffle in each Run.
np.random.seed(10)
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-path1", "--good_path", required=True, help="path to good images directory")
ap.add_argument("-path2", "--bad_path", required=True, help="path to bad images directory")
ap.add_argument("-model_dir", "--model_dir", required=False, help="path to model directory")
args = vars(ap.parse_args())
path1 = args["good_path"]
path2 = args["bad_path"]
			
## Orignal Resolution Of Image
orig_reso = (320,240)
			
## Convret the Images too Corresponding Numpy array 
X, y = data_preprocess(path1, path2, orig_reso)
			
## Resizing the feature space for easier to handle
X = resize(X, orig_reso)

## Splitting the Data for Training and Testing Purpose
X, X1, y, y1 = train_test_split(X, y, test_size=0.3, random_state = 10)

## Creating the Temp Folder for Storing the Result 
if args["model_dir"]:
	filename = args["model_dir"]+str("/")
else:
  filename = "/tmp/blur_clear/"

if not os.path.exists(os.path.dirname(filename)):
	print 'Creating Dir ' + filename + '....'
	os.makedirs(os.path.dirname(filename))

## Storing the Array required for predicting Purposes
np.save(filename + str("/train_images.npy"),X)
np.save(filename + str("/train_labels.npy"),y)
np.save(filename + str("/test_images.npy"),X1)
np.save(filename + str("/test_labels.npy"),y1)

## Rescalling the Training Dataset
X = X/255.0
X = np.insert(X, 0, 1, axis=1) ## Adding the Biases

## Rescalling the Testing Dataset
X1 = X1/255.0
X1 = np.insert(X1, 0, 1, axis=1) ## Adding the Biases

## Parameters for Model
max_iter = 50
alpha = 0.001
lambdaa = 0.0007
nclass = 2
act = 'sig'
			
## May Used for Cal No Of Neuron as hyper-parameters to Good value
nof_neuron = X.shape[0]/(2*(X.shape[1]+10))
theta = NN_Model([X.shape[1],300,nclass])
print "BAckPROP ................."
print
params = back_propagate(X1, y1, theta['Theta1'], theta['Theta2'], X, y, alpha, lambdaa, nclass,
                        max_iter, act, batch_size=10)

## Calculating the predicted labels
pred_y = validate(params['Theta1'], params['Theta2'], X1, act)
y1 = y1.flatten()
accuracy = np.mean(pred_y == y1)*100

## Accuracy of Our Model
print 'Accuracy :' + str(accuracy) + ' '

## Storing the Results in tmp directory 
print 'Saving Results...............'
np.save(filename + str("/result.npy"),{'Theta1':params['Theta1'],'Theta2':params['Theta2']})

## Plotting the Curve
show_plot(params['Loss'])


