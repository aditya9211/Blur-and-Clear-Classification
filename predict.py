import numpy as np
import os

## Sigmoid activation Function
def sigmoid(X):
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

def validate(theta1, theta2, X, act = 'sig'):
    aa1 = h(theta1,X,act)
    aa1 = np.insert(aa1, 0, 1, axis=0)
    aa2 = h(theta2,aa1.T,'softmax')
    accu_matrix = np.argmax(aa2,axis=0) 
    return accu_matrix

filename = "/tmp/blur_clear"
if not os.path.exists(os.path.dirname(filename)):
	print "No dir exists" + filename
	exit()

## Extracting the Info from Training Set Results from tmp dir
params = np.load("/tmp/blur_clear/result.npy")
test_images = np.load("/tmp/blur_clear/test_images.npy")
test_labels = np.load("/tmp/blur_clear/test_labels.npy")

## Rescalling the Inputs
test_images = test_images/255.0
test_images = np.insert(test_images, 0, 1, axis=1)

## Predicting the Labels , Accuracy Score
pred_y = validate(params[()]['Theta1'], params[()]['Theta2'], test_images)
test_labels = test_labels.flatten()

accuracy = np.mean(pred_y == test_labels)*100
print 'Accuracy : ' + str(accuracy) + ' %'
