# Blur-and-Clear Images Classification
## Classifying the Blur and Clear Images

### Program to Classifying the Blur and the Clear Images by Using Supervised Learning by Neural Networks Model

In day to day Life we encounter the poor images clicked from our Camera due to poor focus, motion of objects in frame
or handshaking motion while capturing the Images.

As a CS Student we wnat to filter out the the Clear and Blurred Images to delete thee uneccessary Images.

`Blur is typically the thing which suppress the high frequency of our Images, therefore can be detected by using various low-pass filter
eg. Laplacaian Filter. `

As the Now the era of Deep Conv Nets has supressed the Standard Computer Vision Techniques, 
Thus we are focussing on the root of it which is Neural Nets.
`
Neural Nets very Quickly learn the complex features , therefore can be used much easily then std. CV technique.
Tuning ANN efficiently can provide us the results much better than CV TEchnique.`


## Here the Dependencies Required for Running the Code:
1. Python 2.7xx
2. Numpy , scipy, matplotlib Library Installed 

Code are segmented as follows:

1. Training Part :
    **train.py**
    
2. Testing Part :
    __test.py__

`Our Model has 3 Layers
Containing

 1 Input Layer -> 100*100 U
 
 1 Hidden Layer -> 300 HU
 
 1 Output Layer -> 2 U`


**We have used the Backprop Algorithm for Training using the SGD Optimizer with Momentum .
Rescaled the Images to 100 x 100 Pixels in Grayscale Coding and done median filtering to filter the noise in Images.**

`Need the Images that are clear in separate folder and one with blurred in other folder.Because it is a supervised Learning`


Run as :
**python train.py  --good_path  '/home/......'  --bad_path  '/home/.......'**

`and result get stored default in 'tmp/blur_clear/' Folder.`
 

**python predict.py**

`to predict the results.'
