# Blur-and-Clear Images Classification
Classifying the Blur and Clear Images

Program to Classifying the Blur and the Clear Images by Using Supervised Learning by Neural Networks Model

In day to day Life we encounter the poor images clicked from our Camera due to poor focus, motion of objects in frame
and handshaking motion while capturing the Images.

So we get annoyed to filter the Clear rand Blurred Images so as to sort the Images to keep or to delete.

Blur is typically the thing which suppress the high frequency of our Images, therefore can be detect by using low-pass filter
eg. Laplacaian Filter. 

As the Now the era of Deep Conv Nets has supressed the Standard Computer Vision Techniques, 
So in This we are focussing on the root of it which is Neural Nets.

Neural Nets very Quickly learn the complex features , therefore can be used much easily then std. CV technique.
Tuning it to very efficiently can provide the results much better than CV TEchnique.


Here the Dependencies Required for Running the Code:
1. Python 2.7xx
2. Numpy , scipy, matplotlib Library Installed

Code are segmented as follows:
1. Training Part :

    train.py
  
2. TEsting Part :

    test.py
  
Our Model has 3 Layers
Containg
1 Input Layer -> 10001 U
1 Hidden Layer -> 300 HU
1 Output Layer -> 2 U
We have used the Backprop Algorithm for Training using the SGD Optimizer with momentum .
Rescaled the Images to 100 x 100 Pixels in Grayscale Coding and doing median filtering to filter the noise in Images.

Need the Images that are clear in separate folder and one with blurred in other folder.Because it is a supervised Learning.


Run as :
1. python train.py  --good_path  '/home/......'  --bad_path  '/home/.......'

and result get stored default in 'tmp/blur_clear/' Folder.
 

2. python predict.py

to predict the results.
