# Blur-and-Clear Images Classification
## Classifying the Blur and Clear Images

### Program to Classifying the Blur and Clear Images Using Supervised Learning by Neural Networks Model

In day to day Life we encounter the poor images clicked from our Camera due to poor focus, motion of objects in frame
or handshaking motion while capturing the Images.

`Blur is typically the thing which suppress the high frequency of our Images, therefore can be detected by using various low-pass filter
eg. Laplacaian Filter. `

As a smart person(myself a CS guy) we doesn't want to manually filter out the Clear and Blurred Images,
so we need some smart way to delete the uneccessary Images.

I also applied the Laplacian of gausssian filter to detect the blur images, but it was difficult to find
exact value of threshold needed to seggregate; despite results were also not fascinating.
**Used variance LoG filter mentioned in https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/**

Some of its discussions 
https://stackoverflow.com/questions/7765810/is-there-a-way-to-detect-if-an-image-is-blurry
https://stackoverflow.com/questions/5180327/detection-of-blur-in-images-video-sequences

LoG Ref: 
http://academic.mu.edu/phys/matthysd/web226/Lab02.htm
http://aishack.in/tutorials/sift-scale-invariant-feature-transform-log-approximation/

Repo which implemented LoG filter in Python
https://github.com/WillBrennan/BlurDetection2


As the Now the era of Deep Conv Nets has supressed the Standard Computer Vision Techniques, 
Thus I focussed on the root of it which is Neural Nets.

`Neural Nets learn very Quickly the complex features, therefore can be used much easily then std. CV technique.
Tuning ANN efficiently can provide me the results much better than CV TEchnique.`


## Here the Dependencies Required for Running the Code:
1. Python 2.7xx or 3.5xx
2. use `pip install -r requirements.txt'` to fulfill the dependencies


Code are segmented as follows:

1. Training Part :
    **train.py**
 	which train the neural network with given images
 	and stores the trained parameters and splitted train, test set to disk 
    
2. Testing Part :
    __test.py__
 	test the neural network with test data
 	stored by train.py 

3. Predict Part :
    __predict.py__
	predict the label of images(Good/Bad) 
	provided by argument while calling

4. Config File :
    __config.py__
	contains list of constanst used by files
	or hyper-parameters which can be changed
	byediting this file
	
5. Utiltities Part :
    __utils.py__
    	helper functions or coomon function among used in train, test
	and predict

`Model has 3 Layers`
`Containing`
```
 1 Input Layer -> 100*100 U
 
 1 Hidden Layer -> 300 HU
 
 1 Output Layer -> 2 U
```

**I have used the Backprop Algorithm for Training ANN using the SGD Optimizer with Momentum.
Rescaled the Images to 100 x 100 Pixels in Grayscale Coding and done median filtering to filter out the noise from Images.**

`Need the Images that are clear in separate folder and one with blurred in other folder.
 Because it is a training phase of Supervised Learning `




*Run as :*

`python train.py  --good_path  '/home/......'  --bad_path  '/home/.......'`

       `and result get stored default in 'MODEL_PATH' configures in config.py file.`
 

`python test.py`

       `to test the results and gives the accuracy score`

`python predict.py --img '/home/...../..jpg'`

	`to predict the labels of given images`
