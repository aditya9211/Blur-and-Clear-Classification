# Blur-and-Clear Images Classification
## Classifying the Blur and Clear Images

### Introduction
In day to day Life, we encounter the poor images clicked from our Camera due to poor focus, a motion of objects in the frame
or handshaking motion while capturing the Images.

`Blur is typically the thing which **suppress the high-frequency** of our Images, therefore can be detected by using various low-pass filter
eg. Laplacian Filter.`

As a smart person(myself a CS guy) we doesn't want to manually filter out the Clear and Blurred Images,
so we need some smart way to delete the unnecessary Images.

## LoG Filter
I also applied the Laplacian of gaussian(**[LoG](http://academic.mu.edu/phys/matthysd/web226/Lab02.htm/)**) filter to detect the blur images, but it was difficult to find the
exact value of the threshold needed to differentiate images; despite that results were not fascinating.

Used variance of **[LoG filter](https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)**  

Some of its discussions

https://stackoverflow.com/questions/7765810/is-there-a-way-to-detect-if-an-image-is-blurry

https://stackoverflow.com/questions/5180327/detection-of-blur-in-images-video-sequences

LoG Ref: 

http://aishack.in/tutorials/sift-scale-invariant-feature-transform-log-approximation/

Repo which implemented LoG filter in Python:
https://github.com/WillBrennan/BlurDetection2


`As the Now, the era of Deep Conv Nets has suppressed the Standard Computer Vision Techniques, 
Thus I focussed on the root of it which is Neural Nets.`
`Neural Nets learn very Quickly the complex features, therefore can be used much easily then std. CV technique.
Tuning ANN efficiently can provide me the results much better than CV TEchnique.`

## Neural Network Model
`Model has 3 Layers`
`Containing`
```
 1 Input Layer -> 100*100 U
 
 1 Hidden Layer -> 300 HU
 
 1 Output Layer -> 2 U
```
**I have used the Backprop Algorithm for Training ANN using the SGD Optimizer with Momentum.
Rescaled the Images to 100 x 100 Pixels in Grayscale Coding and done median filtering to filter out the noise from Images.**


# Quick Start
`Need the Images that are clear in the separate folder and one with blurred in another folder.`
```ruby

# Python3+ user install Tkinter package (Python 3.5.xx)
# Currently code is supported for Python 3.5.xx version only
sudo apt-get install python3-tk
# Clone the repo
git clone https://github.com/aditya9211/Blur-and-Clear-Classification.git
# Change the working Directory
cd Blur-and-Clear-Classification/
# Install the requirements
pip install -r requirements.txt
# Train the Network
python train.py  --good_path  '/home/......good/'  --bad_path  '/home/......./bad/'
# Test the Network 
python test.py
# Predict output 
python predict.py --img '/home/....../laptop.png'

```

## Code Structure
Code is segmented as follows:

1. Training Part :

	__train.py__
    
 	`which train the neural network with given images
 	and stores the trained parameters and splitted train, test set to disk `
    
2. Testing Part :

	__test.py__
   
 	`test the neural network with test data
 	stored by train.py`

3. Predict Part :

	__predict.py__
    
	`predict the label of images(Good/Bad) 
	provided by argument while calling`

4. Config File :

	__config.py__
    
	`contains the list of constant used by files
	or hyper-parameters which can be changed
	by editing this file`
	
5. Utiltities Part :

	__utils.py__
    		
	`helper functions or common function among used in train/test and predict`
	
6. Requirement Package :

	__requirements.txt__
	
	`packages required for running scripts`
	
