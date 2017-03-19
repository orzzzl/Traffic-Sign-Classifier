#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./NewPictures/1.jpg "Traffic Sign 1"
[image5]: ./NewPictures/2.jpg "Traffic Sign 2" 
[image6]: ./NewPictures/3.jpg "Traffic Sign 3"
[image7]: ./NewPictures/4.jpg "Traffic Sign 4"
[image8]: ./NewPictures/5.jpg "Traffic Sign 5"
[image9]: ./examples/dis.jpg "New Distribution"
[image10]: ./examples/pre.jpg "After Proprocessing"
 
## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/orzzzl/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I did data-normalization with the equation:
X = X / 255 - 0.5 
to convert the value from 0-255 to -0.5 - 0.5
![alt text][image2]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I load the train.p, valid.p, test.p as my training, validation, testing dataset.  

My final training set had 86010 number of images. My validation set and test set had 4410 and 12630 number of images.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following:

1. Rotation of a silght angle.
2. Translation of a slight distandce.
3. A small amount of brightness change.

After data augamentation, I have the same amount of data samples per class:
![alt text][image9]

Data visualization after all preprocessing:
![alt text][image10]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24				|
| Convolution 3x3	    |  1x1 stride, VALID padding, outputs 12x12x32     									|
|RELU |
| Max pooling	      	| 2x2 stride,  outputs 6x6x32
| Convolution 3x3	    |  1x1 stride, VALID padding, outputs 4x4x64    									|
|RELU |
| Max pooling	      	| 2x2 stride,  outputs 2x2x64 
| Fully connected		| Input: 2x2x64 = 256, Output: 512     									|
RELU||
Dropout | 50% |
| Fully connected		| Input: 2x2x64 = 512, Output: 256									|
RELU||
Dropout | 50% |
| Fully connected		| Input: 2x2x64 = 256, Output: 43     									|
RELU||
Dropout | 50% |
Softmax				|       		|
Regulation | L2 regulation|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Beside the classic LeNet structure, I have added the following to imporove the performance:

1. Drpoout to reduce over-fitting.
2. L2 regulation(add weights to the metrics which is to be optimized).


Hyperparameters:

Learning rate = 0.001

Epochs = 14

Batch size = 128

Drop rate = 0.5

Regulation Rate = 0.000001

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.970
* test set accuracy of 0.940

* What architecture was chosen?
LeNet

* Why did you believe it would be relevant to the traffic sign application?


   Because this model works well on image object recognization and traffic sign is a perfect image sign to be recognize from.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
 Becase this model has a fairly high accuracy not only on training and valid set but also on the testing dataset which I have never exposed to the model before.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Pic1: 

Most traffic signs are tri-angles. And this traffic sign is round. Not sure if this will cause some problem on recognization the effective parts of the picture.

Pic2:

The traffic sign is a little bit small and this may cause problem.

Pic3:

Perfect traffic sign picture. Nothing to complain about this pic.

Pic4:

Also the traffic sign is not tri-angle. But again this may not be a problem at all.

Pic5:

The traffic sign is pretty small and has an angle with the z-direction.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)    		| Speed limit (30km/h)   									| 
| Children crossing     			| Children crossing										|
| Bumpy road					| Bumpy road											|
| Stop	      		| Stop				 				|
| Road work			| Road work   							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 99.8. Seems doing a good job here.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image,

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999635       			| Speed limit (30km/h) 									| 
| 0.00032128    				| Speed limit (50km/h)									|
| 2.57218e-05					| Yield											|
| 1.74703e-05  			| Keep right					 				|
| 3.65571e-08			    | End of no passing     							|


For the second image,

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.983319    			| Children crossing									| 
| 0.00715813   				| No vehicles									|
| 0.0038923				| No passing											|
| 0.0027081 			| Ahead only				 				|
| 0.00264168		    | Speed limit (60km/h)   							|

For the third image,

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999833      			| Bumpy road									| 
| 0.000167271   				| Traffic signals									|
| 1.67726e-07				|Road work										|
| 1.53453e-11			| Bicycles crossing				 				|
| 4.34831e-12		    | Road narrows on the right 							|

For the fourth image,

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999997    			| Stop 									| 
| 1.85893e-06   				| Speed limit (20km/h)									|
| 1.42602e-06				| Speed limit (30km/h)											|
| 1.44578e-07 			| Bicycles crossing					 				|
| 1.48027e-08		    | Speed limit (80km/h)     							|

For the fifth image,

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0      			| Road work 									| 
| 1.23617e-15    				| Ahead only									|
| 3.5244e-17				| Beware of ice/snow 										|
| 5.74308e-18 			| Double curve 				 				|
| 8.78056e-20		    | Speed limit (50km/h)     							|
