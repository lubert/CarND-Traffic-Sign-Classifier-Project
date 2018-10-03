# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[vis]: ./distribution.jpg "Visualization"
[gray]: ./grayscale.jpg "Grayscaling"
[aug]: ./augment.jpg "Augmenting"
[vis2]: ./distribution2.jpg "Visualization"
[test1]: ./test_signs/70.png "Traffic Sign 1"
[test2]: ./test_signs/100.png "Traffic Sign 2"
[test3]: ./test_signs/dig.png "Traffic Sign 3"
[test4]: ./test_signs/no.png "Traffic Sign 4"
[test5]: ./test_signs/right.png "Traffic Sign 5"
[topk]: ./topk.png "Top k"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Done

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the classes.

![alt text][vis]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I converted the images to grayscale for three reasons. First, it reduces the size of the input and hence the training time required. Second, using grayscale images allowed me to use the LeNet architecture essentially as-is. Finally, the Sermanet baseline paper on Traffic Sign Recognition experimented with grayscale images and had good results.

![alt text][gray]

After converting the image to grayscale, I normalized the image data, because from what I understand, although it's not strictly necessary, normalizing usually decreases the training time and the chance for getting stuck in local optima.

The first time I trained the model, the prediction accuracy was fairly low and hovered around 91%. Looking at the distribution of the training data, certain classes are underrepresented, so I used a combination of random scaling, translation, and brightness adjustments to generate more training examples for the classes with fewer examples. I used numpy and OpenCV functions, which were pretty slow to run.

![alt text][aug]

After augmentation, the number of training examples increased to 46480

![alt text][vis2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16 					|
|   -> Flatten			| Outputs 400				 					|
|   -> Convolution 4x4	| 1x1 stride, same padding, outputs 2x2x100 	|
|      RELU				|												|
| Merge					| Outputs 800									|
| Dropout				| Keep percentage = 0.5							|
| Fully Connected		| Outputs 43									|
| Softmax				|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the Tensorflow AdamOptimizer, and for the hyperparameters, I used 60 epochs, a batch size of 100, and a learning rate of 0.001. While experimenting with different hyperparameters, using a lower learning rate seemed to have the biggest effect in the training accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.956
* test set accuracy of 0.945

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I first tried the LeNet architecture, since I was familiar with it from the lab assignments. The original architecture was designed for character recognition, which is a similar problem space to traffic sign identification. However, the accuracy on the validation set wasn't particularly high when adapting the last layer to output 43 classes, but it was a good baseline for getting something to work end-to-end. 

After reading through the Sermanet architecture referenced in the project instructions, I tried implementing it by modifying the LeNet architecture, and dubbed it "Sortanet." The modified architecture actually performed fairly poorly than the LeNet architecture with the same hyperparameters. Initially I tried adding a dropout after each layer, which did improve the accuracy but increased the training time. I eventually settled on one dropout layer right before the last fully connected layer, which was a reasonable balance of accuracy and training time.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][test1] ![alt text][test2] 

The first couple of test images that I picked are both speed limit signs, and the similarity in shape among all of the speed limit signs might be difficuly to classify.

![alt text][test4] 

I picked a round "no vehicles" sign, which might be difficult to distinguish from the other speed limit signs. 

![alt text][test3] 

The right turn sign could be difficult to distinguish from other signs with dark backgrounds, especially since I am converting the test images to grayscale.

![alt text][test5]

The road work sign might be difficult to distinguish from other triangular signs with symbols.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100km/h     		| 60km/h   									| 
| 70km/h     			| 70km/h 										|
| Road work					| Road work											|
| No	vehicles      		| No vehicles					 				|
| Turn right			| Turn right      							|


The model correctly guessed 4 of the 5 traffic signs for an accuracy of 80%. This is reasonable performance compared to the test set accuracy of 94.5%. As expected, the model had some difficulty correctly classifying the speed limit signs. I think given this, to further improve the accuracy, I might consider augmenting the speed limit examples. Additionally, I would consider adapting the architecture to support color because I think a good amount of information is lost in the grayscale conversion

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][topk]

The model was absolutely certain of its guesses, which makes me think that I either calculated the softmax probabilities incorrectly, or that the model is somewhat overfitted. In the case of the first test sign, the model was 100% certain it was a 60km/h sign, which was incorrect.


