#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals of the project were to:
* Load a dataset of images that of german traffic signs 
* Explore, summarize and visualize the data set loaded
* Preprocess dataset to improve the potential results of the architecture chosen
* Design, train and test an advanced Convolutional Neural Network model architecture capable of recognizing traffic signs with 95% accuracy
* Use the model to make predictions on new images found on the web
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./write-up-images/im1.jpg "Sample Traffic Sign Classes"
[image2]: ./write-up-images/im2.jpg "Initial distribution of training/validation images per class"
[image3]: ./write-up-images/im3.jpg "Original vs Normalized images"
[image4]: ./write-up-images/im4.jpg "Training/Validation Loss & Learning curves"
[image5]: ./write-up-images/im5.jpg "Web Images downloaded from the web and predictions"


###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First I show a sample of an image for each of the 43 classes in the dataset

![alt text][image1]

Then I show a histogram with the number of images that each class has in the training set and validation set. The distribution of the sample images across the 
different classes is important as it may affect the overall performance of the model if it was very unbalanced.

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step to preprocess the date, I normalized the input images in order for the data to have zero mean and equal variance across the training/testing/validation images.By having the same range of values on the inputs to our CNN network we can improve its ability to converge.

Here is an example of a traffic sign image before and after normalizing.

![alt text][image3]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| Activation			| RELU, output 28x28x6							|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 10x10x16 	|
| Activation			| RELU, output 10x10x16							|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16  		     		|
| Flatten               | Output 400     								|
| Fully connected		| Output 120  									|
| Activation			| RELU, output 120   							|
| Dropout   			| keep_prob=0.5, output 120     				|
| Fully connected		| output 84  									|
| Activation			| RELU, output 84   							|
| Fully connected		| Output 43 (clases)							|



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with the following parameters:

* learning rate = 0.0004
* EPOCHS = 50
* Batch Size = 200

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.939
* test set accuracy of 0.938

* To obtained the 93% minimum accuracy required for this project I started by testing the performance of the LeNet architecture on this particular problem. To do so, I tuned the learning rates, epochs and batch sizes until I achieved a validation accuracy of about 92%.

* While LeNet CNN architecture performed really well, it did not meet the criteria for the project. Therefore, to understand the reason for the lower than desired accuracy, I plotted the loss curves and the learning curves for the training and validation datasets. From the learning curve I concluded that the CNN was overfitting since the training accuracy was close to 100% but the validation accuracy was not up to par at 92%. To solve the overfitting issue I decided to implement a dropout layer. 

* After training the new model with the dropout layer and a keep_prob of 0.5, I was able to achieve a validation accuracy of 93.9%


![alt text][image4]

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web on their normalized and original versions, and with the classes as identified by the CNN:

![alt text][image5] 

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h	    		| End of all speed and passing limits   								| 
| General Caution  		| General Caution 								|
| Pedestrians			| Right of way at the next intersection			|
| Turn Right Ahead  	| Turn left ahead					 				|
| 139 km/h				| No passing		      							|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This compares negatively to the accuracy on the test set of 93.9%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 220th cell of the Ipython notebook.

Based on reviewing the top 5 Softmax probabilities for each of the 5 predictions we can conclude that the neural network has a high degree of certainty on its predictions.
This certainty however is quite incorrect. Nevertheless, there is hope for this model, as for 4 of the 5 images, the correct class was indeed one of the top 5 Softmax 
probabilities. In order to improve the model, we would have to run tests with new images from each class and calculate the precision and recall for each class. These would allow us
to better understand where to augment data to improve the overall robustness of our model.

Image 1 Actual: Speed limit (30km/h)
Probability: 0.743748 - Class: End of all speed and passing limits
Probability: 0.232051 - Class: Priority road
Probability: 0.0241936 - Class: End of speed limit (80km/h)
Probability: 5.97388e-06 - Class: Keep right
Probability: 7.01638e-07 - Class: Speed limit (30km/h)

Image 2 Actual: General caution
Probability: 1.0 - Class: General caution
Probability: 2.32786e-10 - Class: Pedestrians
Probability: 7.20546e-12 - Class: Traffic signals
Probability: 2.35197e-16 - Class: Right-of-way at the next intersection
Probability: 9.72756e-23 - Class: Road narrows on the right

Image 3 Actual: Pedestrians
Probability: 0.994601 - Class: Right-of-way at the next intersection
Probability: 0.00539525 - Class: Pedestrians
Probability: 2.73509e-06 - Class: Double curve
Probability: 3.23267e-07 - Class: Beware of ice/snow
Probability: 2.54457e-07 - Class: Dangerous curve to the left

Image 4 Actual: Turn right ahead
Probability: 0.998838 - Class: Turn left ahead
Probability: 0.00112584 - Class: Turn right ahead
Probability: 2.87467e-05 - Class: Children crossing
Probability: 6.85167e-06 - Class: Ahead only
Probability: 1.51234e-07 - Class: Slippery road

Image 5 Actual: 130 km/h (class does not exist in training set so it would be impossible to predict.)
Probability: 0.943336 - Class: No passing
Probability: 0.0528138 - Class: Speed limit (60km/h)
Probability: 0.00384966 - Class: Vehicles over 3.5 metric tons prohibited
Probability: 1.01153e-07 - Class: Speed limit (80km/h)
Probability: 9.28379e-08 - Class: End of all speed and passing limits

