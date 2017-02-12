**Traffic Sign Recognition** 
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

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/architecture.png "Architecture"
[image3]: ./new-images/1.png "Traffic Sign 1"
[image4]: ./new-images/2.png "Traffic Sign 2"
[image5]: ./new-images/3.png "Traffic Sign 3"
[image6]: ./new-images/4.png "Traffic Sign 4"
[image7]: ./new-images/5.png "Traffic Sign 5"

--

You're reading it! and here is a link to my [project code](https://github.com/Aeonstrife/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

The code for this step is contained in the second code cell of the IPython notebook.  

* The size of training set is ?  
Number of training examples = 34799  
* The size of test set is ?  
Number of testing examples = 12630
* The shape of a traffic sign image is ?  
Image data shape = (32, 32, 3)
* The number of unique classes/labels in the data set is ?  
Number of classes = 43

Also present in the dataset was a validation set (which I would later use).

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and fourth code cells of the IPython notebook.  
A random image from the training set is displayed followed by a histogram depicting the distribution of training data across the 43 classes (of traffic signals).
I used numpy for the histogram and matplotlib.pyplot libraries for the image dislay.
Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.  

As a first step, I decided to convert the images to grayscale because dealing with 1 channel only would decrease the time taken for my network to train. I used a numpy sum of the different color channels divided by 3 to convert the images to grayscale. This is not exactly the best way to do it, but it got the job done.  

As a last step, I normalized the data so that it was spread out around 0 to avoid dealing with extremely high or extremely low values.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

I used all the examples from the train.p dataset for training my network. I used the valid.p dataset to validate my network.
My idea was to generate additional data for the classes which had lesser number of examples (as seen from the histogram earlier) by getting some images from the internet or augmenting existing images, but initial results without the extra data looked promising.

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images respectively.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

![alt text][image2]

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Preprocessing (normalization + grayscale) | 32x32x1 grayscale image |  
| Convolution 5x5     	| valid padding, outputs 28x28x16 	|  
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|  
| RELU					|												|
| Convolution 5x5     	| valid padding, outputs 10x10x36 	|  
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|  
| RELU					|												|
| Flattening layer   | output 1x900     |  
| Fully connected		|  output 128       									|  
| RELU					|												|  
| Fully connected		|  output 43						|  
| Softmax				|         									|  
  
Note: tf.one_hot was causing python to crash when using tensorflow-gpu. So, had to implement a workaround.  

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the ninth cell of the ipython notebook. 
To train the model, I used an Adam optimizer with a modest learning rate of 0.001.  
I chose to use the Adam optimizer as a research on the type of optimizer suggested that Adam works well in practice and compares favorably to other adaptive learning-method algorithms.  
  
My batch size was 100 and number of epochs was 50. I initially tried with higher and lesser number of epochs and decided on 50 as a good compromise (achieved around 90%~ accuracy on the validation set). Other hyper-parameters which I used were mu and sigma (values 0 and 0.1 respectively) to introduce slight variance in the weights between training cycles.  

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?  
I didnot check my accuracy on the test dataset.  
* validation set accuracy of ?  
The network was able to achieve a Validation Accuracy of 96%
* test set accuracy of ?  
The network was able to achieve a Test Set Accuracy of 95.3%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?  
I initially tried the iterative approach (with no epochs) but was not able to achieve validation set accuracies beyond 80%~.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?  
I adapted the LeNet architecture with some changes to the number of layers and features. I also used some elements from the Hvass-labs implementation of a [convolutional neural network](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb). Since in both cases (LeNet and Hvass-Labs) the problem to be tackled is classification of images of a similar nature to traffic signs, I assumed that the architecture of the network would still be relevant (albeit with some changes to the feature sizes).  
The final accuracy levels which I got on the validation and test sets (90%+) suggested that the model is working well.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]

The images seemed pretty straightforward to classify (atleast in my eyes :P).  
The second image is similar to other traffic signs which might create problems.  
The last image is at an angle and is not very discernable from the background.  

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the eleventh cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)     		| Speed limit (30km/h) 									| 
| Road work   			| Road work									|
| Priority road					| Priority road										|
| Right-of-way at the next intersection      		| Right-of-way at the next intersection		 				|
| Turn left ahead			| Turn left ahead    							|

The model was able to classify all the images correctly.  

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located from the 11th cell of the Ipython notebook.

From the softmax proabilities, we can see that for all the 5 images, the model is pretty sure of predicting the correct sign. In fact, it is 100% for 4 out of the 5 images.  
For 1 image (the Road work sign), the probability is 98%. From the other probability of 2$ of the sign to which it can be confused with, we can see that it is similar (if we squint and turn our head sideways :P) to the correct sign.  
From the visualization, we can see that for most of the cases, the system correctly classifies even the new images with high confidence on the correct sign.
