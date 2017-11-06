#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_imgs/after_cropping.jpg "After cropping"
[image2]: ./report_imgs/b&w.png "Black & white"
[image3]: ./report_imgs/color.png "Color"
[image4]: ./report_imgs/steering_unbalanced.jpg "Unbalanced Steering values"
[image5]: ./report_imgs/throttle_unbalanced.jpg "Unbalanced throttle values"
[image6]: ./report_imgs/model_arch.jpg "Architecture"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* script.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* report_imgs containing writeup-images

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes (just like the one used in the NVIDIA paper: https://arxiv.org/pdf/1604.07316.pdf)
Model:
![alt text][image6] 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer. I have also included a 2D MaxPool at the end of the last convolutional layer.

####2. Attempts to reduce overfitting in the model

For regularization, I have used L2 regularization on every layer to penalise the weights.
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, with a learning rate decay of 0.01 (default = 0.0)

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and I also drove in the opposite direction to help the model generalise.

I randomly chose half of the data where the steering angle was 0.0. This was done because the training data consisted mainly of straight driven data. This would have made my model biased towards always predicting to go straight.

![alt text][image4]
![alt text][image5]

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to reduce training and validation loss and increase validation accuracy.

My first step was to use a simple regression model just to test if things were working. Then I used the architecture described in the paper cited above. I thought this model might be appropriate because it already works on a SDC.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that each layer has its own l2 regularization.
Then I used maxpool after the last convolution for better results

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and to improve the driving behavior in these cases, I increased the magnitude of correction of the steering angles from the left and right camera inputs.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes
Conv2D (5, 5, 24)
Conv2D (5, 5, 36)
Conv2D (5, 5, 48)
Conv2D (3, 3, 64)
Conv2D (3, 3, 64)
MaxPool2D
Dense(100)
Dense(50)
Dense(10)
Dense(1)

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image6]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then converted these images to grayscale:

![alt text][image2]

The results were not very good.

After the collection process, I then preprocessed this data by normalizing the pixel values.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 and the best performing batch_size was 64. I used an adam optimizer so that manually training the learning rate wasn't necessary.
