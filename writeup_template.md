# **Behavioral Cloning** 

## Writeup Template


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/training_problems.png "Training Problems"
[image2]: ./examples/track1.png "Track1"
[image3]: ./examples/ptrack2.png "Track2"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* training.ipynb containing the script to create the dataset and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The training.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 5 layers of Convolutional neural networks, followed by 5 fully connected layers. The architecture is based on the [Nvidia end-to-end driving paper](https://arxiv.org/abs/1604.07316)
I increased the number of filters in the convolutional layers and added a new fully connected layer with more neurons.
I also added batch normalization in all layers to help with training.
The goal of adding more filters was creating a network that could generalize better, to be able to learn both tracks with one model. Although the generalization was good, it has the downside of adding latency to the processing.

#### 2. Attempts to reduce overfitting in the model

The model contains one dropout layer in order to reduce overfitting.

It was trained on both tracks at once, with different behaviors

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (training.ipynb line 126).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use architectures that have worked for me in the past.

My first step was to use a convolution neural network model similar to the one presented in the Nvidia paper I thought this model might be appropriate because it was suggested for the same purpose.

I realized that in this case validation error didn't necessarily translate to a good performance on the track. Still I split 20% of the dataset for validation, so that I could evaluate the model's performance.

I left the model runs over many epochs and used the checkpoint functionality to keep the ones with the lowest validation error.

I also used Nvidia's suggested technique of adding additional cameras on the sides, to train the model to recover from bad positions. The solution is not perfect and if the correction factor is not correct the model would oscillate in the lane. 

One silly mistake that harmed my model's performance was training to an incorrect colorspace. Because I was using openCV for loading the training data, the output color format was BGR, while the simulator uses mpimg and an RGB color format. 
This lead to my model engaging in weird behavior like avoiding shadows. You can watch a video of this behavior here:  
[![image1]](https://www.youtube.com/edit?o=U&video_id=WsGICYafbP8)
Once I corrected the error the model's performance improved dramatically.](


#### 2. Final Model Architecture

The final model architecture (training.ipynb lines 102-123) consisted of a convolution neural network with the following layers and layer sizes ...

*model.add(Conv2D(34,5,5,subsample=(2,2), activation="elu"))
*model.add(BatchNormalization())
*model.add(Convolution2D(46,5,5,subsample=(2,2), activation="elu"))
*model.add(BatchNormalization())
*model.add(Convolution2D(58,5,5,subsample=(2,2), activation="elu"))
*model.add(BatchNormalization())
*model.add(Convolution2D(74,3,3, activation="elu"))
*model.add(BatchNormalization())
*model.add(Convolution2D(74,3,3, activation="elu"))
*model.add(BatchNormalization())
*model.add(Dropout(0.5))
*model.add(Flatten())
*model.add(Dense(1164))
*model.add(BatchNormalization())
*model.add(Dense(200))
*model.add(BatchNormalization())
*model.add(Dense(100))
*model.add(BatchNormalization())
*model.add(Dense(20))
*model.add(BatchNormalization())
*model.add(BatchNormalization())
*model.add(Dense(1))


#### 3. Creation of the Training Set & Training Process

To capture good driving data I drove the first track several times in both directions, using the mouse as input because it is proportional unlike a keyboard.
Then, I recorded some recovery scenarios, where the car started at a course that would have taken it off the road, and then was taken back to the road correctly.
Finally I also ran several laps on the second track. Some keeping the lane and some using the entire road. A model trained only on the second track was able to keep the lane perfectly.
I did the recording at low speed on the second track to be more precise. Since the vehicle dynamics don't make much of a difference up to 20mph, it was possible for the network to drive faster than the training data was recorded at.

Here are the resulting videos for track 1 and track 2

