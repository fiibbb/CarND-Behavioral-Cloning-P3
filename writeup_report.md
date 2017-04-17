#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:  

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The neural network model is largely based on NVIDIA's self driving car neural network model. It consists 5 convolutional layers with 5x5 or 3x3 filters (model.py L25-L29), and 3 dense (fully-connected) layers(model.py L35-L39).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py L36, L38). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py L89).

####4. Appropriate training data

The training data consists of the official dataset provided by udacity (model.py L113), and additional dataset I collected using the simulator (model.py L114-L117). The additional dataset focused on several scenarios, including center lane driving, recovery from left and right of the lane, and reverse.

###Model Architecture and Training Strategy

####Solution Design Approach

The overall strategy for deriving a model architecture was to experiment with the existing dataset, observe how well the model works when put fed to the simulator, then either tune the model or collect more data to train the model.

During my first test, I observed that the was able to stay in the lane, but was very reluctant to turn. This behavior indicated the model predicted a smaller steering angle than necessary. My intutuition was that a larger steering angle may have resulted in a higher error compared to a lower steering angle in general. Thus I swtiched from `MSE` to `MAE` in my optimzer, hoping that changing from squared error to average error could help the situation. As it turns out, this worked very well.

Then I realized the car was behaving over sensitive to the surroudings as it kept adjusting steering angle even when it was only slightly to the left or right of the lane. This was an indication of overfitting. To combat overfitting, I added two dropout layers between the fully connected layers. I also collected more data on my own using the simulator to feed into the training.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. The vehicle is also able to recover after I override autonomous mode and point it off the center. As long as the vehicle is not off the track, it's able to recover to the center lane in most cases.

####Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded couple laps on track one using center lane driving, see `data/02_keep_lane`, and because the track has a huge portion of almost straight lane, reducing the percentage of curve lane driving in the dataset, I added some clips focusing on driving smoothly through the curves, see `data/04_curve`. Also `data/05_reverse` was added because the track was mostly turning left, so some reverse laps was used to balance the percentage of left vs right turns.

It turned out the most useful dataset, however, was the `03_recover`, where I recorded some clips for the car to steer back to the center lane when it's in a unconfortable position. This can be tested out by manually overriding the car to the side of lane in the simulator, and autonomous mode would steer it back to the center.

All the datasets fed to the training process were augmented using the methodologies suggested in the lecture. I'm using all three cameras, with a `+0.1` or `-0.1` steering angle offset for the cameras on the side. I also flip every image to double the dataset I have.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the fact that marginal decrese in loss became very small after epoch 4-5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
