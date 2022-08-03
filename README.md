# HY543 ask3 , Nikolaos Gounakis , csdp1254

## 1

We build a regression model that can predict the quality of a wine from 1 to 10 given the following features:

* `fixed acidity`
* `volatile acidity`
* `citric acid`
* `residual sugar`
* `chlorides`
* `free sulfur dioxide`
* `total sulfur dioxide`
* `density`
* `pH`
* `sulphates`
* `alcohol`

## 2

We used tensorflow and build a deep neural network that consists of:

* normalizer layer
* dense layer 64 , activation relu
* dense layer 64 , activation relu
* dense layer 1

## 3

We used python , so using a package manager like pip it is easy to install tensorflow

## 4

The hard part was to study a bit about regression and how to do it on tensorflow by building a neural net.

## 5

We think the most intuitive thing was the building of the neural network using the sequential model. It was very straight forward.

## 6

The API must be very similar among different programming languages that support tensorflow.

## 7

The only thing that we would change is the fact that it supports only NVidia GPU. It would be nice to create some universal API for all GPUs so for example a pc with 2 GPUs (GPU of CPU + External GPU) could make use of both or a pc with any other GPU instead of Nvidia could use tensorflow and speed up computation time.

## 8

We learned a bit, how to use tensorflow to build a neural network regressor and overall how to train and evaluate it.

## 9

The most surprising it was that the laptop we used supports CUDA so we could speed up computation time
