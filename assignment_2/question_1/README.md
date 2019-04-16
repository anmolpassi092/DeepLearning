# Foundations of Convolutional Neural Networks

This problem of assignment 2 helps in learning the basics principles of desiginind deep convolutions neural networks for image classification.

## Problem Statement
* Given architechture of a Convolutional Neural Network make the model using higher level api (Keras is used in this assignment) to classify MNIST and LINE dataset into 10 and 96 classes respectively.
* Changing hyperparmeters make another model to get the greater accurace than model given.
* Plot confusion matrices and calculate F-scores.

#### Line Dataset:

* Length - 7 (short) and 15 (long) pixels.
* Width - 1 (thin) and 3 (thick) pixels.
* Angle with X-axis - Angle θ ∈ [0
, 180
) at intervals of 15
.
* Color - Red and Blue.

#### Given architecture of CNN

* 7x7 Convolutional Layer with 32 filters and stride of 1.
* ReLU Activation Layer.
* Batch Normalization Layer
* 2x2 Max Pooling layer with a stride of 2
* fully connected layer with 1024 output units.
* ReLU Activation Layer.

## Requirements
The code is tested on Ubuntu 18.04 with:
* numpy: (1.16.2)
* pillow: (5.1.0)
* scipy: (1.2.1)
* pickel

## Running and understanding of code

Importing all required modules:

```python
# Used keras, tensorflow, matplotlib, pickle, sklearn
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

```
Load MNIST dataset using keras and LINE dataset using library function respective to the format in which it is saved. In this assignment it is saved in '.pickle' format.

Below code is for normalizing the RBG values:

```python
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
        
```

#### Model-1 (Given)
Total parameters for this model are around 6,520,000.
```python

# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(7,7), strides=(1,1), activation=tf.nn.relu , input_shape=input_shape, padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(1024, activation=tf.nn.relu))
```
Summary For Line Dataset.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 28, 28, 32)        4736      
_________________________________________________________________
batch_normalization_1 (Batch (None, 28, 28, 32)        128       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6272)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              6423552   
_________________________________________________________________
dense_2 (Dense)              (None, 97)                99425     
=================================================================
Total params: 6,527,841
Trainable params: 6,527,777
Non-trainable params: 64
_________________________________________________________________
```

#### Model-2
Total parameters for this model are around 50,000 for MNIST dataset and 83,000 for line dataset . This model is deeper as compared to the previous network with 1x1 convolutions layer to decrease the number of parameters. Also Dense layer has less neurons than 1024. It also uses dropouts for regularization.

```python

# Creating a Sequential Model and adding the layers
model = Sequential()

model.add(Conv2D(64, kernel_size=(5,5), strides=(1,1), activation=tf.nn.relu , input_shape=input_shape, padding='same'))
model.add(Conv2D(8, kernel_size=(1,1), strides=(1,1), activation=tf.nn.relu , input_shape=input_shape, padding='same'))
model.add(Conv2D(32, kernel_size=(1,1), strides=(1,1), activation=tf.nn.relu , input_shape=input_shape, padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), activation=tf.nn.relu , input_shape=input_shape, padding='same'))
model.add(Conv2D(8, kernel_size=(1,1), strides=(1,1), activation=tf.nn.relu , input_shape=input_shape, padding='same'))
model.add(Conv2D(32, kernel_size=(1,1), strides=(1,1), activation=tf.nn.relu , input_shape=input_shape, padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(2,2), strides=(1,1), activation=tf.nn.relu , input_shape=input_shape, padding='same'))
model.add(Conv2D(8, kernel_size=(1,1), strides=(1,1), activation=tf.nn.relu , input_shape=input_shape, padding='same'))
model.add(Conv2D(32, kernel_size=(1,1), strides=(1,1), activation=tf.nn.relu , input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(64, activation=tf.nn.relu))
model.add(Dense(10,activation=tf.nn.softmax))
```
Summary For Line Dataset.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 28, 28, 64)        4864      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 8)         520       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 28, 28, 32)        288       
_________________________________________________________________
batch_normalization_1 (Batch (None, 28, 28, 32)        128       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 14, 14, 64)        18496     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 14, 14, 8)         520       
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 14, 14, 32)        288       
_________________________________________________________________
batch_normalization_2 (Batch (None, 14, 14, 32)        128       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 7, 7, 32)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 7, 7, 32)          0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 7, 7, 64)          8256      
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 7, 7, 8)           520       
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 7, 7, 32)          288       
_________________________________________________________________
batch_normalization_3 (Batch (None, 7, 7, 32)          128       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 3, 3, 32)          0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 3, 3, 32)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 288)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               36992     
_________________________________________________________________
dense_2 (Dense)              (None, 97)                12513     
=================================================================
Total params: 83,929
Trainable params: 83,737
Non-trainable params: 192
_________________________________________________________________

```
## Models and Line Dataset
Models and Line image dataset sizes are very large to upload on github so click the link for [Models](https://drive.google.com/open?id=1Qcmj_0aphWdrUaNpPP5RBPKkXUijPfgE) and [Line dataset](https://drive.google.com/open?id=1xhZ2JvRfknUbMgYd2blr_b8SkJcozC6O).
