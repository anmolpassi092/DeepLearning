# Multi-Head Classification
The objective of this problem is to design non-sequential networks.

## Problem Statement
Design a non-sequential convolutional neural network for classifying the line dataset. This
network will have 4 outputs based on the 4 kind of variations(length, width, color, angle).
You are required to divide your network architecture into two parts a) Feature network and
b) Classification heads. The feature network will be responsible for extracting the required
features from the input and attached to it would be the four classification heads one for each
variation. The network is shown in figure 1.
The first 3 classification heads are for 2 class problems namely length, width and color
classification. In all these the final layer contains a single neuron with a sigmoid activation
followed by binary crossentropy loss.
The last classification head is a 12 class problem for each 12 angles of variation. In this the
final layer contains 12 neurons with softmax activation and Categorical Cross entropy loss


## Requirements

* keras
* python


## Running and understanding of code

The feature map consisting of two convulation layers.

```python
#feature Map
conv11 = Conv2D(256, kernel_size=4, activation='relu')(input_layer)
pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
conv12 = Conv2D(256, kernel_size=4, activation='relu')(pool11)
pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)



```

4 classification heads for different classification.

```python
#for classification of length
flat1a = Flatten()(pool12)
dense1a=Dense(64)(flat1a)
dense2a=Dense(1, activation='sigmoid')(dense1a)

#for classification of width
flat1b = Flatten()(pool12)
dense1b=Dense(64)(flat1b)
dense2b=Dense(1, activation='sigmoid')(dense1b)

#for classification of angle
flat1c = Flatten()(pool12)
dense1c=Dense(64)(flat1c)
dense2c=Dense(12, activation='softmax')(dense1c)

#for classification of color
flat1d = Flatten()(pool12)
dense1d=Dense(64)(flat1d)
dense2d=Dense(1, activation='sigmoid')(dense1d)
```
