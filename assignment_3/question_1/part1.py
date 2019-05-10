from datetime import datetime
from keras import optimizers
from keras import regularizers
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, GlobalMaxPooling2D
from keras.models import Model, load_model
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf

# %matplotlib inline

# command line argument
usage = 'Usage: python3 part1.py [options]\nOptions:\n\tpredict\n\tdump_iou'
arg = sys.argv[1:]
if(len(arg) < 1):
  print(usage)
  exit(1)
elif(arg[0] != 'predict' and arg[0] != 'dump_iou'):
  print(usage)
  exit(1)

data_dir = '.'
x_data = []
y_data = []
x_orignal_dim =[]
dim = 224, 224
temp = ''
file = ''
i=0

# Reading test data
directory = data_dir + '/test'
print('Reading from file...', end = '')
print(directory + '/groundtruth.txt')
with open(directory + '/groundtruth.txt', 'r') as openfileobject:
    for line in openfileobject:
        file = directory + '/' + line.split(',')[0]
        if( os.path.isfile(file) ):
            img = cv2.imread(file,1)
            try:
                x_orignal_dim.append(img.shape)
                x_data.append(cv2.resize(img, dim, interpolation = cv2.INTER_AREA))
                y_data.append(line.split(','))
            except:
                pass
    print('OK')

x_orignal_dim = np.asarray(x_orignal_dim)
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)

input_shape = (224, 224, 3)

y_data_class  = []
y_data_coords = []
factor = []
i = 0
x1 = 0.0
y1 = 0.0
x2 = 0.0
y2 = 0.0
y_data_len = y_data.shape[0]
for i in range(0, y_data_len):
    if (y_data[i][5] == 'Palm\n'):
        y_data_class.append(0)
    elif (y_data[i][5] == 'veins\n'):
        y_data_class.append(1)
    else: y_data_class.append(2)

    # Normalize Coordinates of bounding Box  
    x1 = (y_data[i][1].astype(float)/x_orignal_dim[i][1].astype(float))
    x2 = (y_data[i][3].astype(float)/x_orignal_dim[i][1].astype(float))
    y1 = (y_data[i][2].astype(float)/x_orignal_dim[i][0].astype(float))
    y2 = (y_data[i][4].astype(float)/x_orignal_dim[i][0].astype(float))
    factor.append(x1 + x2 + y1 +y2)
    y_data_coords.append((x1/factor[i], y1/factor[i], x2/factor[i], y2/factor[i]))

# Model
# Contains ResNet50 to get convolution feature maps  
# Global max pooling  
# heads:  
# ```
# * Classification
# * Regression
# ```

# ResNet50 for creating feature maps
model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling='None')
inp = model.inputs
x = model.output
out1 = Flatten()(x)
out1 = Dense(64, activation='relu')(out1)
out1 = Dense(64, activation='relu')(out1)
out1 = Dense(3, activation='softmax', name='classification')(out1)
out2 = GlobalMaxPooling2D()(x)
out2 = Dense(4, activation='softmax', name='localisation')(out2)
model = Model(inputs = inp, outputs = [out1, out2])

model.summary()

losses = {'classification': 'sparse_categorical_crossentropy', 'localisation': 'logcosh'}
l_weights = {'classification': .5, 'localisation': 2}

# load weights
model.load_weights(data_dir + "/output/weights.best.hdf5")
# Compile model (required to make predictions)
model.compile(optimizer='adam', loss=losses, loss_weights=l_weights, metrics=['accuracy'])
print("Created model and loaded weights from file")

# Bounding Box And IOU
def bb_intersection_over_union(boxA, boxB):
	
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	if((xB - xA)==0 or (yB - yA)==0):
		iou = 0.0
		return iou
	interArea =  max(0, xB - xA + 1) * max(0, yB - yA + 1)                                                                                                                                                                                                                                                             
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

# Print iou to file
bb_iou = []
for i in range(0, len(y_data_class)):
  result = model.predict(x_data[i].reshape(1, 224, 224, 3), verbose=1)
  bounding_box_normalised = (result[1]*factor[i])[0] # Normalised
  x1 = bounding_box_normalised[0] * x_orignal_dim[i][1]
  y1 = bounding_box_normalised[1] * x_orignal_dim[i][0]
  x2 = bounding_box_normalised[2] * x_orignal_dim[i][1]
  y2 = bounding_box_normalised[3] * x_orignal_dim[i][0]
  bounding_box = [x1.astype(int), y1.astype(int), x2.astype(int), y2.astype(int)]

  # Orignal coordinates
  x1 = y_data[i][1].astype(int)
  y1 = y_data[i][2].astype(int)
  x2 = y_data[i][3].astype(int)
  y2 = y_data[i][4].astype(int)
  orignal_bb = [x1.astype(int), y1.astype(int), x2.astype(int), y2.astype(int)]

  
  if(arg[0] == 'predict'):
    print('Predicted: ', bounding_box)
    print('Orignal: ', orignal_bb)
  if(arg[0] == 'dump_iou'):
    t = bb_intersection_over_union(bounding_box, orignal_bb)
    bb_iou.append(t)
    print('IOU: ', t)
  x = x_data[i]
  dim = x_orignal_dim[i][1], x_orignal_dim[i][0]
  img = cv2.resize(x, dim, interpolation = cv2.INTER_AREA)
  
if(arg[0] == 'dump_iou' or arg[1] == 'dump_iou'):
  bb_iou = np.asarray(bb_iou)
  file = open(data_dir + "/output/iou.txt","w") 
  for i in range(0, bb_iou.shape[0]):
    file.write('%f\n' % bb_iou[i]) 
  file.close()

