
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
get_ipython().magic('matplotlib inline')


# In[2]:

filein = pd.read_csv('./data/both.csv')


# In[3]:

from os import listdir

paths_forward = []
parent_forward = 'data/IMG/'
filenames_forward = listdir('data/IMG/')


# In[4]:

paths_backward = []
parent_backward = 'data/data_backward/IMG/'
filenames_backward = listdir('data/data_backward/IMG/')


# In[5]:

for file in filenames_forward:
    paths_forward.append(parent_forward + file)


# In[6]:

for file in filenames_backward:
    paths_backward.append(parent_backward + file)


# In[7]:

paths = paths_forward + paths_backward


# In[8]:

left = []
for each in paths:
    if 'left' in each.split('/')[-1]:
        left.append(each)


# In[9]:

center = []
for each in paths:
    if 'center' in each.split('/')[-1]:
        center.append(each)


# In[10]:

right = []
for each in paths:
    if 'right' in each.split('/')[-1]:
        right.append(each)


# In[11]:

steering = np.array(filein['STEERING'])


# In[12]:

images = []
y_train = []
for left_img, center_img, right_img, label in zip(left, center, right, steering):
    cen = plt.imread(center_img)
#     img = img[60:126,60:260,:]
#     Choosing only half of the straight values randomly
    if (label == 0.0) and (np.random.randint(1,100) % 2 == 0):
        images.append(cen)
        y_train.append(label)
        images.append(cv2.flip(cen,1))
        y_train.append(-label)
    elif (label != 0.0):
        images.append(cen)
        y_train.append(label)
        images.append(cv2.flip(cen,1))
        y_train.append(-label)
        
    lhs = plt.imread(left_img)
    images.append(lhs)
    y_train.append(label + 0.3)
    
    rhs = plt.imread(right_img)
    images.append(rhs)
    y_train.append(label - 0.3)


# In[13]:

X_train = np.array(images)
y_train = np.array(y_train)


# In[14]:

X_train.shape


# In[15]:

y_train.shape


# In[16]:

from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv2D, Flatten, Lambda, Cropping2D, MaxPooling2D
from keras import regularizers
from keras.optimizers import Adam


# In[17]:

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)


# In[18]:

model = Sequential()
model.add(Cropping2D(cropping = ((60, 20),(0,0)), input_shape = (160, 320, 3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Conv2D(24, (5, 5), strides = (2,2), activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
# model.add(MaxPooling2D())
model.add(Conv2D(36, (5, 5), strides = (2,2), activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
model.add(Conv2D(48, (5, 5), strides = (2,2), activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
model.add(MaxPooling2D(data_format = "channels_last"))
model.add(Flatten())
model.add(Dense(100, kernel_regularizer = regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(50, kernel_regularizer = regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(10, kernel_regularizer = regularizers.l2(0.01)))
model.add(Dense(1))


# In[19]:

model.compile(loss = 'mse', optimizer = adam)
model.fit(X_train, y_train, batch_size = 64, validation_split = 0.2, shuffle = True, epochs = 20, verbose = 1)
model.save('models/new_nvidia_w_dropout.h5')
model.summary()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



