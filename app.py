# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 11:21:52 2020

@author: putlu
"""
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import os
from CNN import *

plastic=os.path.abspath('plastic')
onlyfiles = [ f for f in listdir(plastic) if isfile(join(plastic,f)) ]
non_plastic=os.path.abspath('non_plastic')
onlyfiles1 = [ f for f in listdir(non_plastic) if isfile(join(non_plastic,f)) ]
train_x = np.zeros((len(onlyfiles)+len(onlyfiles1),384,512,3), dtype=np.uint8)
train_labels = []
for n in range(0, len(onlyfiles)):
    train_x[n] = cv2.imread( join(plastic,onlyfiles[n]),1 )
    train_labels.append(1)

for n in range(0, len(onlyfiles1)):
    train_x[len(onlyfiles)+n] = cv2.imread( join(non_plastic,onlyfiles1[n]),1 )
    train_labels.append(0)

train_labels=[train_labels]
train_y=np.asarray(train_labels)
#print(train_labels_array.shape)


plastic_test=os.path.abspath('plastic_test')
onlyfiles_test = [ f for f in listdir(plastic_test) if isfile(join(plastic_test,f)) ]
non_plastic_test=os.path.abspath('non_plastic_test')
onlyfiles_test1 = [ f for f in listdir(non_plastic_test) if isfile(join(non_plastic_test,f)) ]
test_x = np.zeros((len(onlyfiles_test)+len(onlyfiles_test1),384,512,3), dtype=np.uint8)
test_labels = []
for n in range(0, len(onlyfiles_test)):
    test_x[n] = cv2.imread( join(plastic_test,onlyfiles_test[n]),1 )
    test_labels.append(1)

for n in range(0, len(onlyfiles_test1)):
    test_x[len(onlyfiles_test)+n] = cv2.imread( join(non_plastic_test,onlyfiles_test1[n]),1 )
    test_labels.append(0)

test_labels=[test_labels]
test_y=np.asarray(test_labels)
#print(train_labels_array.shape)


conv1,conv2,Final=model(train_x,train_y,0.0001,15)
p=predict(train_x, train_y, conv1, conv2, Final)

'''end'''