'''
Traffic sign classification

prediction
'''

import os, sys
import cv2
import numpy as np
from keras.models import load_model

from classes import *

model_load = True
pred_labels = []

net = Classifier()
net.load_labels()

total = 0
cor = 0 
inc = 0

if model_load == True:
    model = load_model('./model/tsc.h5')

csv_data = open('./dataset/test.csv').read().strip().split('\n')[1:]

true_labels = [row.split(',')[-2] for row in csv_data]
true_labels = np.array(true_labels).astype(np.int)
img_paths = [row.split(',')[-1] for row in csv_data]

for i in range(len(img_paths)):
    total += 1
    
    img_path = './dataset/' + img_paths[i]
    true_label = true_labels[i]
    
    ans, pred_label = net.predict(model, img_path, true_label, save_img = True)
    
    if ans == 'correct':
        cor += 1
    elif ans == 'incorrect':
        inc += 1
    
    pred_labels.append([ans, pred_label])

accuracy = cor/total * 100