'''
Traffic sign classification
training network

'''
import os, sys
import numpy as np
import cv2

from classes import *
from sklearn.externals import joblib

# if __name__ == '__main__':
    
data = Data()

net = Classifier()

print('## Step 1. data preprocessing... ##')
train_data = data.preprocessing(csv_name = 'train.csv')

train_data, valid_data = data.data_split(data = train_data)

print('## Step 2. Creating ANN...  ##')
net.n_classes = len(open(net.metadata_path).read().strip().split('\n')[1:])
net.class_weight = data.class_weight
model = net.build()

print('## Step 3. Training ANN...  ##')
history, trained_model = net.train(model = model, train_data = train_data,
                                    valid_data = valid_data)
joblib.dump(history, './model/training_result.pkl')

print('## Network training completed ##')