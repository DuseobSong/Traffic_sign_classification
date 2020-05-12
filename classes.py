'''
Traffic sign classifier


'''

import sys, os
import pandas as pd
import matplotlib.pyplot as plt

import keras
import cv2
import numpy as np

from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import *
from keras.models import Sequential, load_model
from sklearn.metrics import classification_report

from skimage import exposure
'''
class Data

functions:
    - data_preprocessing
    - data_split: Splitting training data into training- and validation-data
'''
class Data:
    def __init__(self):
        self.root_path = './dataset/'
        self.metadata_path = './dataset/meta.csv'
        
        self.label_names = []
        
        self.class_weight = None
        
    def preprocessing(self, csv_name):
        data = []
        labels = []
        
        rows = open(self.root_path + '/' + csv_name).read().strip().split('\n')[1:]
        np.random.shuffle(rows)
        
        for i, row in enumerate(rows):
            (label, img_name) = row.strip().split(',')[-2:]
            
            if i > 0 and i%1000 == 0:
                print('Completed: {}'.format(i))
            # load image and adapt CLAHE(Contrast Limited Adaptive istogram Equalization)
            # img_path = os.path.join(self.root_path, img_name)
            img = cv2.imread(self.root_path + img_name)
            img = cv2.resize(img, (32,32))
            img = exposure.equalize_adapthist(img, clip_limit = 0.1)
            
            # CLAHE in OpenCV can be only adapted to gray-scale image
            #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            #img = cv2.resize(image, (32,32))
            #clahe = cv2.createCLAHE(cliplimit = 0.1)
            #img = clahe.apply(img)
            
            data.append(img)
            labels.append(label)
        data = np.array(data)
        labels = np.array(labels)
        label_names = open(self.metadata_path).read().strip().split('\n')[1:]
        self.label_names = [l.split(',')[-1] for l in label_names]
        trainX = data.astype('float32') / 255.
        n_labels = len(self.label_names)
        trainY = to_categorical(labels, n_labels)
        
        class_totals = trainY.sum(axis = 0)
        self.class_weight = class_totals.max() / class_totals
        
        return (trainX, trainY)
    
    def data_split(self, data, split_rate = 0.7):
        x, y = data
        split = int(len(y) * split_rate)
        train_data = (x[:split], y[:split])
        valid_data = (x[split:], y[split:])
        
        return train_data, valid_data
    
'''
class: Classifier

functions: 
    - train
    - predict

'''
class Classifier:
    def __init__(self):
        # Hyper parameters
        self.NUM_EPOCHS = 32
        self.INIT_LR = 1e-3
        self.BATCH_SIZE = 4
        self.optimizer = Adam(lr = self.INIT_LR, decay = self.INIT_LR / (self.NUM_EPOCHS * 0.5))
        
        self.img_size = (32,32,3)
        self.n_classes = None
        
        self.model_path = './model/'
    
        self.label_names = []
        self.metadata_path = './dataset/meta.csv'
        self.class_weight = None
        
    def load_labels(self):
        label_names = open(self.metadata_path).read().strip().split('\n')[1:]
        label_names = [l.split(',')[-1] for l in label_names]
        
        self.label_names = label_names
        
    def build(self):
        model = Sequential()
        
        # src image size: (32,32,3)
        model.add(Conv2D(8, (5,5), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal', input_shape = self.img_size))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2,2)))
        
        # first order feature map size: (16, 16, 3)
        model.add(Conv2D(16, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization())
        model.add(Conv2D(16, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2,2)))
        
        # second order feature map size: (8, 8, 3)
        model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2,2)))
        
        # first set of feature components
        model.add(Flatten())
        model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # second set of feature components
        model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # classification
        model.add(Dense(self.n_classes, activation = 'softmax'))
        
        model.compile(loss = 'categorical_crossentropy',
                      optimizer = self.optimizer,
                      metrics = ['accuracy'])
        
        return model
    
    def train(self, model, train_data, valid_data, load_model = False):
        if load_model == True:
            model = load_model(self.model_path + 'tsc.h5')
            
        augmentor = ImageDataGenerator(rotation_range = 10,
                                        zoom_range = 0.15,
                                        width_shift_range = 0.1,
                                        height_shift_range = 0.1,
                                        shear_range = 0.15,
                                        horizontal_flip = False,
                                        vertical_flip = False,
                                        fill_mode = 'nearest'
                                        )
        (trainX, trainY) = train_data
        (validX, validY) = valid_data
        
        history = model.fit_generator(augmentor.flow(trainX, trainY, batch_size = self.BATCH_SIZE),
                                      validation_data = (validX, validY),
                                      steps_per_epoch = trainX.shape[0] // self.BATCH_SIZE,
                                      epochs = self.NUM_EPOCHS,
                                      class_weight = self.class_weight
                                      )
        
        if not os.path.exists(self.model_path):
            os.makedir(self.model_path)
            
        model.save(self.model_path + 'tsc.h5')
        
        return history, model
        
    def predict(self, model, img_path, true_label, save_img = True):
        img = cv2.imread(img_path)
        result = img.copy()
        img = cv2.resize(img, (32,32))
        img = exposure.equalize_adapthist(img, clip_limit = 0.1)
        
        img = img.astype('float32')/255.
        img = np.expand_dims(img, axis = 0)
        
        preds = model.predict(img)
        j = preds.argmax(axis = 1)[0]
        pred_label = self.label_names[j]
        
        if j == true_label:
            ans = 'correct'
        else:
            ans = 'incorrect'
        
        if save_img == True:
            
            result = cv2.resize(result, (256,256))
            cv2.putText(result, pred_label + ' (' + ans + ')', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
            
            if not os.path.exists('./prediction/'):
                os.makedir('./prediction/')
            
            result_path = './prediction/predicted_' + img_path.split('/')[-1]
            cv2.imwrite(result_path, result)
        
        return ans, pred_label
    
    def plot_history(self, history):
        N = np.arange(0, self.NUM_EPOCHS)
        plt.figure()
        plt.plot(N, history.history['loss'], label = 'train_loss')
        plt.plot(N, history.history['val_loss'], label = 'validation_loss')
        plt.plot(N, history.history['acc'], label = 'train_acc')
        plt.plot(N, history.history['val_acc'], label = 'validation_acc')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Accuracy')
        plt.legend(loc = 'upper right')
        plt.save_fig(self.model_path + '/training_result.png')