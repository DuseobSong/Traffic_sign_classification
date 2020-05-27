'''
Traffic sign classification

prediction
'''

import os, sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import load_model

from classes import *

SAVE_PATH = './Accuracy/'
if not os.path.exists(SAVE_PATH):
    os.makedir(SAVE_PATH)
    
model_load = True

net = Classifier()
net.load_labels()

for i in range(len(net.label_names)):
    globals()['class_{}'.format(i)] = []
    globals()['class_{}_img'.format(i)] = []
    globals()['class_{}_true'.format(i)] = []
    globals()['class_{}_pred'.format(i)] = []

total = 0
cor = 0 
inc = 0

if model_load == True:
    model = load_model('./model/tsc.h5')

csv_data = open('./dataset/test.csv').read().strip().split('\n')[1:]

true_labels = [row.split(',')[-2] for row in csv_data]
true_labels = np.array(true_labels).astype(np.int)
img_paths = [row.split(',')[-1] for row in csv_data]
img_names = [img_path.split('/')[-1] for img_path in img_paths]

for i in range(len(img_paths)):
    total += 1
    
    img_path = './dataset/' + img_paths[i]
    img_name = img_names[i]
    true_label = true_labels[i]
    
    eval('class_{}_img'.format(true_label)).append(img_name)
    
    ans, pred_label = net.predict(model, img_path, true_label, save_img = True)
    
    eval('class_{}_true'.format(true_label)).append(net.label_names[true_label])
    eval('class_{}_pred'.format(true_label)).append(pred_label)
    
    if net.label_names[true_label] == pred_label:
        eval('class_{}'.format(true_label)).append(1)
    elif net.label_names[true_label] != pred_label:
        eval('class_{}'.format(true_label)).append(0)
    
    if ans == 'correct':
        cor += 1
    elif ans == 'incorrect':
        inc += 1
accuracy = cor/total * 100    
class_accuracy = []
samples = []

for i in range(len(net.label_names)):
    tmp_pred = np.array(eval('class_'+str(i)))
    tmp_acc = np.sum(tmp_pred) / len(tmp_pred) * 100
    class_accuracy.append(tmp_acc)
    samples.append(len(tmp_pred))
    df = pd.DataFrame({'image_name': eval('class_' + str(i)),
                       'true_label': eval('class_' + str(i) + '_true'),
                       'pred_label': eval('class_' + str(i) + '_pred'),
                       })
    df.to_csv(SAVE_PATH + 'Prediction_class_' + str(i) + '.csv', index = False)
    
acc = pd.DataFrame({'Label_name': net.label_names,
                    'No_of_samples': samples, 
                    'Accuracy': class_accuracy
                    })
acc.to_csv(SAVE_PATH + 'accuracy.csv', index = False)
tags = ['No_of_test_samples', 'No_of_label_classes', 'No_of_correct_predictions', 'No_of_incorrect_predictions', 'Accuracy']
values = [len(csv_data), len(net.label_names), cor, inc, accuracy]
result = pd.DataFrame([tags, values])
result.to_csv(SAVE_PATH + 'prediction_result.csv', index = False)

class_accuracy = np.array(class_accuracy)

half = len(net.label_names) // 2

plt.figure(figsize = (32,10))
plt.subplot(1,2,1)
pos1 = np.arange(half) + 0.5
plt.barh(pos1, class_accuracy[:half], height = 0.4)
plt.xticks([20, 40, 60, 80, 100, 120], ['20', '40', '60', '80', '100', ' '], fontsize = 14)
plt.yticks(pos1, net.label_names[:half], fontsize = 15)
plt.gca().invert_yaxis()
plt.gca().xaxis.grid()

posx = 100 * 1.02
for i in range(half):
    posy = 0.5 + i
    plt.text(posx, posy, '%d samples' % samples[i], rotation = 0, ha = 'left', va = 'center', fontsize = 14)
    
plt.xlabel('Accuracy [%]', fontsize = 16)



plt.subplot(1,2,2)
pos2 = np.arange(len(net.label_names)-half) + 0.5
plt.barh(pos2, class_accuracy[half:], height = 0.4)
plt.xticks([20, 40, 60, 80, 100, 120], ['20', '40', '60', '80', '100', ' '], fontsize = 14)
plt.yticks(pos2, net.label_names[half:], fontsize = 15)
plt.gca().invert_yaxis()
plt.gca().xaxis.grid()
posx = 100 * 1.02
for i in range(len(net.label_names) - half):
    posy = 0.5 + i
    plt.text(posx, posy, '%d samples' % samples[half + i], rotation = 0, ha = 'left', va = 'center', fontsize = 14)
plt.xlabel('Accuracy [%]', fontsize = 16)

plt.subplots_adjust(wspace = 0.6)

plt.savefig(SAVE_PATH + 'accuracy.png')