Project: Traffic sign classification with CNN
=============================================

## Objective
The goalof this project is to develop an CNN-model for traffic sign classification. 

## Files
>* [main.py](https://github.com/DuseobSong/Traffic_sign_classification/blob/master/main.py): main code for model generation and training
>* [clsses.py](https://github.com/DuseobSong/Traffic_sign_classification/blob/master/classes.py): contains following classes
>> * class ***Data***      : data preprocessing and train/validation data split
>> * class ***Classifier***: generation, training and prediction functions of CNN-model
>* [predict.py](https://github.com/DuseobSong/Traffic_sign_classification/blob/master/predict.py): trained model classifies the traffic sign images and calculate accuracies for each traffic sign

## Development Environment
> OS: Windows 10 (x64), Python 3.7, OpenCV 4.2.0, Keras (backend: Tensorflow 2.2.0)
>   Dataset: Dataset: GTSRB – German Traffic Sign Recognition Benchmark [benchmark.ini.rub.de](benchmark.ini.rub.de)

## Structure
>
>![image](https://github.com/DuseobSong/Traffic_sign_classification/blob/master/images/flow_chart.png)
> 
> ###***Step 1. Training***
>> 1.1 Data preprocessing
>> In order to apply traffic sign images to CNN, all images are resized into (32x32). Hier, we apply CLAHE(Contrast Limited Adaptive Histogram Equalization) to each image, in order to enhance image contrast.
>> True-labels of each images can be found in [Train.csv](https://github.com/DuseobSong/Traffic_sign_classification/blob/master/dataset/Train.csv) and [Test.csv](https://github.com/DuseobSong/Traffic_sign_classification/blob/master/dataset/Test.csv) files in dataset folder. 
>>If there's no pretrained model, 
>![image](https://github.com/DuseobSong/Traffic_sign_classification/blob/master/images/model.png)
>
> The structure of the CNN-model is shown in the figure above. 

