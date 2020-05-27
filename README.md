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
> Dataset: Dataset: GTSRB – German Traffic Sign Recognition Benchmark [benchmark.ini.rub.de](benchmark.ini.rub.de)

## Structure
>  
>![image](https://github.com/DuseobSong/Traffic_sign_classification/blob/master/images/flow_chart.png)
>  
> ### Step 1. Training
>> ***1.1 Data preprocessing***
>> In order to apply traffic sign images to CNN, all images are resized into (32x32). Hier, we apply [CLAHE(Contrast Limited Adaptive Histogram Equalization)](http://amroamroamro.github.io/mexopencv/opencv/clahe_demo_gui.html) to each image, in order to enhance image contrast.
>> True-labels of each images can be found in [Train.csv](https://github.com/DuseobSong/Traffic_sign_classification/blob/master/dataset/Train.csv) and [Test.csv](https://github.com/DuseobSong/Traffic_sign_classification/blob/master/dataset/Test.csv) files in dataset folder. 
These modified images and their true labels are devided into train- and validattion dataset.
>  
>> ***1.2 Build CNN-model***
>>If there's no pretrained model, we can generate a CNN-model using the ***build( )*** function of ***Classifier*** class. Hyperparameters, such as number of training epochs, initial learning rate, optimizer and batch size can be customized.  
>> The structure of the CNN-model is shown in the figure below.  
>>  
>>![image](https://github.com/DuseobSong/Traffic_sign_classification/blob/master/images/model.png)
>>  
>> In order to improve prediction accuracy, we use keras built-in function ***[ImageDataGenerator](https://keras.io/api/preprocessing/image/)*** and augment the training images.
>  
>> ### Step 2. Prediction
>> Test-images are preprocessed in the same way as in the prvious step. The CNN predict the class of traffic signs and these predicted labels are compared with true labels. At the end, the accuracies for each sign and total accuracy are calculated. 
>> The total accuracy reaches 95.76% and accuricies for each sign are shown in the figure below.
>>  
>> ![image](https://github.com/DuseobSong/Traffic_sign_classification/blob/master/Accuracy/accuracy.png)

## Discussion and future works
> - The total prediction accuracy is quite satisfactory. However, the accuracies for some specific signs are under 80%. To improve these accuracies, we need to train the CNN-model with additional datasets.
> - To apply this model to other application, we need a traffic sign detection algorithm. Traffic sign detection can be realized using Deep-learning approachs(such as YOLO or mask R-CNN) or object detection algorithms(random forest with HOG or SIFT).


