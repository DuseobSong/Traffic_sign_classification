'''
RSC - GUI

'''

import sys, os
import numpy as np
import cv2
import pandas as pd
from skimage import exposure

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from datetime import datetime
from keras.models import load_model
from PyQt5.QtCore import *
# from keras import *
class LogHistory(QMainWindow):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setupUI()
        
        self.data = None
    
    def setupUI(self):
        self.setGeometry(200, 100, 600, 800)
        self.setWindowTitle('Log History')
        
        mainTitle = QLabel('Log History', self)
        mainTitle.setGeometry(40, 40, 520, 40)
        mainTitle.setStyleSheet('background-color: white; border: 1px solid black')
        mainTitle.setFont(QtGui.QFont('Arial', 16, QtGui.QFont.Bold))
        mainTitle.setAlignment(Qt.AlignCenter)
        
        saveLogAction = QAction('Save logs', self)
        saveLogAction.setShortcut('Ctrl+S')
        saveLogAction.triggered.connect(self.parent.save_log)
        
        exitAction = QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.triggered.connect(lambda:self.close())
        
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(saveLogAction)
        fileMenu.addAction(exitAction)
        
        
        # table
        self.table = QTableWidget(self)
        self.table.setGeometry(20, 100, 560, 640)
        self.table.setColumnCount(3)
        self.table.setRowCount(10) # default
        self.table.setHorizontalHeaderLabels(['Date','Time','Message'])
        self.table.setColumnWidth(0, 100)
        # self.table.horizontalHeaderItem(0).setTextAlignment(Qt.AlignCenter)
        self.table.setColumnWidth(1, 100)
        # self.table.horizontalHeaderItem(1).setTextAlignment(Qt.AlignCenter)
        self.table.setColumnWidth(2, 380)
        self.show_history()
        
        # Functions
        self.exitButton = QPushButton('Exit', self)
        self.exitButton.setGeometry(480, 760, 80, 20)
        self.exitButton.setStyleSheet('border: 1px solid black')
        self.exitButton.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        self.exitButton.clicked.connect(lambda:self.close())
        
        self.saveLog = QPushButton('Save Log history', self)
        self.saveLog.setGeometry(40, 760, 200, 20)
        self.saveLog.setStyleSheet('border: 1px solid black')
        self.saveLog.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        self.saveLog.clicked.connect(self.parent.save_log)
    
    def show_history(self):
        date = []
        time = []
        msg = []            
        for log in self.parent.log:
            date.append(log.split(' | ')[0])
            time.append(log.split(' | ')[1])
            msg.append(log.split(' | ')[2])
            
        self.table.setRowCount(len(date))
        print(date)
        for idx in range(len(date)):
            d = QTableWidgetItem(date[idx])
            t = QTableWidgetItem(time[idx])
            m = QTableWidgetItem(msg[idx])
            d.setTextAlignment(Qt.AlignCenter)
            t.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(idx, 0, d)
            self.table.setItem(idx, 1, t)
            self.table.setItem(idx, 2, m)

class Classifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()
        self.modelPath = None
        self.imgPath = None
        self.correctModelPath = False
        self.loadMetadata = False
        self.loadInputMetadata = False
        self.model = None
        
        self.metadata = None
        self.inputMetadata = None
        self.labelNames = None
        self.trueLabels = None
        
        self.imgList = None
        
        self.tmpImgPath = None
        self.imW = None
        self.imH = None
        
        self.tmpTrueLabel = None
        self.tmpPred = None
        
        self.predIMG = None
        
        self.lastTimestamp = None
        self.log = []
        
        self.status = None
        
    def setupUI(self):
        
        titleY = 20
        modelY = titleY + 40
        metaY = modelY + 30
        pathY = metaY + 60
        imgY = pathY + 80
        labelY = imgY + 360
        statY = labelY + 70   
        
        self.setWindowTitle('Traffic Sign Classifier (TSC)')
        
        # Menubar
        # Actions
        exitAction = QAction(QIcon('exit.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)
        
        loadModel = QAction('Load model', self)
        loadModel.setShortcut('Ctrl+O')
        loadModel.setStatusTip('Load trained CNN-Model')
        loadModel.triggered.connect(self.setModelPath)
        
        loadMeta = QAction('Load metadata', self)
        loadMeta.setShortcut('Ctrl+M')
        loadMeta.setStatusTip('Load metadata file (.csv) of GTSRB dataset.')
        loadMeta.triggered.connect(self.openMetadata)
        
        loadInputMeta = QAction('Load metadata for input images', self)
        loadInputMeta.setShortcut('Ctrl+I')
        loadInputMeta.setStatusTip('(optional) Load metadata file (.csv) for input images.')
        loadInputMeta.triggered.connect(self.openInputMetadata)
        
        setInputPath = QAction('Set input image path', self)
        setInputPath.setShortcut('Ctrl+P')
        setInputPath.setStatusTip('Set the path of input images and generate image list.')
        setInputPath.triggered.connect(self.setImgPath)
        
        log_save = QAction('Save log', self)
        log_save.setShortcut('Ctrl+L')
        log_save.setStatusTip('Saves log history')
        log_save.triggered.connect(self.save_log)
        
        logReview = QAction('Review logs', self)# not implemented
        logReview.setShortcut('Ctrl+R')
        logReview.setStatusTip('Shows log history')
        logReview.triggered.connect(self.logHistoryWindow)
        
        helpAction = QAction('Information', self)
        helpAction.setShortcut('F1')
        helpAction.setStatusTip('About TSC')
        # helpAction.triggered.connect(qApp.exit)
        
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        
        fileMenu = menubar.addMenu('&File') # & : allows short-cut
        logMenu = menubar.addMenu('&Log')
        helpMenu = menubar.addMenu('&Help')
        
        
        fileMenu.addAction(loadModel)
        fileMenu.addAction(loadMeta)
        fileMenu.addAction(loadInputMeta)
        fileMenu.addAction(setInputPath)
        fileMenu.addAction(exitAction)
        
        logMenu.addAction(log_save)
        logMenu.addAction(logReview)
        
        helpMenu.addAction(helpAction)
        
        
        
        
        self.statusBar()
        # 
        self.setGeometry(600,100,900,statY + 100)
        
             # GUI initiated
        
        titleMain = QLabel('Traffic Sign Classifier', self)
        titleMain.setAlignment(Qt.AlignCenter)
        titleMain.setGeometry(20, titleY, 860, 30)
        titleMain.setFont(QtGui.QFont('Arial', 16, QtGui.QFont.Bold))
        # titleMain.setStyleSheet('background-color: white; border: 1px solid block;')
        
        self.init_log()
        
        # import model 
        labelModel = QLabel('Model path: ', self)
        labelModel.setGeometry(40, modelY, 140, 20)
        labelModel.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        # labelModel.setStyleSheet('background-color: white; border: 1px solid black')
        
        self.labelModelPathIn = QLabel('N/A', self)
        self.labelModelPathIn.setGeometry(190, modelY, 560, 20)
        self.labelModelPathIn.setStyleSheet('background-color: white; border: 1px solid black')
        
        modelBrowseButton = QPushButton('Browse', self)
        modelBrowseButton.setGeometry(760, modelY, 100, 20)
        modelBrowseButton.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        modelBrowseButton.clicked.connect(self.setModelPath)
        
        # import meta data csv file
        labelMetaPath = QLabel('Meta data: ', self)
        labelMetaPath.setGeometry(40, metaY, 140, 20)
        labelMetaPath.setFont(QtGui.QFont('Arial', 9, QtGui.QFont.Bold))
        # labelMetaPath.setStyleSheet('background-color: white; border: 1px solid black;')
        
        self.labelMetaPathIn = QLabel('N/A', self)
        self.labelMetaPathIn.setGeometry(190, metaY, 560, 20)
        self.labelMetaPathIn.setStyleSheet('background-color: white; border: 1px solid black')
        
        metaBrowseButton = QPushButton('Browse', self)
        metaBrowseButton.setGeometry(760, metaY, 100, 20)
        metaBrowseButton.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        metaBrowseButton.clicked.connect(self.openMetadata)
        
        # meta data file of input images
        labelInputMetaPath = QLabel('Meta data (inputs): ', self)
        labelInputMetaPath.setGeometry(40, metaY+30, 140, 20)
        labelInputMetaPath.setFont(QtGui.QFont('Arial', 9, QtGui.QFont.Bold))
        # labelInputMetaPath.setStyleSheet('background-color: white; border: 1px solid black;')
        
        self.labelInputMetaPathIn = QLabel('optional', self)
        self.labelInputMetaPathIn.setGeometry(190, metaY+30, 560, 20)
        self.labelInputMetaPathIn.setStyleSheet('background-color: white; border: 1px solid black')
        
        inputMetaBrowseButton = QPushButton('Browse', self)
        inputMetaBrowseButton.setGeometry(760, metaY+30, 100, 20)
        inputMetaBrowseButton.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        inputMetaBrowseButton.clicked.connect(self.openInputMetadata)
        
        # Set image path
        labelPath = QLabel('Image path: ', self)
        labelPath.setGeometry(40, pathY, 140, 20)
        labelPath.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        # labelPath.setStyleSheet('background-color: white; border: 1px solid black;')
        
        self.labelImgPathIn = QLabel('N/A', self)
        self.labelImgPathIn.setGeometry(190, pathY, 560, 20)
        self.labelImgPathIn.setStyleSheet('background-color: white; border: 1px solid black')
        
        imgBrowseButton = QPushButton('Browse', self)
        imgBrowseButton.setGeometry(760, pathY, 100, 20)
        imgBrowseButton.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        imgBrowseButton.clicked.connect(self.setImgPath)
        
        # image list box
        labelImgList = QLabel('Image: ', self)
        labelImgList.setGeometry(40, pathY + 30, 140, 20)
        # labelImgList.setStyleSheet('background-color: white; border: 1px solid black')
        labelImgList.setFont(QtGui.QFont('Arial', 10, QtGui.QFont.Bold))
        
        self.listBox = QComboBox(self)
        self.listBox.setGeometry(190, pathY + 30, 560, 20)
        self.listBox.activated[str].connect(self.loadCurrentImg)
        
        # images
        # input image
        self.imW = 300
        self.imH = 300
        defaultImg = QPixmap('./dataset/no_image.png').scaledToWidth(self.imW)
        
        inputTitle = QLabel('Input image', self)
        inputTitle.setAlignment(Qt.AlignCenter)
        inputTitle.setGeometry(80, imgY, self.imW, 30)
        # inputTitle.setStyleSheet('background-color: white; border: 1px solid black')
        inputTitle.setFont(QtGui.QFont('Arial', 12, QtGui.QFont.Bold))
        
        self.inputIMG = QLabel('', self)
        self.inputIMG.setGeometry(80, imgY + 40, self.imW, self.imH)
        self.inputIMG.setStyleSheet('background-color: white; border: 1px solid black;')
        self.inputIMG.setPixmap(defaultImg)
        
        # prediction: meta image
        outputTitle = QLabel('Prediction', self)
        outputTitle.setAlignment(Qt.AlignCenter)
        outputTitle.setGeometry(520, imgY, self.imW, 30)
        # outputTitle.setStyleSheet('background-color: white; border: 1px solid black')
        outputTitle.setFont(QtGui.QFont('Arial', 12, QtGui.QFont.Bold))
        
        self.predictionIMG = QLabel('', self)
        self.predictionIMG.setGeometry(520, imgY + 40, self.imW, self.imH)
        self.predictionIMG.setStyleSheet('background-color: white; border: 1px solid black;')
        self.predictionIMG.setPixmap(defaultImg)
        
        # true and predicted labels
        # true label
        label1 = QLabel('True Label: ', self)
        label1.setGeometry(80, labelY, 130, 20)
        # label1.setStyleSheet('border: 1px solid black')
        label1.setFont(QtGui.QFont('Arial', 11))
        
        self.trueLabel = QLabel('N/A', self)
        self.trueLabel.setGeometry(220, labelY, 600, 20)
        self.trueLabel.setStyleSheet('background-color: white; border: 1px solid black')
        
        # prediction label
        label2 = QLabel('Prediction: ', self)
        label2.setGeometry(80, labelY + 30, 130, 20)
        # label2.setStyleSheet('border: 1px solid black')
        label2.setFont(QtGui.QFont('Arial', 11))
        
        self.predLabel = QLabel('N/A', self)
        self.predLabel.setGeometry(220, labelY + 30, 600, 20)
        self.predLabel.setStyleSheet('background-color: white; border: 1px solid black')
        
        # log and Status
        logBox = QLabel('Log: ', self)
        logBox.setGeometry(80, statY + 30 , 100, 20)
        # logBox.setStyleSheet('background-color: white; border: 1px solid black')
        logBox.setFont(QtGui.QFont('Arial', 10))
        
        self.logLabel = QLabel(self.log[-1],self)
        self.logLabel.setGeometry(200, statY + 30, 510, 20)
        self.logLabel.setStyleSheet('background-color: white; border: 1px solid black;')
        
        saveLog = QPushButton('Save Log', self)
        saveLog.setGeometry(720, statY + 30, 100, 20)
        saveLog.setStyleSheet('background-color: white; border: 1px solid black')
        saveLog.clicked.connect(self.save_log)
        
        statusBox = QLabel('Status: ', self)
        statusBox.setGeometry(80, statY, 100, 20)
        # statusBox.setStyleSheet('background-color: white; border: 1px solid black')
        statusBox.setFont(QtGui.QFont('Arial', 10))
        
        self.labelStatus = QLabel('Programm is initiated.', self)
        self.labelStatus.setGeometry(200, statY, 620, 20)
        self.labelStatus.setStyleSheet('background-color: white; border: 1px solid black')
    
    def getTmpTime(self):
        def convert(target):
            return str(target // 10) + str(target % 10)
        tmp = datetime.now()
        self.lastTimestamp = '{}.{}.{} | {}:{}:{}'.format(convert(tmp.year), convert(tmp.month), convert(tmp.day), convert(tmp.hour), convert(tmp.minute), convert(tmp.second))
    
    def init_log(self):
        self.log = []
        tmp = self.getTmpTime()
        self.log.append(self.lastTimestamp + ' | Programm is initiated.')
        
    def show_status(self):
        self.labelStatus.setText(self.status)
        
    def openMetadata(self):
        fname = QFileDialog.getOpenFileName(self)
        if fname[0].endswith('.csv'):
            self.loadMetadata = True
            self.metadata = pd.read_csv(fname[0])
            if not self.metadata.empty:
                self.loadMetadata = True
                self.labelMetaPathIn.setText(fname[0])
                self.getTmpTime()
                tmp_log = self.lastTimestamp + ' | Metadata file is loaded.'
                self.log.append(tmp_log)
                self.status = '[INFO] Metadata is loaded.'
                self.labelStatus.setText(self.status)
                self.labelNames = list(self.metadata['SignName'])
                
            else:
                tmp_log = self.lastTimestamp + ' | [ERROR] Incorrect metadata file'
                self.log.append(tmp_log)
                self.status = '[ERROR] Metadata file cannot be opened.'
                self.labelStatus.setText(self.status)
                
    def openInputMetadata(self):
        fname = QFileDialog.getOpenFileName(self)

        if fname[0].endswith('.csv'):
            self.loadInputMetadata = True
            self.labelInputMetaPathIn.setText(fname[0])
            self.inputMetadata = pd.read_csv(fname[0])
            
            if not self.inputMetadata.empty:
                self.getTmpTime()
                tmp_log = self.lastTimestamp + ' | Metadata file of input images is loaded.'
                self.log.append(tmp_log)
                self.status = '[INFO] Metadata of input images is loaded.'
                self.labelStatus.setText(self.status)
                fnames = []
                fpaths = []
                fclass = []
                data = open(fname[0]).read().strip().split('\n')[1:]
                for row in data:
                    img_name = row.strip().split('/')[-1]
                    img_path = row.strip().split(',')[-1]
                    img_class = row.strip().split(',')[-2]
                    fnames.append(img_name)
                    fpaths.append(img_path)
                    fclass.append(img_class)
                self.trueLabels = (fnames,fpaths, fclass)
                
            else:
                tmp_log = self.lastTimestamp + ' | [ERROR] Incorrect metadata file of input images'
                self.log.append(tmp_log)
                self.status = '[ERROR] Metadata file cannot be opened.'
                self.labelStatus.setText(self.status)
                
            
    def setModelPath(self):
        fname = QFileDialog.getOpenFileName(self)
        if fname[0].endswith('.h5'):
            self.correctModelPath = True
            self.modelPath = fname[0]
            self.labelModelPathIn.setText(fname[0])
            self.getTmpTime()
            tmp_log = self.lastTimestamp + ' | CNN-Model path is set.'
            self.logLabel.setText(tmp_log)
            self.log.append(tmp_log)
            
            self.status = '[INFO] Loading CNN-Model... It takes several minutes.'
            self.labelStatus.setText(self.status)
            
            self.model = load_model(fname[0])
            
            if self.model:
                self.getTmpTime()
                tmp_log = self.lastTimestamp + ' | CNN-Model is loaded.'
                self.log.append(tmp_log)
                self.logLabel.setText(tmp_log)
                self.status = '[INFO] The CNN-model is loaded'
                self.labelStatus.setText(self.status)
                
            else:
                self.status = '[ERROR] The CNN-Model cannot be opened. Please check the file name.'
                self.labelStatus.setText(self.status)
                self.getTmpTime()
                tmp_log = self.lastTimestamp + ' | [ERROR] Loading CNN-Model is failed. (Incorrect or damaged file)'
                self.log.appen(tmp_log)
            # check cnn modell
                # true : log update
                # False: error message
        else:
            self.labelModelPathIn.setText('Please select correct file. (*.h5)')
            self.getTmpTime()
            tmp_log = self.lastTimestamp + ' | [ERROR] Loading CNN-Model is Failed. (Incorrect file name)'
            self.status = '[ERROR] Please select correct file. (*.h5)'
            self.labelStatus.setText(self.status)
        
    def getImgList(self):
        imgPath = []
        imgList = []
        for (path, dir, files) in os.walk(self.imgPath):
            for fname in files:
                if fname.endswith('.jpg') | fname.endswith('.jpeg')| fname.endswith('.png'):
                    imgPath.append(os.path.join(path, fname))
                    imgList.append(fname)
        
        self.imgList = (imgList, imgPath)
    
    def setImgPath(self):
        fPath = QFileDialog.getExistingDirectory(self)
        self.imgPath = fPath
        self.labelImgPathIn.setText(fPath)
        self.getTmpTime()
        tmp_log = self.lastTimestamp + ' | Input-image directory is set.'
        self.logLabel.setText(tmp_log)
        self.log.append(tmp_log)
        
        self.listBox.clear()
        self.getImgList()
        if self.imgList:
            self.getTmpTime()
            tmp_log = self.lastTimestamp + ' | Input-image list is generated.'
            self.status = '[INFO] Input-image list is generated.'
            self.logLabel.setText(tmp_log)
            self.log.append(tmp_log)
            
            # update image list on the QComboBox
            tmp_list = self.imgList[0]
            for item in tmp_list:
                self.listBox.addItem(item)
        else:
            self.getTmpTime()
            tmp_log = self.lastTimestamp + " | [ERROR] No images."
            self.logLabel.setText(tmp_log)
            self.log.append(tmp_log)
            self.status = "[ERROR] There's no image."
            self.labelStatus.setText(self.status)
    
    def loadCurrentImg(self, fname):
        # pathList = self.imgList[0]
        # imgList = self.imgList[1]
        tmp_idx = self.imgList[0].index(fname)
        self.tmpImgPath = self.imgList[1][tmp_idx]
        pixmap = QPixmap(self.tmpImgPath).scaledToWidth(300)
        self.inputIMG.setPixmap(pixmap)
        
        input_idx = self.trueLabels[0].index(fname)
        true_class  = int(self.trueLabels[2][input_idx])
        self.tmpTrueLabel = self.labelNames[true_class]
        self.trueLabel.setText(self.tmpTrueLabel)
        
        if self.model and self.loadMetadata == True:
            self.predict(self.tmpImgPath)
            
        else:
            self.getTmpTime()
            tmp_log = self.lastTimestamp + ' | [WARNING] CNN-Model is not loaded.'
            self.log.append(tmp_log)
            
            self.status = '[WARNING] CNN-Model is not loaded.'
            self.labelStatus.setText(self.status)
        
    def predict(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32,32))
        img = exposure.equalize_adapthist(img, clip_limit = 0.1) # CLAHE
        
        img = img.astype('float32') / 255.
        img = np.expand_dims(img, axis = 0)
        
        pred_vec = self.model.predict(img)
        pred_idx = pred_vec.argmax(axis = 1)[0] # label index
        
        self.tmpPred = self.labelNames[pred_idx]
        self.predIMG = QPixmap('./dataset/Meta/' + str(pred_idx) + '.png').scaledToWidth(300)
        self.predictionIMG.setPixmap(self.predIMG)
        
        self.predLabel.setText(self.tmpPred)
    
    def save_log(self):
        def convert(target):
            return str(target // 10) + str(target % 10)
        tmp = datetime.now()
        save_path = './{}{}{}{}{}{}_logs.csv'.format(convert(tmp.year), convert(tmp.month), convert(tmp.day), convert(tmp.hour), convert(tmp.minute), convert(tmp.second))
        msg = []
        date = []
        time = []
        
        for log in self.log:
            msg.append(log.split(' | ')[2])
            date.append(log.split(' | ')[0])
            time.append(log.split(' | ')[1])
        
        df = pd.DataFrame({'Date': date,
                           'Time': time,
                           'Message': msg})
        df.to_csv(save_path, index = False)
        
    def logHistoryWindow(self):
        self.history = LogHistory(parent = self)
        self.history.show()
        
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    classifier = Classifier()
    classifier.init_log()
    classifier.show()
    
    app.exec_()