# -*- coding: utf-8 -*-
# @author: Zijun

# Builtin packages and PyQt
from PyQt5 import QtWidgets, QtCore, QtGui
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import glob
import pydicom as dicom
from mainWindow import Ui_MainWindow as AnnotationMainWindow 


class GraphicsScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent=None):
        QtWidgets.QGraphicsScene.__init__(self, parent)
        # https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QGraphicsScene.html?
        self.index = 0
        self.circle_items = []

    def set_index(self, num):
        self.index = num

    def mousePressEvent(self, event):
        # https://stackoverflow.com/questions/13965505/how-to-save-a-graphics-image-from-qgraphicsscene-in-pyqt
        pen = QtGui.QPen(QtCore.Qt.black)
        brush = QtGui.QBrush(QtCore.Qt.black)
        pensize = 10
        x = event.scenePos().x()
        y = event.scenePos().y()
        self.circle_items.append(self.addEllipse(x-pensize/2, y-pensize/2, pensize, pensize, pen, brush))
        print(x, y)

class MainWindow(QtWidgets.QMainWindow, AnnotationMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.ref_size = 512
        self.gray_color_table = [QtGui.qRgb(i, i, i) for i in range(256)]
        self.setup_view_scene()

        ## Browse
        self.browseButton.clicked.connect(self.browsefoler)
        self.clearButton.clicked.connect(lambda x: self.loadDirlineEdit.setText(' '))
        self.loadDirlineEdit.textChanged.connect(lambda text: self.loadDirlineEdit.setText(str(text)))
        self.loadButton.clicked.connect(self.get_dicom_list)
        self.indexSpinBox.valueChanged.connect(lambda num: self.display_dicom(num))
        self.displayButton.clicked.connect(self.display_dicom)

        self.frameHorizontalScrollBar.valueChanged.connect(lambda num: self.display_image(num-1))
        self.previousButton.clicked.connect(self.spinbox_increase)
        self.nextButton.clicked.connect(self.spinbox_decrease)
        # (lambda num: self.indexSpinBox.setValue(num))

    def spinbox_increase(self):
        current_value = self.indexSpinBox.value() + 1
        self.indexSpinBox.setValue(current_value)

    def spinbox_decrease(self):
        current_value = self.indexSpinBox.value() - 1
        self.indexSpinBox.setValue(current_value)

    def browsefoler(self):      
        if len(self.loadDirlineEdit.text()) != 0 and os.path.exists(self.loadDirlineEdit.text()):
            self.videoDir = Path(self.loadDirlineEdit.text())
        else:
            videoDir = QtWidgets.QFileDialog.getExistingDirectory(self,
                'Select a Vedio Folder', os.getenv('HOME'), 
                QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks)
            self.videoDir = Path(videoDir)
            self.loadDirlineEdit.setText(str(videoDir)) 

    def get_dicom_list(self):
        self.videoPath = glob.glob(os.path.join(self.videoDir,'*.dcm'))
        # print(self.videoPath[0], len(self.videoPath))
        self.indexSpinBox.setValue(1)
        self.totalLabel.setText(str(len(self.videoPath)))

    def display_dicom(self, display_index):
        # The key of dispaly: SpinBox value!!
        # display_index = self.indexSpinBox.value()
        dir_name = Path(self.videoPath[display_index]).name
        self.filenameLabel.setText(dir_name.split('.')[0])
        dicom_read = dicom.dcmread(self.videoPath[display_index])   
        frame_array = dicom_read.pixel_array
        if 2**8 < np.amax(frame_array) < 2**16:
            ratio = np.amax(frame_array) / 256     
            self.frame_array = (frame_array / ratio).astype('uint8')
        else:
            self.frame_array = frame_array.astype('uint8')
        frame_count, _, _ = self.frame_array.shape
        self.maxFrameLabel.setText(str(frame_count))
        self.frameHorizontalScrollBar.setRange(1,frame_count)
        self.display_image(0)

    def setup_view_scene(self):
        # Initate a customized QGraphicsScene in the Widget
        angioScene = GraphicsScene(self)
        # Initate a QGraphicsPixmapItem 
        self.angioImageItem = QtWidgets.QGraphicsPixmapItem()
        # Add the item to the QGraphicsPixmapItem in the QGraphicsScene
        angioScene.addItem(self.angioImageItem) 
        # Set the GraphicsView with the QGraphicsScene and centered on QGraphicsPixmapItem
        self.frameGraphicsView.setScene(angioScene)
        self.frameGraphicsView.centerOn(self.angioImageItem)

    def display_image(self, index):
        # Create QtGui.Pixmap from self.frame_array and index for displaying
        im = self.frame_array[index] # Shape of (n,n), n=512 or n=1024
        frameQImage = QtGui.QImage(im.data, im.shape[1], im.shape[0], 
            im.strides[0], QtGui.QImage.Format_Indexed8)
        frameQImage.setColorTable(self.gray_color_table)
        frameQPixmap = QtGui.QPixmap.fromImage(frameQImage).scaled(self.ref_size, 
            self.ref_size, QtCore.Qt.KeepAspectRatio)
        self.angioImageItem.setPixmap(frameQPixmap)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
