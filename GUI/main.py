# -*- coding: utf-8 -*-
# @author: Zijun

# Builtin packages and PyQt
from PyQt5 import QtWidgets,QtCore,QtGui
import os, sys
from pathlib import Path
import pandas as pd
import glob
import pydicom as dicom
from mainWindow import Ui_MainWindow as AnnotationMainWindow 


# class GraphicsScene(QGraphicsScene):
#     def __init__(self, parent=None):
#         QGraphicsScene.__init__(self, parent)
#         self.setSceneRect(-100, -100, 200, 200)
#         self.opt = ""

#     def setOption(self, opt):
#         self.opt = opt

#     def mousePressEvent(self, event):
#         pen = QPen(QtCore.Qt.black)
#         brush = QBrush(QtCore.Qt.black)
#         x = event.scenePos().x()
#         y = event.scenePos().y()
#         if self.opt == "Generate":
#             self.addEllipse(x, y, 4, 4, pen, brush)
#         elif self.opt == "Select":
#             print(x, y)


# class SimpleWindow(QtWidgets.QMainWindow, points.Ui_Dialog):
#     def __init__(self, parent=None):
#         super(SimpleWindow, self).__init__(parent)
#         self.setupUi(self)

#         self.scene = GraphicsScene(self)
#         self.graphicsView.setScene(self.scene)
#         self.graphicsView.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

#         group = QButtonGroup(self)
#         group.addButton(self.radioButton)
#         group.addButton(self.radioButton_2)

#         group.buttonClicked.connect(lambda btn: self.scene.setOption(btn.text()))
#         self.radioButton.setChecked(True)
#         self.scene.setOption(self.radioButton.text())

class MainWindow(QtWidgets.QMainWindow, AnnotationMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        ## Browse
        self.browseButton.clicked.connect(self.browsefoler)
        self.clearButton.clicked.connect(lambda x: self.loadDirlineEdit.setText(' '))
        self.loadDirlineEdit.textChanged.connect(lambda text: self.loadDirlineEdit.setText(str(text)))
        self.loadButton.clicked.connect(self.get_dicom_list)
        self.indexSpinBox.valueChanged.connect(lambda num: self.display_dicom)
        self.displayButton.clicked.connect(self.display_dicom)

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

    def display_dicom(self):
        # The key of dispaly: SpinBox value!!
        display_index = self.indexSpinBox.value()
        dir_name = Path(self.videoPath[display_index]).name
        self.filenameLabel.setText(dir_name.split('.')[0])
        ds = dicom.dcmread(self.videoPath[display_index])
        self.ds_array = ds.pixel_array
        count, _, _ = ds_array.shape
        self.maxFrameLabel.setText(str(count))
        self.frameHorizontalScrollBar.setRange(1,count)

    def display_image(self):
        self.frameGraphicsView
        # Initate a QGraphicsScene in the Widget
        angioScene = QtWidgets.QGraphicsScene(self)
        # Initate a QGraphicsPixmapItem 
        angioImageItem = QtWidgets.QGraphicsPixmapItem()
        # Add the item to the QGraphicsPixmapItem in the QGraphicsScene
        angioScene.addItem(angioImageItem) 
        # Set the GraphicsView with the QGraphicsScene and centered on QGraphicsPixmapItem
        self.frameGraphicsView.setScene(angioScene)
        self.frameGraphicsView.centerOn(angioImageItem)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
