# -*- coding: utf-8 -*-
# @author: Zijun

# Builtin packages and PyQt
from PyQt5 import QtWidgets, QtCore, QtGui
import os, sys
from pathlib import Path
import numpy as np
import glob
import pydicom as dicom
from mainWindow import Ui_MainWindow as AnnotationMainWindow 


class GraphicsScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent=None):
        QtWidgets.QGraphicsScene.__init__(self, parent)
        # https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QGraphicsScene.html?
        self.index = 1
        self.circle_items = []
        self.circle_position = []
        self.pensize = 10
        self.pen = QtGui.QPen(QtCore.Qt.black)
        self.brush = QtGui.QBrush(QtCore.Qt.black)

    def increase_index(self):
        self.index += 1

    def get_index(self):
        return self.index

    def reset_index(self):
        self.index = 1

    def mousePressEvent(self, event):
        x = event.scenePos().x()
        y = event.scenePos().y()
        self.circle_items.append(self.addEllipse(x-self.pensize/2, y-self.pensize/2, 
            self.pensize, self.pensize, self.pen, self.brush))
        self.circle_position.append((x,y))

    def clearCircleItems(self):
        for item in self.circle_items:
            self.removeItem(item) 
        self.circle_items = []
        self.circle_position = []

    def get_location(self):
        return self.circle_position 

class MainWindow(QtWidgets.QMainWindow, AnnotationMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Some Default Settings
        self.ref_size = 512
        self.gray_color_table = [QtGui.qRgb(i, i, i) for i in range(256)]
        self.setup_view_scene()
        self.data_home_dir = os.path.join('Z:','Datasets','Angiogram')
        self.output_home_dir = os.path.join('Z:','Projects','Angiogram','Data','Processed')
        self.saveDirLineEdit.setText(os.path.join(self.output_home_dir,'Zijun','Synthetic','BackGround_Image'))
        self.endPointButton.setChecked(True)
        self.label_identifier = lambda ep_button : 'endpoint' if ep_button.isChecked() else 'stenosis'
        self.saveDir = Path(self.saveDirLineEdit.text())
        self.save_flag = False

        ## Load Dir Operations
        self.browseButton.clicked.connect(self.browsefoler)
        self.clearButton.clicked.connect(lambda x: self.loadDirlineEdit.setText(' '))
        self.loadDirlineEdit.textChanged.connect(lambda text: self.loadDirlineEdit.setText(str(text)))
        self.loadButton.clicked.connect(self.get_dicom_list)

        ## Save Dir Operations
        self.browseSetButton.clicked.connect(self.browsesavefoler)
        self.saveDirLineEdit.textChanged.connect(lambda text: self.saveDirLineEdit.setText(str(text)))
        self.outputClearButton.clicked.connect(lambda x: self.saveDirLineEdit.setText(' '))
        self.endPointButton.toggled.connect(lambda bool: self.stenosisLabelButton.setChecked(1-bool))
        self.stenosisLabelButton.toggled.connect(lambda bool: self.endPointButton.setChecked(1-bool))

        ## Display on Dicom File Level 
        self.indexSpinBox.valueChanged.connect(lambda num: self.display_dicom(num-1))
        self.previousButton.clicked.connect(self.spinbox_decrease)
        self.nextButton.clicked.connect(self.spinbox_increase)

        ## Display on Frame Level
        self.frameHorizontalScrollBar.valueChanged.connect(lambda num: self.display_image(num-1))
        self.clearDotsButton.clicked.connect(self.angioScene.clearCircleItems)

        ## Output Operations
        self.saveButton.clicked.connect(self.save_annotation)

    def spinbox_increase(self):
        if not self.save_flag:
            # The background and image has not been saved and there are annotation
            # automatically save the output
            self.save_annotation()
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
                'Select a Vedio Folder', self.data_home_dir, QtWidgets.QFileDialog.DontResolveSymlinks)
            self.videoDir = Path(videoDir)
            self.loadDirlineEdit.setText(str(videoDir)) 

    def browsesavefoler(self): 
        if len(self.saveDirLineEdit.text()) >= 2 and os.path.exists(self.saveDirLineEdit.text()):
            self.saveDir = Path(self.saveDirLineEdit.text())
        else:
            saveDir = QtWidgets.QFileDialog.getExistingDirectory(self,
                'Select a Save Folder', self.output_home_dir, QtWidgets.QFileDialog.DontResolveSymlinks)
            self.saveDir = Path(saveDir)
            self.saveDirLineEdit.setText(str(saveDir)) 

    def get_dicom_list(self):
        """Must Have the saveDir and videoDir present"""
        self.videoPath = glob.glob(os.path.join(self.videoDir,'*.dcm'))
        self.video_folder = self.saveDir / f'{self.videoDir.parent.stem}_{self.videoDir.stem}'
        self.video_folder.mkdir(parents=True, exist_ok=True)
        # Start from where we lefted
        self.indexSpinBox.setValue(len(glob.glob(os.path.join(self.video_folder, "*", "")))+1)
        self.totalLabel.setText(str(len(self.videoPath)))

    def display_dicom(self, display_index):
        # Though not directly, all the dicom level display is related with the value change of indexSpinBox.
        # Therefore, indexSpinBox's value must not be changed within this function, otherwise it's a loop.
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
        self.display_image(self.frameHorizontalScrollBar.value()-1)
        # Set up save folder
        self.filename_folder = f'{self.filenameLabel.text()}'
        self.saveDirParent = self.video_folder / self.filename_folder
        # Reset the counting for frames
        self.angioScene.reset_index()

    def setup_view_scene(self):
        # Initate a customized QGraphicsScene in the Widget
        self.angioScene = GraphicsScene(self)
        # Initate a QGraphicsPixmapItem 
        self.angioImageItem = QtWidgets.QGraphicsPixmapItem()
        # Add the item to the QGraphicsPixmapItem in the QGraphicsScene
        self.angioScene.addItem(self.angioImageItem) 
        # Set the GraphicsView with the QGraphicsScene and centered on QGraphicsPixmapItem
        self.frameGraphicsView.setScene(self.angioScene)
        self.frameGraphicsView.centerOn(self.angioImageItem)

    def display_image(self, index):
        # Create QtGui.Pixmap from self.frame_array and index for displaying
        im = self.frame_array[index] # Shape of (n,n), n=512 or n=1024
        frameQImage = QtGui.QImage(im.data, im.shape[1], im.shape[0], 
            im.strides[0], QtGui.QImage.Format_Indexed8)
        frameQImage.setColorTable(self.gray_color_table)
        self.frameQPixmap = QtGui.QPixmap.fromImage(frameQImage).scaled(self.ref_size, 
            self.ref_size, QtCore.Qt.KeepAspectRatio)
        self.angioImageItem.setPixmap(self.frameQPixmap)
        self.save_flag = False

    def save_annotation(self):
        if len(self.angioScene.get_location()) > 0:
            # Get region of scene to capture as a QPixmap
            capturePixelImage = QtGui.QPixmap(self.ref_size, self.ref_size)
            painterForCapture = QtGui.QPainter(capturePixelImage)
            self.angioScene.render(painterForCapture)
            # Set up image saving directory
            idnetifier = self.label_identifier(self.endPointButton)
            index = self.angioScene.get_index()
            self.saveDirParent.mkdir(parents=True, exist_ok=True)
            capturePixelImage.save(str(self.saveDirParent / f'{idnetifier}_{index}.png'), "PNG")
            self.frameQPixmap.save(str(self.saveDirParent / f'background_{index}.png'), "PNG")
            csv_save_path = self.saveDirParent / f'{idnetifier}_{index}.csv'
            self.save_information(csv_save_path)
            self.save_flag = True
            painterForCapture.end()
            self.angioScene.clearCircleItems()
            self.angioScene.increase_index()

    def save_information(self, save_dir):
        headline = ['filename','load_dir','frame_num','x','y']
        location_list = self.angioScene.get_location()
        with open(save_dir, 'w') as fileHandle:
            fileHandle.write(','.join(headline) + '\n')
            for location_pairs in location_list:
                fileHandle.write(f'{self.filename_folder},{str(self.videoDir)},{self.frameHorizontalScrollBar.value()},')
                location_pairs = [str(i) for i in location_pairs]
                fileHandle.write(','.join(location_pairs)+'\n')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
