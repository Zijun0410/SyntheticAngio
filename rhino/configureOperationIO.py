__author__ = "zijung@umich.edu"
__version__ = "2021.10.25"

import os
import Rhino
import System
from configureAngulation import viewportByName
import random 
import math

def GetString(message=None, defaultString=None, strings=None):
    """Pauses for user input of a string value
    Inputs:
        message: <python str>, optional, a prompt or message
        defaultString: <python str>, optional, a default value
        strings ([str, ...], optional): list of strings to be displayed as a click-able command options.
                                      Note, strings cannot begin with a numeric character
    Outputs:
        str: The string either input or selected by the user if successful.
           If the user presses the Enter key without typing in a string, an empty string "" is returned.
        None: if not successful, on error, or if the user pressed cancel.

    """
    gs = Rhino.Input.Custom.GetString()
    gs.AcceptNothing(True)
    if message: gs.SetCommandPrompt(message)
    if defaultString: gs.SetDefaultString(defaultString)
    if strings:
        for s in strings: gs.AddOption(s)
    result = gs.Get()
    if result==Rhino.Input.GetResult.Cancel: return None
    if (result == Rhino.Input.GetResult.Option):
        return gs.Option().EnglishName
    return gs.StringResult()

def ReadMetaData(baseDir, meta_file_name):
    """
    Read in the csv file and store all the relevant information in a dictionary
    Inputs:
        baseDir: <python string> a base dir where all the meta data are saved
        meta_file_name: <python string> the name of the meta file
    Outputs:
        metaData: <python dictionary>, as its name, key are the information name, 
          values are list of related information
          {'FileName': ['001 (7)_R',  '001 (8)_R',  '001 (9)_R', ..]}
          Headers are:
          ['FileName','CenterX', 'CenterY', 'Width', 'Height', 'PositionerPrimaryAngle',
           'PositionerSecondaryAngle','DistanceSourceToDetector','DistanceSourceToPatient',
           'EstimatedRadiographicMagnificationFactor','ExposureTime','TableHeight',
           'DistanceObjectToTableTop', 'BeamAngle']
          If there are missing values in the orginal dicom file then the default is -1
        recordNum: <python int> the number of records in the file
    """
    metaData = dict()
    with open(os.path.join(baseDir, 'Meta_Data', meta_file_name), 'r') as fileHandle:
        lineIndex = 0
        metaDataList = []
        for line in fileHandle.readlines():
            metaDataRow = line.strip().split(',')
            if lineIndex == 0:
                nameList = metaDataRow
            dataIndex = 0
            for iData in metaDataRow:
                if lineIndex == 0:
                    metaData[dataIndex] = []
                else:
                    metaData[dataIndex].append(iData)
                dataIndex += 1
            lineIndex += 1
    metaData = dict(zip(nameList, list(metaData.values())))
    recordNum = lineIndex-1 # minus 1 because of the header column
    return metaData, recordNum

def MetaValueCheck(metaData, iRecord, side):
    """
    Check and Return necessary metaData values under different metadata settings.
    e.g., the UKR Dataset and the UoMR Dataset
    When values are not avalible, return sampled values or mean values
    Inputs:
         metaData: <python dict>
         iRecord: <python string>
    Outputs:
        metaInfor: <python list> []
        fileName: <python string>
    """
    #-# The angleSampleListR data is obtained from the UKR dataset
    random.seed(0)
    angleSampleListR = [(41.1, 2.3), (21.1, 20.1), (-35.5, 0.4), (22.2, 20.7), 
        (-38.7, -0.7), (38.9, 0.2), (38.5, -0.1), (20.2, 20.2), (-30.6, -2.6), 
        (40.8, 23.7), (-31.6, 0.4), (40.8, 2.0), (19.8, 20.7), (-31.2, -1.5), 
        (40.5, -0.2), (38.1, -1.1), (38.2, 26.3), (-33.1, 0.4), (40.2, -0.6), 
        (20.5, 20.8), (-31.1, 2.2), (20.4, 19.7), (-32.8, -0.3), (40.2, -0.1), 
        (41.4, 0.0), (20.1, 21.3), (-31.7, -0.5), (18.7, 19.6), (-29.6, 0.4), 
        (40.3, -0.2), (-41.7, -2.2), (39.2, 1.5), (20.9, 20.6), (-30.4, -0.5), 
        (34.4, 0.2), (20.0, 22.1), (-30.7, 0.8), (40.2, 0.0), (22.5, 21.1), 
        (-32.5, 0.5), (38.8, 2.6), (19.5, 19.8), (-33.2, 2.0), (-35.7, 0.7), 
        (39.8, -2.1), (39.5, 1.2), (16.4, 20.7), (-30.0, 1.3), (41.1, 0.1), 
        (19.6, 18.8), (-34.4, 0.9), (-34.6, -1.1), (34.6, -1.0), (21.3, 21.7), 
        (40.2, -2.3), (-33.6, -2.3), (39.8, 0.3), (19.6, 21.4), (-34.1, -0.1), 
        (-29.2, 1.1), (39.7, -0.2), (20.5, 19.9), (40.7, 1.0), (19.5, 21.8), 
        (24.9, 27.4), (-0.8, 0.4), (-33.6, 0.4), (39.8, 1.3), (-30.4, -0.7), 
        (39.8, 0.0), (20.4, 19.4), (-31.1, 1.9), (23.8, 21.7), (-29.8, 1.4), 
        (40.2, 0.0), (20.3, 25.0), (40.2, 1.0), (20.5, 19.8), (-33.2, -0.8), 
        (38.3, 2.0), (-33.6, 1.9), (41.4, -0.4), (-43.9, 1.0), (-45.7, 2.3), 
        (-2.7, 15.9), (43.5, 0.1), (0.4, 23.2), (-38.7, 1.5), (46.2, -0.6), 
        (-42.4, -0.8), (0.6, 22.0), (-87.4, -1.3), (2.3, 23.0), (34.5, 0.4), 
        (-2.1, 21.1), (-32.8, 3.0), (41.3, -0.7), (28.9, 22.5), (-28.4, 1.5), 
        (41.2, -0.9), (36.8, 20.9), (-40.3, -0.4), (11.5, 16.0), (-30.4, 1.2), 
        (8.5, 24.4), (-29.5, 3.0), (-31.0, 0.2), (29.5, 0.0), (-29.5, 0.3), 
        (28.9, 0.3), (-31.1, 0.3), (-29.0, -0.3), (30.7, -0.6), (30.5, 0.0), 
        (-29.5, 0.3), (30.7, -0.4), (29.8, 27.8), (0.7, 31.3), (-31.4, 3.0), 
        (-91.0, 2.8), (29.2, -15.5), (30.5, -16.1), (1.2, 29.9), (33.6, 4.1), 
        (-29.8, 4.6), (-29.1, 4.0), (29.9, 0.3), (32.5, -1.2), (0.5, 27.0), 
        (-30.2, -10.6), (-0.7, 33.1), (-28.9, 3.4), (27.2, -2.6), (31.3, -7.5), 
        (29.9, -20.7), (3.0, 35.9), (-34.1, 0.5), (39.8, -0.7), (21.3, 23.2), 
        (-34.7, 2.3), (35.1, 1.3), (-33.7, 1.3), (19.8, 20.8), (19.8, 20.8), 
        (40.0, 0.3), (21.1, 23.0), (-32.5, 2.6), (37.0, 0.9), (19.6, 23.3), 
        (-31.0, -1.5), (20.1, 20.4), (-33.3, 1.5), (39.7, -0.4), (34.8, -0.3), 
        (20.3, 19.7), (-35.4, 0.2), (37.9, 0.1), (-32.1, 0.1), (40.3, 0.1), 
        (39.2, 0.1), (19.8, 21.3), (-35.3, -0.2), (39.4, 0.0), (-32.1, 0.0), 
        (39.3, 0.0), (-35.3, 0.0), (40.2, 0.0), (40.2, 0.0), (29.4, 20.4), 
        (-33.8, -1.8), (39.4, -2.4), (39.4, 24.7), (-32.4, 0.4), (40.5, -1.1), 
        (-30.2, 2.0), (20.5, 21.1), (-29.5, 0.6), (28.8, 1.0), (-27.2, 1.0), 
        (-0.2, 1.0), (40.2, -0.2), (-40.5, -0.4), (0.0, 31.2), (44.1, 1.5), 
        (-2.0, 33.6), (-40.6, -3.1), (35.1, -0.8), (-1.1, 27.0), (-39.3, 0.9), 
        (34.6, 24.5), (33.7, 0.4), (-35.2, 0.3), (-3.0, 26.8), (5.1, 21.4), 
        (35.5, -0.3), (-32.0, 0.0), (27.1, -0.2), (-31.8, -0.2), (32.5, 1.3), 
        (12.9, 18.5), (12.9, 18.5), (-32.1, 0.1), (16.8, 20.3), (-30.8, 1.1), 
        (29.7, -0.3), (29.0, -0.8), (-29.6, -0.3), (33.4, -0.7), (31.4, -0.1), 
        (18.2, 17.1), (31.7, 1.8), (18.0, 18.8), (-28.4, 2.0), (52.1, 0.2), 
        (-34.3, 0.2), (-31.7, 0.0), (47.6, 0.0), (38.8, 1.2), (-33.9, 1.2), 
        (39.9, -0.1), (-36.6, -0.1), (39.6, -0.1), (-31.4, -0.1), (42.2, -1.7), 
        (-45.3, -1.7), (-0.4, -0.4), (39.0, 0.1), (-30.2, -0.1), (38.7, 0.0), 
        (-28.7, 0.0), (30.7, -0.5), (-29.0, 1.9), (-29.0, 1.9), (12.6, 30.1), 
        (-31.2, 0.7), (35.1, 0.0), (-27.8, 0.0), (39.5, 1.1), (39.9, 0.1), 
        (39.9, 0.1), (-31.3, 0.0), (15.7, 32.0), (45.4, 0.0), (-45.9, 0.0), 
        (37.4, 21.7), (44.9, 0.0), (41.8, 0.0), (35.8, 24.4), (1.6, 0.7), 
        (-40.4, 0.7), (34.5, -7.0), (-1.1, 28.9), (-36.6, 0.3), (28.8, 0.1), 
        (-30.9, -1.8), (-34.2, 0.8), (30.9, -0.3), (26.0, -11.5), (-0.4, 28.0), 
        (0.5, 28.3), (-32.3, -2.0), (35.1, 0.1), (-34.4, 0.1), (36.9, 0.1), 
        (36.9, 0.1), (40.0, -0.5), (-30.3, -0.4), (38.8, 0.1), (-28.7, 0.0), 
        (31.9, -0.5), (-29.9, -0.3), (32.5, 0.5), (27.3, 3.4), (10.6, 21.7), 
        (-30.6, 0.4), (-27.8, -0.5), (31.0, 0.0), (16.8, 22.2), (39.4, -0.3), 
        (21.6, 27.3), (-29.9, 2.3)]

    try:
        fileName = metaData['filename'][iRecord]
    except Exception:
        fileName = metaData['FileName'][iRecord]

    #-# Check the primary and secondary angles
    try:
        positionerPrimaryAngle = float(metaData['PositionerPrimaryAngle'][iRecord])
        positionerSecondaryAngle = float(metaData['PositionerSecondaryAngle'][iRecord])
        # Only pass the assert when: 1. assert not (False or False)
        #                            2. assert (True and True)
        assert not ( math.isnan(positionerPrimaryAngle) or math.isnan(positionerSecondaryAngle)) 
        assert (positionerSecondaryAngle != -1 and positionerPrimaryAngle != -1)
    except Exception:
        if side=='L':
            # TODO
            positionerPrimaryAngle = -1
            positionerSecondaryAngle = -1
        else: 
            # The RCA cases
            angle_1, angle_2 = random.choice(angleSampleListR)
            random_noise = random.choice([-0.1, 0, 0.1])
            positionerPrimaryAngle = angle_1 + random_noise
            random_noise = random.choice([-0.1, 0, 0.1])
            positionerSecondaryAngle = angle_2 + random_noise

    #-# Check the distance variables
    try:
        distanceSourceToDetector = float(metaData['DistanceSourceToDetector'][iRecord])
        assert distanceSourceToDetector != -1
    except Exception as e:
        distanceSourceToDetector = 1040.3    

    try:
        distanceSourceToPatient = float(metaData['DistanceSourceToPatient'][iRecord])
        assert distanceSourceToPatient != -1
    except Exception as e:
        distanceSourceToPatient = 776.5    

    metaInfor = [positionerPrimaryAngle, positionerSecondaryAngle, distanceSourceToDetector, distanceSourceToPatient]
    return metaInfor, fileName 

def LoadCurveFromTxt(baseDir, defaultBranches, folderName='RCA_Brief'):
    """
    Load back the reconstructed curve from the text file
    Input:
        baseDir: a valid directory where the curve control points are saved
        defaultBranches: a dictionary of the correspondance of branch and their index 
    Output:
        reconstructedCurves: a dictionary of {'identifier':<Rhino.Geometry.Curve>} value pairs
    """
    reconstructedCurves = {}
    for branchIdx in list(defaultBranches.keys()):
        pintList = []
        with open(os.path.join(baseDir, 'Construction', folderName, '{}.txt'.format(branchIdx)), 'r') as fileHandle:
            linesIn = fileHandle.readlines()
            for line in linesIn:
                locations = line.strip('\n').split(', ')
                # Turn into list of floats
                locations = list(map(float,locations))
                pintList.append(Rhino.Geometry.Point3d(locations[0], locations[1], locations[2]))
        identifier = defaultBranches[branchIdx]
        reconstructedCurves[identifier] = Rhino.Geometry.Curve.CreateControlPointCurve(pintList)
    return reconstructedCurves


def CaptureViewToFile(filePath, viewport, width=None, height=None, displayMode='Shaded', transparent=True):

    """
    Capture a Viewport to a PNG file path.
    Args:
        filePath: Full path to the file where the image will be saved.
        viewport: An active 'Perspective' viewport that has its display mode set
        width: Integer for the image width in pixels. If None, the width of the
            active viewport will be used. (Default: None).
        height: Integer for the image height in pixels. If None, the height of the
            active viewport will be used. (Default: None).
        displayMode: Text for the display mode to which the Rhino viewport will be
            set. For example: Wireframe, Shaded, Rendered, etc. If None, it will
            be the current viewport's display mode. (Default: Shaded).
        transparent: Boolean to note whether the background of the image should be
            transparent or have the same color as the Rhino scene. (Default: True).

    Returns:
        Full path to the image file that was written.
    """

    # create the view capture object
    activeViewport = viewportByName()
    imgW = activeViewport.Size.Width if width is None else width
    imgH = activeViewport.Size.Height if height is None else height
    # print(imgW, imgH)
    imgSize = System.Drawing.Size(imgW, imgH)

    # capture the view
    if displayMode is not None:
        modeObj = Rhino.Display.DisplayModeDescription.FindByName(displayMode)
        pic = viewport.ParentView.CaptureToBitmap(imgSize, modeObj)
    else:
        pic = viewport.ParentView.CaptureToBitmap(imgSize)

    # remove the background color if requested
    if transparent:
        backCol = Rhino.ApplicationSettings.AppearanceSettings.ViewportBackgroundColor
        pic.MakeTransparent(backCol)

    # save the bitmap to a png file
    if not filePath.endswith('.png'):
        filePath = '{}.png'.format(filePath)
    System.Drawing.Bitmap.Save(pic, filePath)

    return filePath


def saveStenosisInfor(saveInfor, inforSaveDir, csvName='stnosis_infor.csv'):
    """
    Save the random generated stenosis information to file
    Inputs:
        saveInfor: <python dict> with key of (index, fileName) and value of a 
            <python list> that contains [stenosis_flag, stenosis_location, 
            effect_region, percentage, distanceSourceToDetector, 
            distanceSourceToPatient, positionerPrimaryAngle, positionerSecondaryAngle]
        inforSaveDir: <python string>, the directory of saving the information
        csvName: <python string>, the name of the saved file
    """
    headline = ['index', 'fileName', 'stenosis_flag', 'stenosis_location', 
            'effect_region', 'percentage', 'distanceSourceToDetector', 
            'distanceSourceToPatient', 'positionerPrimaryAngle', 'positionerSecondaryAngle']
    with open(os.path.join(inforSaveDir,csvName), 'w') as fileHandle:
        fileHandle.write(','.join(headline) + '\n')
        for identifier in list(saveInfor.keys()):
            fileHandle.write(','.join(identifier))
            fileHandle.write(',')
            fileHandle.write(','.join(saveInfor[identifier])+'\n')
            
def uniformResult(*args):
    return args[0], args[1], args[2]