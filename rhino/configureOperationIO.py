__author__ = "zijung@umich.edu"
__version__ = "2021.09.22"

import os
import Rhino
import System
from configureAngulation import viewportByName

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

def ReadMetaData(baseDir):
    """
    Read in the csv file and store all the relevant information in a dictionary
    Inputs:
        baseDir: <python string> a base dir where all the meta data are saved
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
    with open(os.path.join(baseDir,'meta_summary.csv'), 'r') as fileHandle:
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

def LoadCurveFromTxt(baseDir, defaultBranches):
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
        with open(os.path.join(baseDir,'{}.txt'.format(branchIdx)), 'r') as fileHandle:
            linesIn = fileHandle.readlines()
            for line in linesIn:
                locations = line.strip('\n').split(', ')
                # Turn into list of floats
                locations = list(map(float,locations))
                pintList.append(Rhino.Geometry.Point3d(locations[0], locations[1], locations[2]))
        identifier = defaultBranches[branchIdx]
        reconstructedCurves[identifier] = Rhino.Geometry.Curve.CreateControlPointCurve(pintList)
    return reconstructedCurves


def CaptureViewToFile(filePath, viewport, width=1896, height=1127, displayMode='Shaded', transparent=True):

    """
    Capture a Viewport to a PNG file path.
    Args:
        filePath: Full path to the file where the image will be saved.
        viewport: An active 'Perspective' viewport that has its display mode set
        width: Integer for the image width in pixels. If None, the width of the
            active viewport will be used. (Default: 1100).
        height: Integer for the image height in pixels. If None, the height of the
            active viewport will be used. (Default: 1100).
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
            distanceSourceToDetector, positionerPrimaryAngle, positionerSecondaryAngle]
        inforSaveDir: <python string>, the directory of saving the information
        csvName: <python string>, the name of the saved file
    """
    headline = ['index', 'fileName', 'stenosis_flag', 'stenosis_location', 
            'effect_region', 'percentage', 'distanceSourceToDetector', 
            'distanceSourceToDetector', 'positionerPrimaryAngle', 'positionerSecondaryAngle']
    with open(os.path.join(inforSaveDir,csvName), 'w') as fileHandle:
        fileHandle.write(','.join(headline) + '\n')
        for identifier in list(saveInfor.keys()):
            fileHandle.write(','.join(identifier))
            fileHandle.write(','.join(saveInfor[identifier])+'\n')
            
def uniformResult(*args):
    return args[0], args[1], args[2]