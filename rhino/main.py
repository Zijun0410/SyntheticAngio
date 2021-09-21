__author__ = "zijung@umich.edu"
__version__ = "2021.09.20"

import os
import random # For random generator
from configureStructure import LoadCurveFromTxt, GenerateMesh
from configureLayer import addLayer, deleteLayerObject


def readMetaData(baseDir):
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

def RandomStenosisGenerator(rng=1, position_range=(0,1), effect_range=(0.01, 0.05), percentage=(0,0.98)):
    """
    """
    stenosis_location = random.uniform(position_range[0], position_range[1])
    effect_region = random.uniform(effect_range[0], effect_range[1])
    percentage = random.uniform(percentage[0], percentage[1])
    return stenosis_location, effect_region, percentage

def main(baseDir, defaultBranchesNum):
    """

    """
    #-# Read in metaData
    metaData, recordNum = readMetaData(baseDir)

    #-# Loop through Records 
    for iRecord in range(recordNum):
        
        #-# Get meta data of a record
        fileName = metaData['FileName'][iRecord]
        positionerPrimaryAngle = metaData['PositionerPrimaryAngle'][iRecord]
        positionerSecondaryAngle = metaData['PositionerSecondaryAngle'][iRecord]
        distanceSourceToDetector = metaData['DistanceSourceToDetector'][iRecord]
        distanceSourceToPatient = metaData['DistanceSourceToPatient'][iRecord]
        
        #-# Value Check
        if distanceSourceToDetector == -1:
            distanceSourceToDetector = 1000
        if distanceSourceToPatient == -1:
            distanceSourceToDetector = 740
        
        #-# Generate Meshes
        reconstructedCurves = LoadCurveFromTxt(baseDir, defaultBranchesNum)

        stenosis_location, effect_region, percentage = RandomStenosisGenerator()
        # Random movement generator TTODO
        # reconstructedCurves = HeartMovementGenerator(reconstructedCurves)
        
        # -- vesselMeshes is a <python dict> with branch identifier as key,
        #    and <Rhino.Geometry.Mesh> as values
        vesselMeshes = GenerateVesselMesh(reconstructedCurves, stenosis_location, effect_region, percentage)

        #-# Config Angle and Receive Screen 
        visualizatioinPlane, lightVector = ConfigAngulation(positionerPrimaryAngle, positionerSecondaryAngle)
        receiveScreenPlane, receiveScreen = ConfigreceiveScreen(visualizatioinPlane, distanceSourceToPatient, 
            distanceSourceToDetector, planeSize)

        #-# Set Active Viewport for Rhino
        viewport = setView(lightVector, receiveScreenPlane, distanceSourceToPatient)

        #-# Project the receiveScreen and hatch as white. TODO

        #-# Project the stnosis point. TODO
        #-# Hatch as black TODO
        #-# Save to File TODO
        #-# Remove the object from document TODO

        #-# Project the initial point of major branch. TODO
        #-# Hatch as black TODO
        #-# Save to File TODO
        #-# Remove the object from document TODO

        #-# Loop through different meshes 
        for vessel_identifier in list(vesselMeshes.keys()):
            vesselMesh = vesselMeshes[vessel_identifier]
            # Get the outline and project 
            # Get the the countour and project 


            #-# Capture the file and save to folder
            filePath = os.path.join(baseDir, fileName, '.png'.format(identifier))
            outFilePath = CaptureView(filePath, viewport)

if( __name__ == "__main__" ):
    baseDir = r'C:\Users\gaozj\Desktop\Angio\SyntheticAngio\data'
    defaultBranchesNum = {0:'branch_4', 1:'branch_2', 2:'branch_3', 3:'major', 4:'branch_5', 5:'branch_1'}
    main(baseDir, defaultBranchesNum)