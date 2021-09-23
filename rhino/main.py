__author__ = "zijung@umich.edu"
__version__ = "2021.09.20"

import os
from configureStructure import RandomStenosisGenerator, GenerateVesselMesh, StenosisSphere, StartPointSphere
from configureAngulation import ConfigAngulation, ConfigreceiveScreen, setView
from configureLayer import AddLayer, DeleteLayerObject
from configureHatch import HatchProjection 
from configureOperationIO import LoadCurveFromTxt, ReadMetaData, GetString, CaptureViewToFile

# Some reference page 
# https://github.com/mcneel/rhinoscriptsyntax/tree/rhino-6.x/Scripts/rhinoscript
# https://developer.rhino3d.com/api/RhinoScriptSyntax/

def main(baseDir, defaultBranchesNum, debug=False):
    """

    """
    #-# Read in metaData
    metaData, recordNum = ReadMetaData(baseDir)

    #-# Loop through Records 
    for iRecord in range(recordNum):
        
        #-# Get meta data of a record
        fileName = metaData['FileName'][iRecord]
        positionerPrimaryAngle = float(metaData['PositionerPrimaryAngle'][iRecord])
        positionerSecondaryAngle = float(metaData['PositionerSecondaryAngle'][iRecord])
        distanceSourceToDetector = float(metaData['DistanceSourceToDetector'][iRecord])
        distanceSourceToPatient = float(metaData['DistanceSourceToPatient'][iRecord])
        
        #-# Value Check
        # TODO: Get the mean value 
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
        receiveScreenPlane, receiveScreenMesh = ConfigreceiveScreen(visualizatioinPlane, distanceSourceToPatient, 
            distanceSourceToDetector, planeSize=130)

        #-# Set Active Viewport for Rhino
        viewport = setView(lightVector, receiveScreenPlane, distanceSourceToPatient)

        #-# Set up layers
        AddLayer()

        #-# Project the receiveScreen and hatch as white.
        whiteRGB = (255, 255, 255)
        # -- HatchProjection(targetMesh, receiveScreenPlane, viewport, colorCode, alpha=255, offset=0)
        HatchProjection(receiveScreenMesh, receiveScreenPlane, viewport, whiteRGB, offset=-1)
        #-# Project the stnosis point and hatch as black
        blackRGB = (255, 255, 255)
        # -- StenosisSphere(curveObject, stenosis_location, segmentNum=100, radius=1, create_mesh=True)
        stenosisMesh = StenosisSphere(reconstructedCurves['major'], stenosis_location)
        HatchProjection(stenosisMesh, receiveScreenPlane, viewport, blackRGB)
        #-# Save to the screenshot to file
        CaptureViewToFile(os.path.join(baseDir, fileName, 'stnosis.png'), viewport)
        if debug:
            userInput = GetString(message="Check Output for Stenosis Point: ")
            if userInput == "Stop": # else just continue
                return
        #-# Remove the object from the layer
        DeleteLayerObject()

        #-# Project the receiveScreen and hatch as white.
        HatchProjection(receiveScreenMesh, receiveScreenPlane, viewport, whiteRGB, offset=-1)
        #-# Project the start point of major branch and hatch as black
        startPointMesh = StartPointSphere(reconstructedCurves['major'])
        HatchProjection(stenosisMesh, receiveScreenPlane, viewport, blackRGB)
        #-# Save to File
        CaptureViewToFile(os.path.join(baseDir, fileName, 'start.png'), viewport)
        if debug:
            userInput = GetString(message="Check Output for Vessel Start Point: ")
            if userInput == "Stop": # else just continue
                return
        #-# Remove the object from layer
        DeleteLayerObject()

        #-# Loop through different meshes 
        for vessel_identifier in list(vesselMeshes.keys()):
            #-# Project the receiveScreen and hatch as white.
            HatchProjection(receiveScreenMesh, receiveScreenPlane, viewport, whiteRGB, offset=-1)
            # Get the outline of mesh, project and hatch
            vesselMesh = vesselMeshes[vessel_identifier]
            HatchProjection(vesselMesh, receiveScreenPlane, viewport, blackRGB)
            #-# Capture the file and save to folder
            filePath = os.path.join(baseDir, fileName, '.png'.format(identifier))
            outFilePath = CaptureViewToFile(filePath, viewport)
            if debug:
                userInput = GetString(message="Check Output for Vessel Mesh: ")
                if userInput == "Stop": # else just continue
                    return
            #-# Remove layer object from layer
            DeleteLayerObject() 

if( __name__ == "__main__" ):
    baseDir = r'C:\Users\gaozj\Desktop\Angio\SyntheticAngio\data'
    defaultBranchesNum = {0:'branch_4', 1:'branch_2', 2:'branch_3', 3:'major', 4:'branch_5', 5:'branch_1'}
    main(baseDir, defaultBranchesNum, debug=True)