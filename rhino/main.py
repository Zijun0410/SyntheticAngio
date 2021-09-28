__author__ = "zijung@umich.edu"
__version__ = "2021.09.20"

import os
from configureStructure import RandomStenosisGenerator, GenerateVesselMesh, StenosisSphere, StartPointSphere, CrateContourCurves
from configureAngulation import ConfigAngulation, ConfigreceiveScreen, setView
from configureLayer import AddLayer, DeleteLayerObject
from configureHatch import HatchProjection, AddHatch
from configureOperationIO import LoadCurveFromTxt, ReadMetaData, GetString, CaptureViewToFile

# Some reference page 
# https://github.com/mcneel/rhinoscriptsyntax/tree/rhino-6.x/Scripts/rhinoscript
# https://developer.rhino3d.com/api/RhinoScriptSyntax/

def main(baseDir, defaultBranchesNum, batch_num, debug=False):
    """

    """
    #-# Read in metaData
    metaData, recordNum = ReadMetaData(baseDir)

    #-# Initiate Stenosis Infor Saver
    saveInfor = {}
    #-# Dir for saving the stenosis information 
    inforSaveDir = os.path.join(baseDir, 'output', batch_num)

    #-# Loop through Records 
    for iRecord in range(recordNum):
        
        #-# Get meta data of a record
        fileName = metaData['FileName'][iRecord]
        positionerPrimaryAngle = float(metaData['PositionerPrimaryAngle'][iRecord])
        positionerSecondaryAngle = float(metaData['PositionerSecondaryAngle'][iRecord])
        distanceSourceToDetector = float(metaData['DistanceSourceToDetector'][iRecord])
        distanceSourceToPatient = float(metaData['DistanceSourceToPatient'][iRecord])
        
        #-# Create save directory if not exist
        saveDir = os.path.join(baseDir, 'output', batch_num, fileName)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        #-# Value Check
        # if value does not exist (-1 by defalut), then set as the mean value
        # Code in noteboook/MetaData_Handling.ipynb
        if distanceSourceToDetector == -1:
            distanceSourceToDetector = 1040.3
        if distanceSourceToPatient == -1:
            distanceSourceToDetector = 776.5
        
        #-# Generate Meshes
        reconstructedCurves = LoadCurveFromTxt(baseDir, defaultBranchesNum)
        #-# Generate Stenosis
        # TODO: Change stenosis_flag into a random number between 0 and 1
        stenosis_flag = 1
        if stenosis_flag:
            stenosis_location, effect_region, percentage = RandomStenosisGenerator()
            # Random movement generator TODO
            # reconstructedCurves = HeartMovementGenerator(reconstructedCurves)
        else:
            stenosis_location, effect_region, percentage = 0, 0, 0

        # Update Stenosis Infor Saver
        saveInfor[(str(iRecord), fileName)] = [stenosis_flag, stenosis_location, effect_region, percentage,
            distanceSourceToDetector, distanceSourceToDetector, positionerPrimaryAngle, positionerSecondaryAngle]

        # -- vesselMeshes is a <python dict> with branch identifier as key,
        #    and <Rhino.Geometry.Mesh> as values
        vesselBreps, vesselMeshes = GenerateVesselMesh(reconstructedCurves, stenosis_location, 
            effect_region, percentage, stenosis_flag)

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
        blackRGB = (0, 0, 0)
        ## TODO: Only when there are stenosis
        # -- StenosisSphere(curveObject, stenosis_location, segmentNum=100, radius=1, create_mesh=True)
        stenosisMesh = StenosisSphere(reconstructedCurves['major'], stenosis_location)
        HatchProjection(stenosisMesh, receiveScreenPlane, viewport, blackRGB)
        #-# Save to the screenshot to file
        CaptureViewToFile(os.path.join(saveDir, 'stnosis.png'), viewport)
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
        HatchProjection(startPointMesh, receiveScreenPlane, viewport, blackRGB)
        #-# Save to File
        CaptureViewToFile(os.path.join(saveDir, 'start.png'), viewport)
        if debug:
            userInput = GetString(message="Check Output for Vessel Start Point: ")
            if userInput == "Stop": # else just continue
                return
        #-# Remove the object from layer
        DeleteLayerObject()

        #-# Loop through different meshes and breps
        for vessel_identifier in list(vesselMeshes.keys()):
            ### STEP 1 ###
            #-# Project the receiveScreen and hatch as white.
            HatchProjection(receiveScreenMesh, receiveScreenPlane, viewport, whiteRGB, offset=-1)
            #-# Get the outline of mesh, project and hatch
            vesselMesh = vesselMeshes[vessel_identifier]
            HatchProjection(vesselMesh, receiveScreenPlane, viewport, blackRGB)
            #-# Capture the file and save to folder
            filePath = os.path.join(saveDir, '{}.png'.format(vessel_identifier))
            outFilePath = CaptureViewToFile(filePath, viewport)
            if debug:
                userInput = GetString(message="Check Output for Vessel Mesh: ")
                if userInput == "Stop": # else just continue
                    return
            #-# Remove layer object from layer
            DeleteLayerObject() 
            ### STEP 2 ###
            #-# Project the receiveScreen and hatch as white.
            HatchProjection(receiveScreenMesh, receiveScreenPlane, viewport, whiteRGB, offset=-1)
            #-# Get the contour of brep, project and hatch
            vesselBrep = vesselBreps[vessel_identifier]
            contourCurves = CrateContourCurves(vesselBrep, receiveScreenPlane)            
            # -- AddHatch(curveObjects, receiveScreenPlane, colorCode, alpha)
            AddHatch(contourCurves, receiveScreenPlane, blackRGB, alpha=9)
            #-# Capture the file and save to folder
            filePath = os.path.join(saveDir, '{}_contour.png'.format(vessel_identifier))
            outFilePath = CaptureViewToFile(filePath, viewport)
            if debug:
                userInput = GetString(message="Check Output for Contour Hatch: ")
                if userInput == "Stop": # else just continue
                    return
            #-# Remove layer object from layer
            DeleteLayerObject() 

    saveStenosisInfor(saveInfor, inforSaveDir)

if( __name__ == "__main__" ):
    baseDir = r'C:\Users\gaozj\Desktop\Angio\SyntheticAngio\data'
    defaultBranchesNum = {0:'branch_4', 1:'branch_2', 2:'branch_3', 3:'major', 4:'branch_5', 5:'branch_1'}
    batch_num = '1'
    main(baseDir, defaultBranchesNum, batch_num, debug=True)