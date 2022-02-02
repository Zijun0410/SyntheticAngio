__author__ = "zijung@umich.edu"
__version__ = "2021.10.25"

import os
from configureStructure import RandomStenosisGenerator, GenerateVesselMesh, StenosisSphere, StartPointSphere, CrateContourCurves, HeartMovementGenerator
from configureAngulation import ConfigAngulation, ConfigreceiveScreen, setView
from configureLayer import AddLayer, DeleteLayerObject
from configureHatch import HatchProjection, AddHatch
from configureOperationIO import CaptureViewToFile, saveStenosisInfor, uniformResult, LoadCurveFromTxt, ReadMetaData, MetaValueCheck, GetString
import rhinoscriptsyntax as rs

# Some reference page 
# https://github.com/mcneel/rhinoscriptsyntax/tree/rhino-6.x/Scripts/rhinoscript
# https://developer.rhino3d.com/api/RhinoScriptSyntax/

def main(baseDir, defaultBranchesNum, batch_id, load_folder, adjust=False, debug=False, limit=False, side='R'):
    """
    Load metadata from baseDir/Meta_Data/batch_id
    Save Rhino output to baseDir/Rhino_Output/batch_id

    """
    #-# Read in metaData
    if 'Debug' in batch_id or 'UoMR' in batch_id:
        meta_file = 'UoM_Right_endpoint.csv'
    else:
        meta_file = 'meta_summary.csv'
        
    metaData, recordNum = ReadMetaData(baseDir, meta_file)

    #-# Initiate Stenosis Infor Saver
    saveInfor = {}
    #-# Dir for saving the stenosis information 
    inforSaveDir = os.path.join(baseDir, 'Rhino_Output', batch_id)

    #-# Loop through Records 
    for iRecord in range(recordNum):
        
        metaInfor, fileName = MetaValueCheck(metaData, iRecord, side)
        positionerPrimaryAngle = metaInfor[0]
        positionerSecondaryAngle = metaInfor[1]
        distanceSourceToDetector = metaInfor[2]
        distanceSourceToPatient = metaInfor[3]
        
        #-# Create save directory if not exist
        saveDir = os.path.join(baseDir, 'Rhino_Output', batch_id, fileName)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        
        #-# Generate Meshes
        reconstructedCurves, pointMaxDistance, pointMinDistance = uniformResult(*LoadCurveFromTxt(baseDir, defaultBranchesNum[load_folder], load_folder))
        # print(pointMaxDistance, pointMinDistance)

        #-# Generate Stenosis
        success = 0
        while success == 0:
            stenosis_num = 1
            if stenosis_num:
                stenosis_location, effect_region, percentage = RandomStenosisGenerator()
                # Random heart movement generator
                reconstructedCurves = HeartMovementGenerator(reconstructedCurves, defaultBranchesNum[load_folder], pointMaxDistance, pointMinDistance)
            else:
                stenosis_location, effect_region, percentage = 0, 0, 0
    
            # Update Stenosis Infor Saver
            infor_list = [stenosis_num, stenosis_location, effect_region, percentage,
                distanceSourceToPatient, distanceSourceToDetector, positionerPrimaryAngle, positionerSecondaryAngle]

            saveInfor[(str(iRecord), fileName)] = [str(i) for i in infor_list]
    
            # -- vesselMeshes is a <python dict> with branch identifier as key,
            #    and <Rhino.Geometry.Mesh> as values
            try:
                vesselBreps, vesselStartBreps, vesselMeshes = uniformResult(*GenerateVesselMesh(reconstructedCurves, 
                    stenosis_location, effect_region, percentage, stenosis_num))
                success = 1
            except:
                print("Error!!!!")
                print(stenosis_location,effect_region, percentage)
                continue
                # (0.38267169404413426, 0.034892475013214311, 0.33864133655537093)
                # (0.11923939109000947, 0.04912465119884836, 0.35551291328558277)
                # (0.98854180569293626, 0.035962558490046342, 0.37653743549849805)
                # (0.52075417677609082, 0.016475121364722574, 0.22654419474552556)
                # (0.47372991474187498, 0.023448820559792589, 0.11662228312921692)
                # (0.96682662659530982, 0.049581884697998735, 0.66673771498717416)
                # (0.98281510714822551, 0.038127970332965597, 0.89889744161188434)
                
        #-# Config Angle and Receive Screen 
        visualizatioinPlane, lightVector, Zplus = uniformResult(*ConfigAngulation(positionerPrimaryAngle, positionerSecondaryAngle))
        receiveScreenPlane, receiveScreenMesh = ConfigreceiveScreen(visualizatioinPlane, distanceSourceToPatient, 
            distanceSourceToDetector, planeSize=120)

        #-# Set Active Viewport for Rhino
        viewport = setView(lightVector, receiveScreenPlane, distanceSourceToPatient, Zplus)

        #-# Set up layers
        AddLayer()

        #-# Project the receiveScreen and hatch as white.
        whiteRGB = (255, 255, 255)
        # -- HatchProjection(targetMesh, receiveScreenPlane, viewport, colorCode, alpha=255, offset=0)
        HatchProjection(receiveScreenMesh, receiveScreenPlane, viewport, whiteRGB, offset=-1)
        if adjust:
            return 
        else:
            CaptureViewToFile(os.path.join(saveDir, 'view.png'), viewport, transparent=False)
        #-# Project the stnosis point and hatch as black
        blackRGB = (0, 0, 0)
        
        # -- StenosisSphere(curveObject, stenosis_location, segmentNum=100, radius=1, create_mesh=True)
        stenosisMesh = StenosisSphere(reconstructedCurves['major'], stenosis_location)
        HatchProjection(stenosisMesh, receiveScreenPlane, viewport, blackRGB)
        #-# Save to the screenshot to file
        CaptureViewToFile(os.path.join(saveDir, 'stnosis_{}.png'.format(stenosis_num)), viewport)
        # if debug:
        #     userInput = GetString(message="Check Output for Stenosis Point")
        #     if userInput == "n": # else just continue
        #         return
        #-# Remove the object from the layer
        DeleteLayerObject()

        #-# Project the receiveScreen and hatch as white.
        HatchProjection(receiveScreenMesh, receiveScreenPlane, viewport, whiteRGB, offset=-1)
        #-# Project the start point of major branch and hatch as black
        startPointMesh = StartPointSphere(reconstructedCurves['major'])
        HatchProjection(startPointMesh, receiveScreenPlane, viewport, blackRGB)
        #-# Save to File
        CaptureViewToFile(os.path.join(saveDir, 'start.png'), viewport)
        # if debug:
        #     userInput = GetString(message="Check Output for Vessel Start Point")
        #     if userInput == "n": # else just continue
        #         return
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
            # if debug:
            #     userInput = GetString(message="Check Output for Vessel Mesh")
            #     if userInput == "n": # else just continue
            #         return
            #-# Remove layer object from layer
            DeleteLayerObject() 

            ### STEP 2 ###
            #-# Project the receiveScreen and hatch as white.
            HatchProjection(receiveScreenMesh, receiveScreenPlane, viewport, whiteRGB, offset=-1)
            #-# Get the contour of brep, project and hatch
            vesselBrep = vesselBreps[vessel_identifier]
            contourCurves = CrateContourCurves(vesselBrep, receiveScreenPlane)            
            # -- AddHatch(curveObjects, receiveScreenPlane, colorCode, alpha)
            AddHatch(contourCurves, receiveScreenPlane, blackRGB, alpha=11)
            if vessel_identifier!='major':
                # Try to solve the problem that the minor vessel start portion is barely visable.
                vesselStartBrep = vesselStartBreps[vessel_identifier]
                contourCurvesBall = CrateContourCurves(vesselStartBrep, receiveScreenPlane, interval=0.4) 
                AddHatch(contourCurvesBall, receiveScreenPlane, blackRGB, alpha=4)

            ### STEP 3 ###
            #-# Capture the file and save to folder
            filePath = os.path.join(saveDir, '{}_contour.png'.format(vessel_identifier))
            if debug:
                userInput = GetString(message="Check Output for Contour Hatch")
                if userInput == "n": # else just continue
                    return
            # The normal capture via viewport.ParentView.CaptureToBitmap is unable to capture the
            # shaded effect of transparent project. Therefore, we use the 'ViewCaptureToFile'
            cmd = " _TransparentBackgroud _Yes _DrawGrid _No _Enter"
            rs.Command("-ViewCaptureToFile " + chr(34) + filePath + chr(34)+ cmd)
            # discarded: outFilePath = CaptureViewToFile(filePath, viewport)
            #-# Remove layer object from layer
            DeleteLayerObject() 
        if limit and iRecord == 0:
            break

        if iRecord%100==0:
            saveStenosisInfor(saveInfor, inforSaveDir)

    saveStenosisInfor(saveInfor, inforSaveDir)

if( __name__ == "__main__" ):
    # baseDir = r'C:\Users\gaozj\Desktop\Angio\SyntheticAngio\data'
    baseDir = r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic'
    defaultBranches = dict()
    defaultBranches['RCA_Brief'] = {0:'branch_4', 1:'branch_2', 2:'branch_3', 3:'major', 4:'branch_5', 5:'branch_1'}
    defaultBranches['RCA_Detail_2'] = {11:'major',0:'branch_1', 1:'branch_2', 2:'branch_3', 3:'branch_4', 
                                12:'branch_5', 16:'branch_6', 4:'sub_1', 5:'sub_2', 6:'sub_3', 7:'subsub_1', 
                                8:'sub_4', 9:'sub_5', 10:'sub_6',  13:'sub_7', 14:'sub_8', 15:'sub_9', 
                                17:'sub_10', 18:'sub_11'}
    defaultBranches['RCA_Detail_1'] = {3:'major', 0:'branch_1', 1:'branch_2', 2:'branch_3', 4:'branch_4', 
                                5:'branch_5', 14: 'branch_6', 16: 'branch_7', 6:'sub_1', 8:'sub_2', 9:'sub_3',
                                10:'sub_4', 11:'sub_5', 12:'sub_6', 15:'sub_7', 18:'sub_8', 19:'sub_9',
                                21:'sub_10', 22:'sub_11', 23:'sub_12', 24:'sub_13', 25:'sub_14',
                                7:'subsub_1', 10:'subsub_2', 13:'subsub_3', 17:'subsub_4', 20:'subsub_5'}

    batch_id = 'UKR' # Choose from {'Debug', 'UKR', 'UoMR'}
    main(baseDir, defaultBranches, batch_id, folder='RCA_Detail_1', adjust=True, debug=False, limit=False)