__author__ = "gaozj"
__version__ = "2021.09.16"

import rhinoscriptsyntax as rs
import Rhino
import math
import System
import os

try:
    import scriptcontext as sc
except ImportError as e:  # No Rhino doc is available. This module is useless.
    raise ImportError("Failed to import Rhino scriptcontext.\n{}".format(e))


baseDir = r'C:\Users\gaozj\Desktop\Angio\SyntheticAngio\data'

defaultBranchesNum = {0:'branch_4', 1:'branch_2', 2:'branch_3', 3:'major', 4:'branch_5', 5:'branch_1'}
# Radius of major branch on standarized domain
baseline_radii_major = [1.8, 1.6, 1.5, 1.45, 1.4, 1.35, 1.3, 1.25, 1.1, 0.9, 0.7, 0.5, 0.3]

# Radius of non-major branches on standarized domain
baseline_radii_middle = [0.78, 0.69, 0.51, 0.21]
baseline_radii_small = [0.74, 0.6, 0.47, 0.17]
baseline_radii_minor = [0.62, 0.46, 0.45, 0.16]
# The position paramter for non-major branches 
dedault_position_param = [0, 0.3, 0.63, 1]

nonMajorMatchRadii = {'branch_4':baseline_radii_middle, 'branch_5':baseline_radii_middle, 
    'branch_2':baseline_radii_small, 'branch_3':baseline_radii_small, 
    'branch_1':baseline_radii_minor}

def loadCurveFromTxt(baseDir, defaultBranchesNum):
    """
    Load back the reconstructed curve from the text file
    Input:
        baseDir: a valid directory where the curve control points are saved
        defaultBranchNum: a dictionary of the correspondance of branch and their index 
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
                pintList.append(Rhino.Geometry.Point3d(locations[0], locations[1], locations[2]))
        identifier = defaultBranches[branchIdx]
        reconstructedCurves[identifier] = Rhino.Geometry.Curve.CreateControlPointCurve(pintList)
    return reconstructedCurves

reconstructedCurves = loadCurveFromTxt(baseDir, defaultBranchesNum)

# Iterate through branch
for branch_identifier in reconstructedCurves.keys():
    # Get the positions_param and positions_radii under different settings
    if branch_identifier == 'major':
        positions_param, positions_radii = GenerateStenosis(baseline_radii_major, 
            stenosis_point, effect_region, percentage)
    else:
        positions_param = dedault_position_param
        positions_radii = nonMajorMatchRadii
    # Construct the Pipe



def get_radii(target_point, positions, baseline_radii_major):
    for index, pos in enumerate(positions):
        if pos <= target_point < positions[index+1]:
            relative_indice = (index, index+1)
            ref_position = positions[index:index+2]
            ref_radii = baseline_radii_major[index:index+2]
    target_radii = ref_radii[0] - (target_point - ref_position[0])*(ref_radii[0] - ref_radii[1])/(ref_position[1] - ref_position[0])
    return target_radii

def GenerateStenosis(baseline_radii_major, stenosis_point, effect_region, percentage):
    """
    Generate stenosis on the major vessel curve 
    Input:
        baseline_radii_major: a list of floating point numbers, representing the baseline  
                        radius of the major vessl curve when there are no stenosis
        stenosis_point: float between (0, 1), representing the point where stenosis locate
        effect_region: float the effect region of stenosis
        percentage: float between (0, 1), %DS, a classcial measure of stenosis
    """
    ref_point = len(baseline_radii_major)
    positions_param_prep = list(range(0,ref_point,1))
    positions_param = [round(i/(ref_point-1),2) for i in positions_param_prep]
    
    stenosis_region_start = stenosis_point - effect_region
    stenosis_region_end = stenosis_point + effect_region
    
    # Obtain the indice of the reference points that fall into the stenosis range
    indice_within_region = []
    # Obtain the indice of the reference points that closest to the stenosis point
    for index, pos in enumerate(positions_param):
        if stenosis_region_start <= pos <= stenosis_region_end:
            indice_within_region.append(index)
        if pos <= stenosis_point < positions_param[index+1]:
            stenosis_relative_indices = (index, index+1)
            
    # print(indice_within_region)
    start_radii = round(get_radii(stenosis_region_start, positions_param, baseline_radii_major),2)
    stenosis_radii = round((1-percentage)*get_radii(stenosis_point, positions_param, baseline_radii_major),2)
    end_radii = round(get_radii(stenosis_region_end, positions_param, baseline_radii_major),2)

    if len(indice_within_region) == 0:
        # None of the original reference points fall into the stenosis region, 
        # to create a stenosis, add three additional reference points to the 
        # point list at the position of the stenosis_relative_indices
        ref_index = stenosis_relative_indices[1]
        positions_param[ref_index:ref_index] = [stenosis_region_start, stenosis_point, stenosis_region_end]   
        baseline_radii_major[ref_index:ref_index] = [start_radii, stenosis_radii, end_radii]
    else:
        # One or more of the original reference points fall into the stenosis region, 
        # replace them with the stenosis region reference points
        positions_param[min(indice_within_region):max(indice_within_region)+1] = [stenosis_region_start, stenosis_point, stenosis_region_end]
        baseline_radii_major[min(indice_within_region):max(indice_within_region)+1] = [start_radii, stenosis_radii, end_radii]
    positions_param_out = [round(position,2) for position in positions_param]
    return positions_param_out, baseline_radii_major



