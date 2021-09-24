__author__ = "gaozj"
__version__ = "2021.09.16"

import scriptcontext
import Rhino
import System
import random # For random generator

def AddPipe(curve_object, parameters, radii, blend_type=1, cap=1, fit=True):
    # An modificatiioin for the code from the following source
    # https://github.com/mcneel/rhinoscriptsyntax/blob/rhino-6.x/Scripts/rhinoscript/surface.py

    """Creates a single walled surface with a circular profile around a curve
    Parameters:
        curve_object: <Rhino.Geometry.Curve> the rail curve
        parameters, radii: ([float, ...]), list of radius values at normalized curve parameters
        blend_type: (int, optional), 0(local) or 1(global), The shape blending. 
           Local (pipe radius stays constant at the ends and changes more rapidly in the middle)
           Global (radius is linearly blended from one end to the other, creating pipes that 
                taper from one radius to the other)
        cap: (int, optional), 0(none), 1(flat), 2(round)
        fit: (bool, optional), attempt to fit a single surface
    Returns:
        breps: <Rhino.Geometry.Brep> the created Brep objects  
      
    """
    abs_tol = scriptcontext.doc.ModelAbsoluteTolerance
    ang_tol = scriptcontext.doc.ModelAngleToleranceRadians
    if type(parameters) is int or type(parameters) is float: parameters = [parameters]
    if type(radii) is int or type(radii) is float: radii = [radii]
    parameters = map(float,parameters)
    radii = map(float,radii)
    cap = System.Enum.ToObject(Rhino.Geometry.PipeCapMode, cap)
    breps = Rhino.Geometry.Brep.CreatePipe(curve_object, parameters, radii, blend_type, cap, fit, abs_tol, ang_tol)
    # rc = [scriptcontext.doc.Objects.AddBrep(brep) for brep in breps]
    # scriptcontext.doc.Views.Redraw()
    if len(breps)==0:
        print('Error')
    return breps[0]


def get_radii(target_point, positions, baseline_radii_major):
    # TODO: Sometime the ref_radii is not there
    for index, pos in enumerate(positions):
        if pos <= target_point < positions[index+1]:
            relative_indice = (index, index+1)
            ref_position = positions[index:index+2]
            ref_radii = baseline_radii_major[index:index+2]
    target_radii = ref_radii[0] - (target_point - ref_position[0])*(ref_radii[0] - ref_radii[1])/(ref_position[1] - ref_position[0])
    return target_radii

def RandomStenosisGenerator(rng=1, position_range=(0,1), effect_range=(0.01, 0.05), percentage=(0,0.98)):
    """
    """
    stenosis_location = random.uniform(position_range[0], position_range[1])
    effect_region = random.uniform(effect_range[0], effect_range[1])
    percentage = random.uniform(percentage[0], percentage[1])
    return stenosis_location, effect_region, percentage

def GenerateStenosis(stenosis_location, effect_region, percentage, baseline_radii_major):
    """
    Generate stenosis on the major vessel curve 
    Input:
        baseline_radii_major: a list of floating point numbers, representing the baseline  
                        radius of the major vessl curve when there are no stenosis
        stenosis_location: float between (0, 1), representing the point where stenosis locate
        effect_region: float the effect region of stenosis
        percentage: float between (0, 1), %DS, a classcial measure of stenosis
    """
    updated_radii_major = baseline_radii_major[:]

    ref_point = len(baseline_radii_major)
    positions_param_prep = list(range(0,ref_point,1))
    # Becareful for using division
    positions_param = [round(float(i)/(ref_point-1),2) for i in positions_param_prep]
    
    stenosis_region_start = stenosis_location - effect_region
    stenosis_region_end = stenosis_location + effect_region
    
    # Obtain the indice of the reference points that fall into the stenosis range
    indice_within_region = []
    # Obtain the indice of the reference points that closest to the stenosis point
    for index, pos in enumerate(positions_param):
        if stenosis_region_start <= pos <= stenosis_region_end:
            indice_within_region.append(index)
        if pos <= stenosis_location < positions_param[index+1]:
            stenosis_relative_indices = (index, index+1)
            
    # print(indice_within_region)
    start_radii = round(get_radii(stenosis_region_start, positions_param, baseline_radii_major),2)
    stenosis_radii = round((1-percentage)*get_radii(stenosis_location, positions_param, baseline_radii_major),2)
    end_radii = round(get_radii(stenosis_region_end, positions_param, baseline_radii_major),2)

    if len(indice_within_region) == 0:
        # None of the original reference points fall into the stenosis region, 
        # to create a stenosis, add three additional reference points to the 
        # point list at the position of the stenosis_relative_indices
        ref_index = stenosis_relative_indices[1]
        positions_param[ref_index:ref_index] = [stenosis_region_start, stenosis_location, stenosis_region_end]   
        updated_radii_major[ref_index:ref_index] = [start_radii, stenosis_radii, end_radii]
    else:
        # One or more of the original reference points fall into the stenosis region, 
        # replace them with the stenosis region reference points
        positions_param[min(indice_within_region):max(indice_within_region)+1] = [stenosis_region_start, stenosis_location, stenosis_region_end]
        updated_radii_major[min(indice_within_region):max(indice_within_region)+1] = [start_radii, stenosis_radii, end_radii]
    positions_param_out = [round(position,2) for position in positions_param]

    return positions_param_out, updated_radii_major

def GenerateVesselMesh(reconstructedCurves, stenosis_location=0.3, effect_region=0.02, percentage=0.8,
    baseline_radii_major=[1.8,1.6,1.5,1.45,1.4,1.35,1.3,1.25,1.1,0.9,0.7,0.5,0.3],
    baseline_radii_middle=[0.78, 0.69, 0.51, 0.21], baseline_radii_small=[0.74, 0.6, 0.47, 0.17],
    baseline_radii_minor=[0.62, 0.46, 0.45, 0.16], dedault_position_param=[0, 0.3, 0.63, 1]):
    """
    Generate meshes based on vessel curves rail by 
        1. create pipe brep from the vessel curve rail, 
        2. trim out the overlapping surface between major vessels and small vessels
        3. turn all the berps into meshes
    Inputs:
        reconstructedCurves: <python dict>, stores the vessel curve rail, with <python string> as key
           and <Rhino.Geometry.Curve> as value
        stenosis_loacation, effect_region, percentage: <python float>, just as their names
        basline_radii_major: <python list of float> the baseline radius of major vessel at different
           position parameter reference points.
        baseline_radii_middle, baseline_radii_small, baseline_radii_minor: <python list of float>
           the baseline radius of different grade for non-major vessels at different position 
           parameter reference points
        dedault_position_param: <python list of float> the default position paramter for non-major branches 
    """    
    nonMajorMatchRadii = {'branch_4':baseline_radii_middle, 'branch_5':baseline_radii_middle, 
    'branch_2':baseline_radii_small, 'branch_3':baseline_radii_small, 
    'branch_1':baseline_radii_minor}

    # Iterate through branch and created pipe Brep
    vesselBreps = {}
    for branch_identifier in list(reconstructedCurves.keys()):
        # Get the positions_param and positions_ra7dii under different settings
        if branch_identifier == 'major':
            # -- GenerateStenosis(stenosis_location, effect_region, percentage, baseline_radii_major)
            positions_param, positions_radii = GenerateStenosis(stenosis_location, 
                effect_region, percentage,baseline_radii_major)
        else:
            positions_param = dedault_position_param
            positions_radii = nonMajorMatchRadii[branch_identifier]
        # Construct the Pipe brep
        vesselBreps[branch_identifier] = AddPipe(reconstructedCurves[branch_identifier], positions_param, positions_radii)

    # Cut all the small branches with the main branch and turn the output into a mesh
    majorBrep = vesselBreps['major']
    allIdentifiers = list(vesselBreps.keys())
    allIdentifiers.remove('major')
    nonMajorIdentifiers = allIdentifiers[:] # notice: python 2.x does not have the list.copy() method
    defaultMeshParams = Rhino.Geometry.MeshingParameters.Default 
    vesselMeshes = {}
    meshArrayMajor = Rhino.Geometry.Mesh.CreateFromBrep(majorBrep, defaultMeshParams)
    vesselMeshes['major'] = meshArrayMajor[0]

    for nonMajorIdentifier in nonMajorIdentifiers:
        branchBrep = vesselBreps[nonMajorIdentifier]
        # https://developer.rhino3d.com/5/api/RhinoCommon/html/M_Rhino_Geometry_Brep_Split.htm
        # https://searchcode.com/total-file/16042741/
        tol = scriptcontext.doc.ModelAbsoluteTolerance
        cuttedBrep = branchBrep.Split(majorBrep, tol)
        keptBrep = cuttedBrep[0]
        # https://developer.rhino3d.com/api/RhinoCommon/html/M_Rhino_Geometry_Mesh_CreateFromBrep_1.htm
        meshArrayNonMajor = Rhino.Geometry.Mesh.CreateFromBrep(keptBrep, defaultMeshParams)
        vesselMeshes[nonMajorIdentifier] = meshArrayNonMajor[0]

    return vesselMeshes

def DivideCurve(curveObject, segmentNum, return_points=True):
    """Helper function customized from the following link
    https://github.com/mcneel/rhinoscriptsyntax/blob/rhino-6.x/Scripts/rhinoscript/curve.py
    Divides a curve object into a specified number of segments.
    Inputs:
      curveObject: <Rhino.Geometry.Curve> the curve object to be divided
      segmentNum: <python int> The number of segments.
      return_points: <python bool, optional> If omitted or True, points are returned.
          If False, then a list of curve parameters are returned.
    Returns:
      list(point|number, ...): If `return_points` is not specified or True, then a list containing 3D division points.
      list(point|number, ...): If `return_points` is False, then an array containing division curve parameters.
      None: if not successful, or on error.
    """
    # curveParams are the division curve parameters 
    curveParams = curveObject.DivideByCount(segmentNum, True)
    if return_points:
        dividedPoints = [curveObject.PointAt(t) for t in curveParams]
        return dividedPoints
    return curveParams

def StenosisSphere(curveObject, stenosis_location, segmentNum=100, radius=1, create_mesh=True):
    """
    Create a sphere at the stenosis location on the vessel curve
    Inputs:
        curveObject: <Rhino.Geometry.Curve> the curve object that we want to loacate the stenosis 
            point as the sphere center. 
        stenosis_location: <python float of two digits>, represents the location of stenosis
            from the start(0) and end(1) points of the vessel.
        radius: <python float, optional>, the radius of the sphere created
        create_mesh: <python bool, optional> If omitted or True, return <Rhino.Geometry.Mesh>, 
            otherwise return <Rhino.Geometry.Sphere>
    Outputs:
        stenosisMesh or stenosisBrep depend on the input
    """
    dividedPoints = DivideCurve(curveObject, segmentNum)
    stenosisPoint = dividedPoints[int(segmentNum*stenosis_location)]
    stenosisSphere = Rhino.Geometry.Sphere(stenosisPoint, radius)
    stenosisBrep = stenosisSphere.ToBrep()
    if create_mesh:
        defaultMeshParams = Rhino.Geometry.MeshingParameters.Default
        stenosisMesh = Rhino.Geometry.Mesh.CreateFromBrep(stenosisBrep, defaultMeshParams)
        return stenosisMesh[0]
    return stenosisBrep

def StartPointSphere(curveObject, radius=1, create_mesh=True):
    """
    Create a sphere at the start point on the vessel curve.
    Inputs:
        curveObject: <Rhino.Geometry.Curve> the curve object that we want to loacate the 
            start point as the sphere center.
        radius: <python float, optional>, the radius of the sphere created
        create_mesh: <python bool, optional> If omitted or True, return <Rhino.Geometry.Mesh>, 
            otherwise return <Rhino.Geometry.Sphere>
    Outputs:
        stenosisMesh or stenosisSphere depend on the input
    """
    pointAtStart = curveObject.PointAtStart
    startPointSphere = Rhino.Geometry.Sphere(pointAtStart, radius)
    startPointBrep = startPointSphere.ToBrep()
    if create_mesh:
        defaultMeshParams = Rhino.Geometry.MeshingParameters.Default
        startPointMesh = Rhino.Geometry.Mesh.CreateFromBrep(startPointBrep, defaultMeshParams)
        return startPointMesh[0]
    return startPointBrep    


if( __name__ == "__main__" ):
    import os
    from configureOperationIO import LoadCurveFromTxt
    baseDir = r'C:\Users\gaozj\Desktop\Angio\SyntheticAngio\data'
    defaultBranchesNum = {0:'branch_4', 1:'branch_2', 2:'branch_3', 3:'major', 4:'branch_5', 5:'branch_1'}
    reconstructedCurves = LoadCurveFromTxt(baseDir, defaultBranchesNum)
    vesselMeshes = GenerateVesselMesh(reconstructedCurves)
