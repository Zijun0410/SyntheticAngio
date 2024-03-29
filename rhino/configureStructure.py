__author__ = "zijung@umich.edu"
__version__ = "2021.09.16"

import scriptcontext
import Rhino
import System
import random # For random generator
import math

from configureAngulation import ISOCENTER

def AddPipe(curve_object, parameters, radii, cap=1, blend_type=1, fit=True):
    # An modificatiioin for the code from the following source
    # https://github.com/mcneel/rhinoscriptsyntax/blob/rhino-6.x/Scripts/rhinoscript/surface.py

    """
    Creates a single walled surface with a circular profile around a curve
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

def RandomStenosisGenerator(position_range=(0,0.8), effect_range=(0.01, 0.05), 
    percentages=[(0.2, 0.5),(0.5, 0.7),(0.6, 0.7),(0.7, 0.9)]):
    """
    Randomly generate stenosis related-parameter
    Input:
        position_range: <python tuple>, a stenosis position range to choose from. 
            0 map to the start point of major vesel. 1 maps to the end point.
        effect_range: <python tuple>, a stenosis effect range to choose from.
        percentages: <python list of tuples>, a percentage range to choose from.
    Ouutput:
        stenosis_location, effect_region, percentage
    """
    stenosis_location = random.uniform(position_range[0], position_range[1])
    effect_region = random.uniform(effect_range[0], effect_range[1])
    percentage_range = random.choice(percentages)
    percentage = random.uniform(percentage_range[0], percentage_range[1])
    return stenosis_location, effect_region, percentage


def HeartMovementGenerator(reconstructedCurves, defaultBranches, pointMaxDistance, pointMinDistance, min_scale=0.8, max_scale=1.2):
    """
    Generate twists of vessel centerline to mimic the effect heart movements
    Pseudo Code
        input: time from $0$ to $pi$, the isocenter
        for points on a polyline
            calculate the `distance` between a point and the isocenter
            form a `vector` from isocenter to point
            extend the `vector` by a `distance` factor and a time (cos) factor
            obtain the `endpoint` and add to newpoints
        reconstruct a polyline based on newpoints  
    Inputs:
        min_scale <python float>, the minimum adjust scale
        max_scale <python float>, the maximum adjust scale
        pointMaxDistance <python float>, the maximum distance from point on vessel line to isocenter
        pointMinDistance <python float>, the minimum distance from point on vessel line to isocenter
        defaultBranches <python dict>, the branch index and identifier 
        reconstructedCurves <python dict>, the branch identifier and Rhino.Geometry.Curve
    """
    random_time = random.uniform(0, math.pi)
    isocenter = Rhino.Geometry.Point3d(ISOCENTER[0], ISOCENTER[1], ISOCENTER[2])
    scale_range = max_scale - min_scale
    maxDistance = math.ceil(pointMaxDistance)
    minDistance = math.floor(pointMinDistance)
    med_value = round((minDistance/maxDistance +1)/2,2)
    sign = lambda x: (1, -1)[x<1] # return 1 when value bigger than 1, -1 otherwise
    adjust_func = lambda x, y: math.e**(sign(y)*(med_value-x))

    reconstructedModeCurves = {}
    for identifier in list(defaultBranches.values()):
        curveObject = reconstructedCurves[identifier]
        dividedPoints = DivideCurve(curveObject, segmentNum=80)
        rescaledPoints = []
        for pointObject in dividedPoints:
            dist = isocenter.DistanceTo(pointObject)
            # scale_factor is a value in range (min_scale, max_scale)
            scale_factor = min_scale + math.sin(random_time)*scale_range
            # adjust_factor adjust the scale_factor by multiplying a value defiend by adjust_func
            adjust_factor = scale_factor*(adjust_func(dist/maxDistance, scale_factor))
            ptVector = (pointObject - isocenter)*adjust_factor
            adjust_endpoint = isocenter + ptVector
            rescaledPoints.append(adjust_endpoint)
        updateCurve = Rhino.Geometry.Curve.CreateControlPointCurve(rescaledPoints)
        domain = Rhino.Geometry.Interval(0, 1)
        updateCurve.Domain = domain
        reconstructedModeCurves[identifier] = updateCurve
    return reconstructedModeCurves

def GenerateStenosis(stenosis_location, effect_region, percentage, baseline_radii_major):
    """
    Generate stenosis on the major vessel curve 
    Input:
        baseline_radii_major: a list of floating point numbers, representing the baseline  
                        radius of the major vessl curve when there are no stenosis
        stenosis_location: float between (0, 1), representing the point where stenosis locate
        effect_region: float the effect region of stenosis
        percentage: float between (0, 1), %DS, a classcial measurement of stenosis
    """
    updated_radii_major = baseline_radii_major[:]

    ref_point = len(baseline_radii_major)
    positions_param_prep = list(range(0,ref_point,1))
    # Be careful when using division in python 2.7
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

def GenerateVesselMesh(reconstructedCurves, stenosis_location=0.3, 
    effect_region=0.02, percentage=0.8, stenosis_flag=1, 
    baseline_radii_major=[1.8,1.6,1.5,1.45,1.4,1.35,1.3,1.25,1.1,0.9,0.7,0.5,0.3],
    baseline_radii_middle=[0.78,0.69,0.51,0.21], baseline_radii_small=[0.74,0.6,0.47,0.17],
    baseline_radii_minor=[0.62,0.46,0.45,0.16], dedault_position_param=[0,0.3,0.63,1]):
    """
    Generate meshes based on vessel curves rail by 
        1. create pipe brep from the vessel curve rail, 
        2. trim out the overlapping surface between major vessels and small vessels
        3. turn all the berps into meshes
    Inputs:
        reconstructedCurves: <python dict>, stores the vessel curve rail, with <python string> as key
           and <Rhino.Geometry.Curve> as value
        stenosis_flag: <python boolen>, generate major vessel stenosis when set True.
        return_brep: <python boolen>, return brep instead of mesh when set True.   
        stenosis_loacation, effect_region, percentage: <python float>, just as their names
        basline_radii_major: <python list of float> the baseline radius of major vessel at different
           position parameter reference points.
        baseline_radii_middle, baseline_radii_small, baseline_radii_minor: <python list of float>
           the baseline radius of different grade for non-major vessels at different position 
           parameter reference points
        dedault_position_param: <python list of float> the default position paramter for non-major branches 
    Outputs:
        vesselBreps and vesselMeshes: <python dict> with key of identifier and value of <Rhino.Geometry.Brep>
           or <Rhino.Geometry.Mesh>
    """    
    nonMajorMatchRadii = {'branch_4':baseline_radii_middle, 'branch_5':baseline_radii_middle, 
    'branch_2':baseline_radii_small, 'branch_3':baseline_radii_small, 'branch_1':baseline_radii_minor}

    # Iterate through branch and created pipe Brep
    preVesselBreps = {}
    vesselStartBreps = {}
    for branch_identifier in list(reconstructedCurves.keys()):
        # Get the positions_param and positions_radii under different settings
        if branch_identifier == 'major':
            if stenosis_flag > 0:
                # -- GenerateStenosis(stenosis_location, effect_region, percentage, baseline_radii_major)
            # try:
                positions_param, positions_radii = GenerateStenosis(stenosis_location, 
                    effect_region, percentage, baseline_radii_major)
            # except:
            #    print(stenosis_location, effect_region, percentage)
            #    continue
            else:
                # No stenosis
                positions_radii = baseline_radii_major
                ref_point = len(baseline_radii_major)
                positions_param = [round(float(i)/(ref_point-1),2) for i in list(range(0,ref_point,1))]
        else:
            positions_param = dedault_position_param
            positions_radii = nonMajorMatchRadii[branch_identifier]
        # Construct the Pipe brep
        # try:
        preVesselBreps[branch_identifier] = AddPipe(reconstructedCurves[branch_identifier], positions_param, positions_radii, cap=2)
        # except IndexError:
        #    print(stenosis_location, effect_region, percentage)
        #    continue
        vesselStartBreps[branch_identifier] = StartPointSphere(reconstructedCurves[branch_identifier], radius=positions_radii[0], create_mesh=False)
          
    #-# Prepare to update the vessel breps and the mesh to be created    
    majorBrep = preVesselBreps['major']
    vesselBreps = {}
    vesselBreps['major'] = majorBrep

    allIdentifiers = list(preVesselBreps.keys())
    allIdentifiers.remove('major')
    nonMajorIdentifiers = allIdentifiers[:] # notice: python 2.x does not have the list.copy() method
    defaultMeshParams = Rhino.Geometry.MeshingParameters.Default 
    vesselMeshes = {}
    meshArrayMajor = Rhino.Geometry.Mesh.CreateFromBrep(majorBrep, defaultMeshParams)
    vesselMeshes['major'] = meshArrayMajor[0]

    #-# Trim uncessary parts from the small branches by intercecting the main branch 
    #   and turn the output into a mesh
    for nonMajorIdentifier in nonMajorIdentifiers:
        branchBrep = preVesselBreps[nonMajorIdentifier]
        # Option 1
        # https://developer.rhino3d.com/5/api/RhinoCommon/html/M_Rhino_Geometry_Brep_Split.htm
        # https://searchcode.com/total-file/16042741/
        tol = scriptcontext.doc.ModelAbsoluteTolerance
        cuttedBrep = branchBrep.Split(majorBrep, tol)
        keptBrep = cuttedBrep[0]
        vesselBreps[nonMajorIdentifier] = keptBrep
        # https://developer.rhino3d.com/api/RhinoCommon/html/M_Rhino_Geometry_Mesh_CreateFromBrep_1.htm
        meshArrayNonMajor = Rhino.Geometry.Mesh.CreateFromBrep(keptBrep, defaultMeshParams)
        vesselMeshes[nonMajorIdentifier] = meshArrayNonMajor[0]

    # return vesselBreps, vesselStartBreps, vesselMeshes
    return preVesselBreps, vesselStartBreps, vesselMeshes

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

def CrateContourCurves(vesselBrep, receiveScreenPlane, interval=0.15):
    """
    Given a vesselBrep, create contour curves that are parallel to the receiveScreenPlane
    Inputs:
        vesselBrep: <Rhino.Geometry.Brep> the brep to contour
        receiveScreenPlane: <Rhino.Geometry.Plane> slicing plane for generatinig the base
          point and end point (that gives the contour norm direction).
        interval: <python float>, distance between two contours
    """
    base_point = receiveScreenPlane.Origin
    end_point = receiveScreenPlane.Origin + receiveScreenPlane.ZAxis*1500
    contourCurves = vesselBrep.CreateContourCurves(vesselBrep, base_point, end_point, interval)
    # https://developer.rhino3d.com/api/RhinoCommon/html/M_Rhino_Geometry_Curve_ProjectToBrep.htm
    # https://developer.rhino3d.com/api/RhinoCommon/html/M_Rhino_Geometry_Curve_ProjectToPlane.htm
    #contourCurves = [Rhino.Geometry.Curve.ProjectToPlane(curve,receiveScreenPlane) for curve in contourCurves]
    return contourCurves

if( __name__ == "__main__" ):
    import os
    from configureOperationIO import LoadCurveFromTxt
    baseDir = r'C:\Users\gaozj\Desktop\Angio\SyntheticAngio\data'
    defaultBranchesNum = {0:'branch_4', 1:'branch_2', 2:'branch_3', 3:'major', 4:'branch_5', 5:'branch_1'}
    reconstructedCurves = LoadCurveFromTxt(baseDir, defaultBranchesNum)
    vesselMeshes = GenerateVesselMesh(reconstructedCurves)
