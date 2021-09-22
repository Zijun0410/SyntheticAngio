__author__ = "zijung@umich.edu"
__version__ = "2021.09.14"

import Rhino
import math
import System
import scriptcontext

def ConfigAngulation(positionerPrimaryAngle,positionerSecondaryAngle,
    isocenter_x = 8.5, isocenter_y = -10.5, isocenter_z = 50):
    """
    This function sets up  the angulation based on 
    the primary and secondary angles, returns the direction
    where the light comes in and the visualization plane 
    that holds the projection
        Inputs:
            positionerPrimaryAngle: The primary rotation angle
            positionerSecondaryAngle: The secondary rotation angle
        Output:
            lightVector: <Rhino.Geometry.Vector3d> A vector that represents 
                the direction of the X-ray beam
            visualizatioinPlane: <Rhino.Geometry.Plane> A plane object 
                where the receiving screen holds
        """
    isocenter = Rhino.Geometry.Point3d(isocenter_x,isocenter_y,isocenter_z)
    reference_axis = Rhino.Geometry.Point3d(isocenter_x,isocenter_y-1,isocenter_z) - isocenter
    
    ## Primary Rotation Setup
    point_z_plus = Rhino.Geometry.Point3d(isocenter_x,isocenter_y,isocenter_z+1)
    primaryRotationAxis = point_z_plus - isocenter
    #-# Transform Rotation(double angleRadians,Vector3d rotationAxis,Point3d rotationCenter)
    primary_angle_rotation = Rhino.Geometry.Transform.Rotation(math.pi*positionerPrimaryAngle/180, primaryRotationAxis, isocenter)
    reference_axis.Transform(primary_angle_rotation)
    
    ## Construct the YpZ plane with three points
    YpZ_plane = Rhino.Geometry.Plane(isocenter, isocenter + reference_axis, point_z_plus)
    
    ## Secondary Rotation Setup
    secondaryRotationAxis = YpZ_plane.ZAxis
    seconday_angle_rotation = Rhino.Geometry.Transform.Rotation(math.pi*positionerSecondaryAngle/180, secondaryRotationAxis, isocenter)
    reference_axis.Transform(seconday_angle_rotation)
    
    ## Construct Visualzation plane with norm vector and Origin
    ## TODO: Set the X axis direction
    visualizatioinPlane = Rhino.Geometry.Plane(isocenter, reference_axis)
    
    ## The lightVector is on the opposite direction of reference_axis
    lightVector = -reference_axis
    
    #print(visualizatioinPlane) 
    return visualizatioinPlane, lightVector

def offsetPlane(plane, distance):
    """Helper function
    Offset the Plane along the Z-axis by certain distance
        Inputs:
            plane: <Rhino.Geometry.Plane> the original plan before offset
            distance: float, the distance where the z axis of plane move along
        Outputs:
            new_plane: <Rhino.Geometry.Plane> the new plane created
    """
    directionVector = plane.ZAxis*distance
    # print(directionVector)
    newOrigin = plane.Origin + directionVector
    newPlane = Rhino.Geometry.Plane(newOrigin, plane.ZAxis)
    return newPlane

def ConfigreceiveScreen(visualizatioinPlane, distanceSourceToPatient, 
    distanceSourceToDetector, planeSize=150):
    """
    Configure the Receive plane and screen based on the visualization plane, 
    and parameter of distance from the .dcm metadata
    Inputs:
        visualizatioinPlane: <Rhino.Geometry.Plane> A plane object 
            where the receiving screen holds
        distanceSourceToPatient: float, from metadata
        distanceSourceToDetector: float, from metadata
        planeSize: int, the size of a plane
    Output:
        receiveScreenPlane: <Rhino.Geometry.Plane> the plane that receive projection
            of 3D meshes.
        receiveScreenMesh: <Rhino.Geometry.Mesh> PlaneSurface to Brep to Mesh, used for 
           generating the hatch outline for a receive screen
    """
    # Create receiveScreenPlane
    distanceIsometricToScreen = distanceSourceToPatient - distanceSourceToDetector
    receiveScreenPlane = offsetPlane(visualizatioinPlane, distanceIsometricToScreen)
    # Create receiveScreenSurface
    u_interval = Rhino.Geometry.Interval(0, planeSize)
    v_interval = Rhino.Geometry.Interval(0, planeSize)
    receiveScreenSurface = Rhino.Geometry.PlaneSurface(receiveScreenPlane, u_interval, v_interval) 
    motionVector = (receiveScreenPlane.XAxis + receiveScreenPlane.YAxis)* planeSize / 2
    transform = Rhino.Geometry.Transform.Translation(-motionVector)
    receiveScreenSurface.Transform(transform)
    receiveScreenBrep = receiveScreenSurface.ToBrep()
    receiveScreenMesh = Rhino.Geometry.Mesh.CreateFromBrep(receiveScreenBrep, Rhino.Geometry.MeshingParameters.Default)
    return receiveScreenPlane, receiveScreenMesh

def viewportByName(viewName=None):
    """ Helper function
    Get a Rhino Viewport object using the name of the viewport.
    Inputs:
        viewName: Text for the name of the Rhino Viewport. If None, the
            current Rhino viewport will be used. If the view is a named view that
            is not currently open, it will be restored to the active view of
            the Rhino document.
    Output:  
        viewPort: An active viewport of the given name      
    """
    try:
        return scriptcontext.doc.Views.Find(viewName, False).ActiveViewport \
            if viewName is not None else scriptcontext.doc.Views.ActiveView.ActiveViewport
    except Exception:
        # try to find a named view and restore it
        viewTable = Rhino.RhinoDoc.ActiveDoc.NamedViews
        for i, viewp in enumerate(viewTable):
            if viewp.Name == viewName:
                activeViewport = scriptcontext.doc.Views.ActiveView.ActiveViewport
                viewTable.Restore (i, activeViewport)
                return activeViewport
        else:
            raise ValueError('Viewport "{}" was not found in the Rhino '
                             'document.'.format(viewName))

def setView(lightVector, receiveScreenPlane, distanceSourceToPatient, 
    cameraPosition=None, cameraUpDirection=None):
    """
    Set the camera position and orientation for Rhino's 'Perspective' viewport 
    Inputs:
        lightVector: <Rhino.Geometry.Vector3d> a vector represents the direction of light
      Either
        receiveScreenPlane: <Rhino.Geometry.Plane> a plane that receive the view 
        distanceSourceToPatient: float, the distance from the camera to receive plane
      Or provide the following two params
        cameraPosition: <Rhino.Geometry.Point3d> the point where the camera sits
        cameraUpDirection: <Rhino.Geometry.Vector3d> the direction of camera's up
    Output:  
        viewPort: An active 'Perspective' viewport that has its display mode set
    """
    # Set the 'Perspective' viewport
    cameraPosition = receiveScreenPlane.Origin + receiveScreenPlane.ZAxis*distanceSourceToPatient/6
    cameraUpDirection = -receiveScreenPlane.XAxis
    view_port = viewportByName('Perspective')
    view_port.SetCameraTarget(Rhino.Geometry.Point3d.Add(cameraPosition, lightVector), False)
    view_port.SetCameraDirection(lightVector, False)
    view_port.SetCameraLocation(cameraPosition, False)
    view_port.CameraUp = cameraUpDirection
    return view_port



if( __name__ == "__main__" ):
    
    visualizatioinPlane, lightVector = ConfigAngulation(positionerPrimaryAngle,positionerSecondaryAngle)


    receiveScreenPlane, receiveScreenMesh = ConfigreceiveScreen(visualizatioinPlane, distanceSourceToPatient, 
        distanceSourceToDetector, planeSize)

    viewport = setView(lightVector, receiveScreenPlane, distanceSourceToPatient)

    outFilePath = CaptureView(filePath, viewport, 1100, 1100)