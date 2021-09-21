__author__ = "zijung@umich.edu"
__version__ = "2021.09.20"

import System
import Rhino
import scriptcontext

def HatchMeshProjection(targetMesh, receiveScreenPlane, viewport, colorCode, alpha=255):
    """
    inputs:
        receiveScreenPlane: <Rhino.Geometry.Plane>, plane that receive the outline
        viewport: <Rhino.Display.RhinoViewport>, the viewport info that provides the outline direction.
        targetMesh: <Rhino.Geometry.Mesh>, mesh for outline casting
        colorCode, alpha: See function CreateColor
    """
    # https://developer.rhino3d.com/api/RhinoCommon/html/M_Rhino_Geometry_Mesh_GetOutlines_1.htm
    # https://developer.rhino3d.com/api/RhinoCommon/html/T_Rhino_DocObjects_ViewportInfo.htm
    activeViewportInfor = Rhino.DocObjects.ViewportInfo(viewport)
    meshOutline = targetMesh.GetOutlines(activeViewportInfor, receiveScreenPlane)
    AddHatch(meshOutline, colorCode, alpha)

def AddHatch(curveObjects, colorCode, alpha, index=0):
    """
    Creates hatch objects for closed planar curves
    Inputs: 
        curveObjectes: <Rhino.Geometry.Curve> closed planar curves that defines the boundary of the hatch objects
        index: <python int> The fill type of a hatch pattern.
               0 = solid, uses object color
               1 = lines, uses pattern file definition
               2 = gradient, uses fill color definition
        colorCode, alpha: See function CreateColor
    """
    hatches = Rhino.Geometry.Hatch.Create(curveObjects, index=index, rotation=0, scale=1)
    #-# Setting up color gradient object
    # Rhino.Display.ColorStop(<System.Drawing.Color>, <python float>)
    ColorStops = [Rhino.Display.ColorStop(CreateColor(colorCode), 0), 
        Rhino.Display.ColorStop(CreateColor(colorCode), 1)]
    colorGradient = Rhino.Display.ColorGradient()
    colorGradient.SetColorStops(ColorStops)
    colorGradient.StartPoint = receiveScreenPlane.Origin
    colorGradient.EndPoint = receiveScreenPlane.Origin + receiveScreenPlane.ZAxis
    colorGradient.GradientType = Rhino.Display.GradientType.Linear
    for hatch in hatches:
        # Rhino.Geometry.Hatch.SetGradientFill(<Rhino.Display.ColorGradient>)
        hatch.SetGradientFill(colorGradient)
        #-# For rhino to show the hatch: Rhino.RhinoDoc.ActiveDoc.Objects.AddHatch(hatch)
        scriptcontext.doc.Objects.AddHatch(hatch)
    # scriptcontext.doc.Views.RedrawEnabled = False 
    scriptcontext.doc.Views.Redraw() 
    #-# For grasshopper to preview: Rhino.Geometry.Hatch.CreateDisplayGeometry()
    # Rhino.Display.DisplayPipeline.DrawHatch(hatch)

def CreateColor(colorCode, alpha):
    """
    Inputs:
        colorCode: list of three <python int>. Valid values are from 0 to 255.
            [0] for the red component. 
            [1] for the green component.
            [2] for the blue component.
        alpha: <python int>, The alpha component that represents the 
            degree of transparency. Valid values are 0 through 255.
    """
    return System.Drawing.Color.FromArgb(alpha, colorCode[0], colorCode[1], colorCode[2])

# # might be helper functions 
# def ObjectPrintColor(object_ids, color=None):
#     """Returns or modifies the print color of an object
#     Parameters:
#       object_ids = identifiers of object(s)
#       color[opt] = new print color. If omitted, the current color is returned.
#     Returns:
#       If color is not specified, the object's current print color
#       If color is specified, the object's previous print color
#       If object_ids is a list or tuple, the number of objects modified
#     """
#     id = rhutil.coerceguid(object_ids, False)
#     if id:
#         rhino_object = rhutil.coercerhinoobject(id, True, True)
#         rc = rhino_object.Attributes.PlotColor
#         if color:
#             rhino_object.Attributes.PlotColorSource = Rhino.DocObjects.ObjectPlotColorSource.PlotColorFromObject
#             rhino_object.Attributes.PlotColor = rhutil.coercecolor(color, True)
#             rhino_object.CommitChanges()
#             scriptcontext.doc.Views.Redraw()
#         return rc
#     for id in object_ids:
#         color = rhutil.coercecolor(color, True)
#         rhino_object = rhutil.coercerhinoobject(id, True, True)
#         rhino_object.Attributes.PlotColorSource = Rhino.DocObjects.ObjectPlotColorSource.PlotColorFromObject
#         rhino_object.Attributes.PlotColor = color
#         rhino_object.CommitChanges()
#     scriptcontext.doc.Views.Redraw()
#     return len(object_ids)