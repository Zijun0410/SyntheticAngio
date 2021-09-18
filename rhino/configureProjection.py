import System
import Rhino

def GenerateMeshProjection(receiveScreenPlane, viewport, targetMesh):
    """
    inputs:
        receiveScreenPlane: <Rhino.Geometry.Plane>, plane that receive the outline
        viewport: <Rhino.Display.RhinoViewport>, the viewport info that provides the outline direction.
        targetMesh: <Rhino.Geometry.Mesh>, mesh for outline casting
    """
    # https://developer.rhino3d.com/api/RhinoCommon/html/M_Rhino_Geometry_Mesh_GetOutlines_1.htm
    # https://developer.rhino3d.com/api/RhinoCommon/html/T_Rhino_DocObjects_ViewportInfo.htm
    outline = targetMesh.GetOutlines(Rhino.DocObjects.ViewportInfo(viewport), receiveScreenPlane)

    # Hatch the outlline


    hatch = Rhino.Geometry.Hatch.Create(outline, 0, 0, 1)
    # For rh to show the hatch
    Rhino.RhinoDoc.ActiveDoc.Objects.AddHatch(hatch)
    # For grasshopper to preview
    #  Rhino.Geometry.Hatch.CreateDisplayGeometry()
    Rhino.Display.DisplayPipeline.DrawHatch(hatch, System.Drawing.Color.Red, System.Drawing.Color.Red)




def AddHatches(curve_ids, hatch_pattern=None, scale=1.0, rotation=0.0, tolerance=None):
    """Creates one or more new hatch objects a list of closed planar curves
    Parameters:
      curve_ids ([guid, ...]): identifiers of the closed planar curves that defines the
          boundary of the hatch objects
      hatch_pattern (str, optional):  name of the hatch pattern to be used by the hatch
          object. If omitted, the current hatch pattern will be used
      scale (number, optional): hatch pattern scale factor
      rotation (number, optional): hatch pattern rotation angle in degrees.
      tolerance (number, optional): tolerance for hatch fills.
    Returns:
      list(guid, ...): identifiers of the newly created hatch on success
      None: on error
    Example:
      import rhinoscriptsyntax as rs
      curves = rs.GetObjects("Select closed planar curves", rs.filter.curve)
      if curves:
          if rs.IsHatchPattern("Grid"):
              rs.AddHatches( curves, "Grid" )
          else:
              rs.AddHatches( curves, rs.CurrentHatchPattern() )
    See Also:
      AddHatch
      CurrentHatchPattern
      HatchPatternNames
    """
    __initHatchPatterns()
    id = rhutil.coerceguid(curve_ids, False)
    if id: curve_ids = [id]
    index = scriptcontext.doc.HatchPatterns.CurrentHatchPatternIndex
    if hatch_pattern is None:
        hatch_pattern = CurrentHatchPattern()
    if isinstance(hatch_pattern, int) and hatch_pattern != index:
        index = hatch_pattern
    else:
        pattern_instance = scriptcontext.doc.HatchPatterns.FindName(hatch_pattern)
        index = Rhino.RhinoMath.UnsetIntIndex if pattern_instance is None else pattern_instance.Index
    if index<0: return scriptcontext.errorhandler()
    curves = [rhutil.coercecurve(id, -1, True) for id in curve_ids]
    rotation = Rhino.RhinoMath.ToRadians(rotation)
    if tolerance is None or tolerance < 0:
        tolerance = scriptcontext.doc.ModelAbsoluteTolerance
    hatches = Rhino.Geometry.Hatch.Create(curves, index, rotation, scale, tolerance)
    if not hatches: return scriptcontext.errorhandler()
    ids = []
    for hatch in hatches:
        id = scriptcontext.doc.Objects.AddHatch(hatch)
        if id==System.Guid.Empty: continue
        ids.append(id)
    if not ids: return scriptcontext.errorhandler()
    scriptcontext.doc.Views.Redraw()
    return ids
