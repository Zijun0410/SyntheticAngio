__author__ = "zijung@umich.edu"
__version__ = "2021.09.21"

import Rhino
import scriptcontext
import System

def AddLayer(layerName='Synthetic', layerColor=System.Drawing.Color.Black):
    """
    Creates a Layer in Rhino using a name and optional color. Returns the
    index of the layer requested. If the layer already exists, the color 
    is updated and no new layer is created.
    Inputs: 
        layerName: <python str> the name of the layer
        layerColor: <System.Drawing.Color> the dedault color of the layer
    Outputs:
        layerIndex: <python int>, the index of the layer found or created
    """
    docLyrs = scriptcontext.doc.Layers
    layerIndex = docLyrs.Find(layerName, True)
    if layerIndex == -1:
        layerIndex = docLyrs.Add(layerName,layerColor)
    else: # it exists
        layer = docLyrs[layerIndex] # so get it
        if layer.Color != layerColor: # if it has a different color
            layer.Color = layerColor # reset the color
    return layerIndex

# def MoveToLayer(selectedObject, layerIndex):
#     """
#     Move an object to a layer
#     Not useful for now because the function need drawing happens before
#     the object to be moved to another layer
#     """
#     selectedObject.Attributes.LayerIndex = layerIndex
#     selected_object.CommitChanges()

def DeleteLayerObject(layerName='Synthetic', deleteLayer=False, quiet=True):
    """
    Delete all the object on the layer, or even the layer itself
    Inputs:
        layerName: <python str>, the name of the layer to be deleted
        quiet: <python bool>. If omitted, the operation is done quietly
        deleteLayer: <python bool>. If omitted, layer will not be deleted
    """
    layerIndex = scriptcontext.doc.Layers.Find(layerName, True)
    settings = Rhino.DocObjects.ObjectEnumeratorSettings()
    settings.LayerIndexFilter = layerIndex
    objsOnLayer = scriptcontext.doc.Objects.FindByFilter(settings)
    objIDs = [obj.Id for obj in objsOnLayer]
    scriptcontext.doc.Objects.Delete(objIDs, quiet)
    if deleteLayer:
        scriptcontext.doc.Layers.Delete(layerIndex, quiet)