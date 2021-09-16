import rhinoscriptsyntax as rs
import os


def ExportControlPoints(baseDir=r'C:\Users\gaozj\Desktop\Angio\SyntheticAngio\data'):
    """
    Export curve's control points to a text file
    """
    # Select all the curves 
    curveGUIDs = rs.GetObjects("Select curve", rs.filter.curve)

    for iCurve in range(len(curveGUIDs)):
        curveObjGUID = curveGUIDs[iCurve]
        editPts = rs.CurvePoints(curveObjGUID)
        rs.AddTextDot('{}'.format(iCurve), editPts[-1])
        print('The number of edit points in {} curve is {}'.format(iCurve,len(editPts) ))
        with open(os.path.join(baseDir,'{}.txt'.format(iCurve)), 'w') as fileHandle:
            for pt in editPts:
                fileHandle.write( str(pt.X) )
                fileHandle.write( ", " )
                fileHandle.write( str(pt.Y) )
                fileHandle.write( ", " )
                fileHandle.write( str(pt.Z) )
                fileHandle.write( "\n" ) 

    # The number of edit points in 0 curve is 40
    # The number of edit points in 1 curve is 30
    # The number of edit points in 2 curve is 30
    # The number of edit points in 3 curve is 130
    # The number of edit points in 4 curve is 40
    # The number of edit points in 5 curve is 20

if( __name__ == "__main__" ):
    ExportControlPoints()