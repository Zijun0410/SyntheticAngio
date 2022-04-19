__author__ = "zijung@umich.edu"
__version__ = "2022.02.01"


import rhinoscriptsyntax as rs
import os


def ExportControlPoints(baseDir='..\\data\\Construction\\', 
folder_name = 'RCA_Detail', subfolder_name = ''):
    """
    Export curve's control points to a text file
    r'C:\Users\gaozj\Desktop\Angio\SyntheticAngio\data'

    """
    # Select all the curves 
    curveGUIDs = rs.GetObjects("Select curve", rs.filter.curve)

    for iCurve in range(len(curveGUIDs)):
        curveObjGUID = curveGUIDs[iCurve]
        editPts = rs.CurvePoints(curveObjGUID)
        rs.AddTextDot('{}'.format(iCurve), editPts[-1])
        print('The number of edit points in {} curve is {}'.format(iCurve,len(editPts) ))
        save_directory = os.path.join(baseDir+folder_name, subfolder_name)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        with open(os.path.join(save_directory, '{}.txt'.format(iCurve)), 'w') as fileHandle:
            for pt in editPts:
                fileHandle.write( str(pt.X) )
                fileHandle.write( ", " )
                fileHandle.write( str(pt.Y) )
                fileHandle.write( ", " )
                fileHandle.write( str(pt.Z) )
                fileHandle.write( "\n" ) 

if( __name__ == "__main__" ):
    ExportControlPoints(folder_name='LCA_Detail_2', subfolder_name='branch')