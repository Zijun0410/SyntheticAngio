# SyntheticAngio
 Create synthetic angiogram image based on 3D coronary artery model


## 1. Endpoint Annotation For Real Background Image
Annotate the end point of catheter with GUI. 
Code: `Z:\Projects\Angiogram\Code\Zijun\SyntheticAngio\GUI`, run `python main.py`
Output: The paired images and summary information are saved at `Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\BackGround_Image`

BackGround_Image 
    |-- UoM_Right_endpoint.csv  
    |-- UoM_Right
        |-- 1001-21
            |-- background_1.png
            |-- endpoint_1.csv  
            |-- endpoint_1.png

Columns of summary `Folder_Side_endpoint.csv` are:
[filename, load_dir, frame_num, x, y, idnetifier, index, background, annotate, save_dir, 
PositionerPrimaryAngle, PositionerSecondaryAngle, DistanceSourceToDetector, DistanceSourceToPatient]

Columns of individual `endpoint_1.csv` are:
[filename, load_dir, frame_num, x, y, (PositionerPrimaryAngle, PositionerSecondaryAngle,
DistanceSourceToDetector, DistanceSourceToPatient)]

Previouly, the UK dataset are annotated with matlab and PhotoShop with `Z:\Projects\Angiogram\Code\Zijun\SyntheticAngio\matlab\Background_Image_Preparation.m`, 
these data need to be handled separately in the subsequent pipeline. Especially for the metadata columns that would be used by Rhino:
**[FileName, PositionerPrimaryAngle, PositionerSecondaryAngle, DistanceSourceToDetector, DistanceSourceToPatient]**


Move metadata to `Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Meta_Data`

## 2. Rhino Sythetic Data Generation Based on Real Background Data and MeteData
Generate sythetic images based on the 3D structure and meta data
Code: `Z:\Projects\Angiogram\Code\Zijun\SyntheticAngio\rhino`
1. Open Rhino, load model from `Z:\Projects\Angiogram\Code\Zijun\SyntheticAngio\model`
2. Type in Command: `EditPythonScript`
3. In the Python Script Editor, run `main.py`
    First, set batch number, i.e. `batch_num = 'UK'`.
    Second, `main(baseDir, defaultBranchesNum, batch_num, adjust=True, debug=False)`, adjust the position of receive screen.
    Third, `main(baseDir, defaultBranchesNum, batch_num, adjust=False, debug=False)`.

## 3. Sythetic Image Generation
Generate the sythetic image for training.

Code: `Z:\Projects\Angiogram\Code\Zijun\SyntheticAngio\matlab\Generate_Synthetic_Image.m`

Generate_Synthetic_Image.m (Synthetic Image Generation)
    |-- Load_Rhino_Image.m (Crop, Resize and Match)
        |-- Config_Path.m (Setting up the Path)


Output: `Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Sythetic_Output`

Sythetic_Output
    |-- {batch_num}
        |-- 001 (7)-R
            |-- stenosis_infor.csv [stenosis_location, effect_region, percentage, degree, identifier, x_center, y_center]
            |-- synthetic.png
            |-- angio_struct.mat
            |-- montage.png

## 4. Training Preparation 



