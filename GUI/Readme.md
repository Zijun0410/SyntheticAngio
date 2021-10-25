### Sythetic Angio GUI
Author: Zijun Gao 
Last Update: Oct 25th, 2021

This folder is for GUI-aid manual annotation of the stenosis location and catheter end point.

![Main Window](main.PNG)

**Development Details**
1. Install the following dependencies for dicom file processing
```bash
pip install python-gdcm
pip install pydicom
```
2. Use Qt Designer to design the GUI
3. Turn `.ui` file into `.py` by
```bash
python -m PyQt5.uic.pyuic -x mainWindow.ui -o mainWindow.py
```

**User Manual**
1. From I/O Setting, select `Browse` -> `Load` to settle the loading folder, same for the save folder.
2. `Clear Dots` before you want to `Browse` again.
3. Use the scroll bar to find the right frame. Draw points (by mouse left-click) on the screen, `Clear Dots` if drawn by mistake. 
4. Click `Save Frame` to save frames (and after saving it'll jump to the next frame). For a frame that has been saved, the check box would be checked to avoid double annotation. 
5. `Next File` shows the next Dicom file, `Previous File` shows the last Dicom file. If there were some unsaved annotations, they will be cleared and won't be brought to different files.
6. Click `Bad File` if the quality of the file is low, it will create empty folder useful counting the total file that have been annotated
7. `Clear Folder` will clear all the content under the folder but keep the folder itself.
8. `Summary` created a summary csv files for all annotations, take around 30 seconds for 1000+ folders, the main window may freeze during this process



Notes: Originlly developed in `cd C:\Users\zijung\Desktop\AngioGUI`, then moved to `Z:\Projects\Angiogram\Code\Zijun\SyntheticAngio\GUI` for Github.