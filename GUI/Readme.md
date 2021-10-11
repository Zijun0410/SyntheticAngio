### Sythetic Angio GUI
Author: Zijun Gao 
Last Update: Oct 10th, 2021

This folder is for GUI-aid manual annotation of the stenosis location and catheter end point.

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
2. `Clear` before you want to `Broese` again.
3. Draw points on the Screen, `Clear` if drawn by mistake. Click `Save` for saving the output. Use the scroll bar to find the right frame.
4. `Next` show the next Dicom file, also automatically save the output if not saved by user. `Previous` shows the last Dicom file, will not do the saving stuff though.



Notes: Originlly developed in `cd C:\Users\zijung\Desktop\AngioGUI`, then moved to `Z:\Projects\Angiogram\Code\Zijun\SyntheticAngio\GUI` for Github.