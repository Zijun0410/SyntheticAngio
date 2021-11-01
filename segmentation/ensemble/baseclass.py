import platform
from pathlib import Path 

class BaseDir:
    def __init__(self):
        if platform.system()=='Linux':
            self.base_dir = Path(r'/nfs/turbo/med-kayvan-lab/Projects/Angiogram/Data/Processed/Zijun/UpdateTrainingPipline/data')
        elif platform.system()=='Windows':
            self.base_dir = Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\UpdateTrainingPipline\data') 
    @property
    def dir(self):
        return self.base_dir
