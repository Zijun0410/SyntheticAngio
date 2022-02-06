### Setups
1. source activate digits
2. synthetic
3. cd segmentation
4. initiate a gpu running environment 
srun --account=kayvan1 --time=04:00:00 --ntasks-per-node=1 --mem-per-cpu=12GB --partition=gpu --gres=gpu:1 --pty /bin/bash

srun --account=kayvan1 --time=04:00:00 --ntasks-per-node=1 --mem-per-cpu=16GB  --pty /bin/bash

### Debug Mode
python train.py -c configs/20211101/config_densenet121_unet.json -bg true
python train.py -c configs/20211102/config_densenet121_unet.json -bg true

**pdb through if there are error messages**
python3 -m pdb train.py -c configs/20211101/config_densenet121_unet.json -bg true