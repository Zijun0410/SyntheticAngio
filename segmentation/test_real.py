import sys, os
import pandas as pd
import torch
from torchvision.utils import save_image
import argparse
from base import RealDataLoader

from utils import read_json, MetricTracker

import model.metric as module_metric
import model.model as module_arch

from pathlib import Path
import glob
import numpy as np

def init_obj(config, name, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.

    `object = config.init_obj('name', module, a, b=1)`
    is equivalent to
    `object = module.name(a, b=1)`
    """
    module_name = config[name]['type']
    module_args = dict(config[name]['args'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)


def main(task_side='R', batch_run='20211101', model_index='best'):
    """
    Inputs:
        task_side <python string>: "R", or "L" or "ALL"
        batch_run <python string>: yyyymmdd, in accordance to the scripts and config files
    """
    data_loader = RealDataLoader(load_dir="/nfs/turbo/med-kayvan-lab/Projects/Angiogram/Data/Processed/Zijun/UpdateTrainingPipline/data",
                                 save_dir="/nfs/turbo/med-kayvan-lab/Projects/Angiogram/Data/Processed/Zijun/Synthetic",
                                 side=task_side)

    config_dir = Path('/nfs/turbo/med-kayvan-lab/Projects/Angiogram/Code/Zijun/SyntheticAngio/segmentation/configs')
    model_load_dir = Path(r"/nfs/turbo/med-kayvan-lab/Projects/Angiogram/Code/Zijun/SyntheticAngio/segmentation/saved/models") 


    config_pathes = glob.glob(os.path.join(config_dir, batch_run, "*.json"))
    for config_path in config_pathes:
        config = read_json(config_path)
        metrics = [getattr(module_metric, met) for met in config['metrics']]
        task_name = config['name']
        task_purpose = config['purpose']
        task_dir_list = glob.glob(os.path.join(model_load_dir / task_name, "*", ""))
        task_dirs = list(filter(lambda x: Path(x).name.endswith(f'{task_purpose}'), task_dir_list))
        for task_dir in task_dirs:
            print(f"Current Task Dir: {'/'.join(Path(task_dir).parts[-3:])}")
            if model_index=='best':
                checkpoint_path = Path(task_dir) / 'model_best.pth'
            else: # DenseNet, etc
                checkpoint_path = Path(task_dir) / 'checkpoint-epoch-3.pth'
                model_index = '3'
                if not checkpoint_path.is_file(): # Unet
                    checkpoint_path = Path(task_dir) / 'checkpoint-epoch-10.pth'
                    model_index = '10'

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(str(checkpoint_path), map_location = device)
            state_dict = checkpoint['state_dict']
            model = init_obj(config, 'arch', module_arch)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            # Initiate the dataset loader
            test_set_loader = data_loader.get_data_loader()
            # Initiate Metric Function 
            metric_ftns = [getattr(module_metric, met) for met in config['metrics']]
            test_metrics = MetricTracker('loss', *[m.__name__ for m in metric_ftns])
            # Set up image saving dir
            output_save_dir = data_loader.get_save_dir("Seg_Result") / f'{task_name}_Real' / f"{task_side}_{model_index}"
            print(f"Results saved in {'/'.join(Path(output_save_dir).parts[-4:])}")

            if not os.path.isdir(output_save_dir):
                output_save_dir.mkdir(parents=True, exist_ok=True)

            # Loop through the images
            with torch.no_grad():
                for i, (index, data, target) in enumerate(test_set_loader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)

                    for data_idx, _ in enumerate(target):
                        output_per_image = output[data_idx].squeeze()
                        target_per_image = target[data_idx].squeeze()
                        raw_per_image = data[data_idx].squeeze()

                        for met in metric_ftns:
                            try:
                                test_metrics.update(met.__name__, met(output_per_image, target_per_image))
                            except TypeError:
                                # "model/metric_util.py", line 88, in calculate_metric
                                # TypeError: only size-1 arrays can be converted to Python scalars
                                test_metrics.update(met.__name__, np.nan)
                                print(f"The {met.__name__} metric return error message on image No.{index[data_idx]}")
                                
                        output_image_save_path = output_save_dir / f'{index[data_idx]}_predseg.png'
                        target_image_save_path = output_save_dir / f'{index[data_idx]}_trueseg.png'
                        raw_image_save_path = output_save_dir / f'{index[data_idx]}_real.png'

                        save_image(output_per_image, output_image_save_path)
                        save_image(target_per_image, target_image_save_path)
                        save_image(raw_per_image, raw_image_save_path)

            test_df = pd.DataFrame([test_metrics.result(), test_metrics.std()])

            df_save_path = Path(output_save_dir) / f'test_real.csv'
            test_df.to_csv(df_save_path)

            del model

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-s', '--side', default='R', type=str,
                      help='the side to run on ')
    args.add_argument('-b', '--batch', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-m', '--model', default=None, type=str,
                      help='the model used to make inference')
    args = args.parse_args()
    main(task_side=args.side, batch_run=args.batch, model_index=args.model)