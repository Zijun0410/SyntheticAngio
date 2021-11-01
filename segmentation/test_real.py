import sys, os
import pandas as pd
import torch
from torchvision.utils import save_image

from data_loader.data_loaders import *

from util import read_json, MetricTracker

import model.metric as module_metric
import model.model as module_arch


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

data_loader = RealDataLoader(load_dir = "/nfs/turbo/med-kayvan-lab/Projects/Angiogram/Data/Processed/Zijun/UpdateTrainingPipline/data",
                             save_dir = "/nfs/turbo/med-kayvan-lab/Projects/Angiogram/Data/Processed/Zijun/Synthetic")



config_dir = Path('/nfs/turbo/med-kayvan-lab/Projects/Angiogram/Code/Zijun/SyntheticAngio/segmentation/configs')
model_load_dir = Path(r"/nfs/turbo/med-kayvan-lab/Users/zijung/Code/angio_update/saved/models")
batch_num = 8

metrics = [getattr(module_metric, met) for met in config['metrics']]

batch_runs = ['20211101','20211102']
for batch in batch_runs:
    config_pathes = glob.glob(os.path.join(config_dir, batch_run, "*.json"))
    for config_path in config_pathes:
        config = read_json(config_path)
        task_name = json_config['name']
        task_purpose = json_config['purpose']
        task_dir_list = glob.glob(os.path.join(model_load_dir / task_name, "*"))
        task_dirs = filter(lambda x:x.endswith(f'{task_purpose}'), task_dirs_list)
        for task_dir in task_dirs:
            checkpoint_path = task_dirs / 'model_best.pth'
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(str(checkpoint_path), map_location = device)
            state_dict = checkpoint['state_dict']
            model = init_obj(config, 'arch', module_arch)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            # Initiate the dataset
            dataset = data_loader.get_dataset()
            # Initiate Metric Function 
            metric_ftns = [getattr(module_metric, met) for met in config['metrics']]
            test_metrics = MetricTracker('loss', *[m.__name__ for m in metric_ftns])
            # Set up image saving dir
            output_save_dir = data_loader.get_save_dir("Seg_Result") / f'{task_name}_real'
            if not os.path.isdir(output_save_dir):
                output_save_dir.mkdir(parents=True, exist_ok=True)

            # Loop through the images
            with torch.no_grad():
                for i, (index, data, target) in enumerate(data_loader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)

                    for data_idx, _ in enumerate(target):
                        output_per_image = output[data_idx].squeeze()
                        target_per_image = target[data_idx].squeeze()
                        raw_per_image = data[data_idx].squeeze()

                        for met in metric_ftns:
                            test_metrics.update(met.__name__, met(output_per_image, target_per_image))

                        output_image_save_path = output_save_dir / f'{index[data_idx]}_predseg.png'
                        target_image_save_path = output_save_dir / f'{index[data_idx]}_trueseg.png'
                        raw_image_save_path = output_save_dir / f'{index[data_idx]}_real.png'

                        save_image(output_per_image, output_image_save_path)
                        save_image(target_per_image, target_image_save_path)
                        save_image(raw_per_image, raw_image_save_path)

            test_df = pd.DataFrame([test_metrics.result(), test_metrics.std()])

            df_save_path = task_dir / f'test_real.csv'
            test_df.to_csv(df_save_path)
