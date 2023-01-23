import torch, os, re
import numpy as np


def fix_random_seeds(seed=0):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_save_name_from_hyper_params(hyper_params): 
    """
    Inputs:
        hyper_params <python dict>: all the hyper-params as a dictionary
    Output:
        <python string>: "{hyper-param-name}_{hyper-param-value}"
    """
    output_list = []
    for key, value in list(hyper_params.items()):
        output_list.append(f"{key}_{value}")
    name = '_'.join(output_list)

    return name

def get_the_latest_ckpt(chkpt_folder):
    """
    Inputs:
        chkpt_folder <python string>: the path to the folder containing all the checkpoints
    Output:
        <python string>: the path to the latest checkpoint
    """
    epoch_num_list, path_list = [], []
    for file in os.listdir(chkpt_folder):
        filename = os.fsdecode(file)
        if filename.endswith(".pt") and 'ckpt' in filename: 
            epoch = re.findall(r'\d+', filename)
            if len(epoch) > 0 :
                epoch_num_list.append(int(epoch[0]))
                path_list.append(filename)
    if len(epoch_num_list) == 0:
        return None
    max_epoch = max(epoch_num_list)
    return path_list[epoch_num_list.index(max_epoch)]
