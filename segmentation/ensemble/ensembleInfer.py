from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import GradientBoostingClassifier
import pickle
import platform
from pathlib import Path 
import pandas as pd
import numpy as np
import os



def load_scaler(default_model_dir):

    scale_save_file = default_model_dir  / 'scaler_inference.csv'
        
    if not os.path.isfile(scale_save_file):
        raise FileNotFoundError(f"The scaler file is not found, check the path {default_model_path}")

    scalers_pd = pd.read_csv(scale_save_file)

    scaler_instance = StandardScaler()
    scaler_instance.mean_ = scalers_pd['mean'].values
    scaler_instance.scale_ = np.sqrt(scalers_pd['variance'].values)

    return scaler_instance


def load_model(default_model_dir):
    # Purpose:
    #    1. Load model if it exist
    #    2. Return None if it does not

    model_save_path = default_model_dir / 'final_model_new.sav'#'GBDT.sav'
    if os.path.isfile(model_save_path):
        try:
            modelHandle = pickle.load(open(model_save_path, 'rb'))
        except EOFError as e:
            # Just in case sometime it's empty
            return None
        return modelHandle
    return None

def generate_model_path():

    if platform.system()=='Linux':
        default_model_dir = Path(r'/nfs/turbo/med-kayvan-lab/Projects/Angiogram/Data/Processed/ensemble_model')
    elif platform.system()=='Windows':
        default_model_dir = Path(r'Z:\Projects\Angiogram\Data\Processed\ensemble_model') 
    else:
        raise EOFError("Platform not defined")

    return default_model_dir

def gbdt_classifier(data_in):

    default_model_dir = generate_model_path()
    scaler_instance = load_scaler(default_model_dir)
    transformed_data = scaler_instance.transform(np.array(data_in))
    model_handle = load_model(default_model_dir)
    predict_output = model_handle.predict_proba(transformed_data)[:, 1]

    return predict_output.tolist()

