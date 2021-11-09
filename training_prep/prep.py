from path import Path
from pandas import read_csv
from sklearn.model_selection import train_test_split
import numpy as np

def training_preparation(split_size, dataset_ids, base_dir):
    for dataset_id in dataset_ids:
        csv_path = dir_path / dataset_id / 'stenosis_detail.csv'
        infor_summary_df = read_csv(csv_path)
        idx_train, idx_validation = train_test_split(range(len(infor_summary_df)), test_size=split_size, random_state=1, stratify=infor_summary_df['degree'])
        infor_summary_df['train'] = 0
        infor_summary_df.loc[idx_train, 'train'] = 1
        save_path = csv_path / dir_path / dataset_id / 'infor.csv'
        infor_summary_df.to_csv(save_path, index=False)

if __name__ == '__main__':
	base_dir = Path(r'/nfs/turbo/med-kayvan-lab/Projects/Angiogram/Data/Processed/Zijun/Synthetic/Sythetic_Output')
	dataset_ids = ['UKR', 'UoMR', 'UKR_Movement', 'UoMR_Movement']
	training_preparation(split_size, dataset_ids, base_dir)