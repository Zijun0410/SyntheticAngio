from pathlib import Path
import pandas as pd


def removeDuplicateRow(file_path, col):
	data_in = pd.read_csv(file_path)
	data_out = data_in.drop_duplicates(subset=col, keep='last')
	data_out.to_csv(file_path, index=False)  
	print('Duplicate Removed')

if __name__ == '__main__':
	# file_dir = Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Rhino_Output\UoMR')
	# file_name = 'stnosis_infor.csv'
	# removeDuplicateRow(file_path=file_dir/file_name, col=['fileName'])
	file_dir = Path(r'Z:\Projects\Angiogram\Data\Processed\Zijun\Synthetic\Meta_Data')
	file_name = 'UoM_Right_endpoint.csv'
	removeDuplicateRow(file_path=file_dir/file_name, col=['name_combine'])
	