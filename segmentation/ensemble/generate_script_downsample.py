
import os
from pathlib import Path

def main(date_out,file_name,hour,cpus_per_task=1,mem_per_cpu=4,task_name='test'):


    current_dir = Path.cwd()

    saveDir = current_dir / 'scripts' /f'{date_out}'

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    index_number = 10
    images_number = 130
    indFile = 1
    for start_index in range(1, images_number, index_number):
        
        bash_file = ['#!/bin/bash',
        f'#SBATCH --job-name={task_name}_{start_index}_{start_index+index_number-1}',
        '#SBATCH --mail-user=zijung@umich.edu ',
        '#SBATCH --mail-type=END',
        '#SBATCH --nodes=1', 
        '#SBATCH --ntasks-per-node=1', 
        f'#SBATCH --cpus-per-task={cpus_per_task}', 
        f'#SBATCH --mem-per-cpu={mem_per_cpu}gb', 
        f'#SBATCH --time={hour}:00:00 ',
        '#SBATCH --account=kayvan1', 
        '#SBATCH --partition=standard ',
        f'#SBATCH --output=./scripts/{date_out}/%A_%x.txt ',
        f'python {file_name} -s {start_index} -n {index_number}']
        # python downsample.py -s 1 -n 10
        
        with open(saveDir / f"submit{indFile}.sh", "w") as filehandle:
            filehandle.writelines(f"{line}\n" for line in bash_file)
        
        indFile += 1
        # break

    job_bash = ['#!/bin/bash',
    f'NUMBERS=$(seq 1 {indFile-1})',
    f'DATE={date_out}',
    'for NUM in ${NUMBERS}',
    'do',  
    '    sbatch scripts/${DATE}/submit${NUM}.sh',
    '    sleep 1',
    '    echo "Done ${NUM}"',
    'done']

    # Create the job.sh file
    sh_file_name = file_name.split('.')[0]
    with open(current_dir / f"{sh_file_name}.sh", "w") as filehandle:
        filehandle.writelines(f"{line}\n" for line in job_bash)



if __name__ == '__main__':
    input_dict = dict()    
    input_dict['cpus_per_task'] = 1
    input_dict['mem_per_cpu'] = 6
    input_dict['hour'] = 40
    input_dict['date_out'] = '20210613_4000'
    input_dict['file_name'] = 'downsample.py'
    input_dict['task_name'] = 'CC4000'

    main(**input_dict)

