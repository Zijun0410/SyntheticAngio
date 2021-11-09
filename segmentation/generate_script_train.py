
import os
from pathlib import Path

def main(date_out,file_name,hour,cpus_per_task=1,mem_per_cpu=8,task_name='test'):

    current_dir = Path.cwd()
    batch_runs = ['20211101', '20211102']
    model_names = ['config_inceptionresnet2_unet', 'config_resnet101_unet', 'config_unet', 'config_densenet121_unet']
    for batch_run in batch_runs:
        indFile = 1  
        saveDir = current_dir / 'scripts' /f'{batch_run}'
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        for model_name in model_names:
            bash_file = ['#!/bin/bash',
            f'#SBATCH --job-name={task_name}_{batch_run}_{model_name}',
            '#SBATCH --mail-user=zijung@umich.edu ',
            '#SBATCH --mail-type=END',
            '#SBATCH --nodes=1', 
            '#SBATCH --ntasks-per-node=1', 
            f'#SBATCH --cpus-per-task={cpus_per_task}', 
            f'#SBATCH --mem-per-cpu={mem_per_cpu}gb', 
            f'#SBATCH --time={hour}:00:00 ',
            '#SBATCH --account=kayvan0', 
            '#SBATCH --partition=standard ',
            f'#SBATCH --output=./scripts/{batch_run}/%A_%x.txt ',
            f'python {file_name} -c configs/{batch_run}/{model_name}.json'] 

            with open(saveDir / f"submit{indFile}.sh", "w") as filehandle:
                filehandle.writelines(f"{line}\n" for line in bash_file)
            indFile += 1

        job_bash = ['#!/bin/bash',
        f'NUMBERS=$(seq 1 {indFile-1})',
        f'DATE={batch_run}',
        'for NUM in ${NUMBERS}',
        'do',  
        '    sbatch scripts/${DATE}/submit${NUM}.sh',
        '    sleep 1',
        '    echo "Done ${NUM}"',
        'done']

        # Create the job.sh file
        with open(current_dir / 'scripts' / f"{batch_run}_job.sh", "w") as filehandle:
            filehandle.writelines(f"{line}\n" for line in job_bash)


if __name__ == '__main__':
    input_dict = dict()   

    input_dict['cpus_per_task'] = 1
    input_dict['mem_per_cpu'] = 16
    input_dict['hour'] = 10
    input_dict['date_out'] = f'20211107'
    input_dict['file_name'] = 'train.py'
    input_dict['task_name'] = f'TestSynthetic'

    main(**input_dict)

