
import os
from pathlib import Path

def main(date_out,file_name,hour,cpus_per_task=1,mem_per_cpu=4,task_name='test'):


    current_dir = Path.cwd()

    saveDir = current_dir / 'scripts' /f'{date_out}'

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    fold_total = 4
    cc_number = task_name[2:]
    indFile = 1
    for fold_index in range(fold_total):
        
        bash_file = ['#!/bin/bash',
        f'#SBATCH --job-name={task_name}_{fold_index}',
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
        f'python {file_name} -f {fold_index} -n {cc_number} -j {cpus_per_task}']

        # f'python {file_name} -f {fold_index} -n {cc_number} -j {cpus_per_task}'
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
    with open(current_dir / f"{sh_file_name}{cc_number}.sh", "w") as filehandle:
        filehandle.writelines(f"{line}\n" for line in job_bash)


if __name__ == '__main__':
    input_dict = dict()   
    # undersample_names = [2000,3000,4000,5000]
    undersample_names = ['TL']
    for name in undersample_names:
        input_dict['cpus_per_task'] = 6
        input_dict['mem_per_cpu'] = 12
        input_dict['hour'] = 72
        input_dict['date_out'] = f'20210517_cv{name}'
        input_dict['file_name'] = 'train_par.py'
        input_dict['task_name'] = f'CV{name}'

        main(**input_dict)

