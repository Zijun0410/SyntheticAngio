
import os
from pathlib import Path

def main(date_out,file_name,hour,cpus_per_task=1,mem_per_cpu=4,task_name='test'):


    current_dir = Path.cwd()

    saveDir = current_dir / 'scripts' /f'{date_out}'

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    indFile = 1
    CCS = [2000,3000,4000,5000]
    for cc_number in CCS:    
        bash_file = ['#!/bin/bash',
        f'#SBATCH --job-name={task_name}_{cc_number}',
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
        f'python {file_name} -n {cc_number} ']
        
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
    with open(current_dir / 'scripts' / f"{sh_file_name}_for_test.sh", "w") as filehandle:
        filehandle.writelines(f"{line}\n" for line in job_bash)


if __name__ == '__main__':
    input_dict = dict()   

    input_dict['cpus_per_task'] = 1
    input_dict['mem_per_cpu'] = 12
    input_dict['hour'] = 30
    input_dict['date_out'] = f'20210525_test'
    input_dict['file_name'] = 'train.py'
    input_dict['task_name'] = f'Test'

    main(**input_dict)

