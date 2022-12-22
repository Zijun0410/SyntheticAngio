#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=Tryout
#SBATCH --output=./scripts/%A_%x.txt
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8000m 
#SBATCH --time=10:00:00
#SBATCH --account=kayvan1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# The application(s) to execute along with its input arguments and options:
python main.py 