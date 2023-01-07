## This folder is for the training of sythetic image generation model 

### Debug from terminal
```bash
module load python3.9-anaconda/2021.11
source activate digits
```
### RunGPU from terminal
```bash
salloc --account=kayvan1 --nodes=1 --ntasks-per-node=1 --mem-per-cpu=8GB --cpus-per-task=1 --partition=gpu --gres=gpu:1 --time=00:20:00
```
