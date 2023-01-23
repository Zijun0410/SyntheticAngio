### Using FOCS for stenosis detection

### Debug from terminal
```bash
module load python3.9-anaconda/2021.11
source activate digits
```

### RunGPU from terminal
```bash
salloc --account=kayvan1 --nodes=1 --ntasks-per-node=1 --mem-per-cpu=8GB --cpus-per-task=1 --partition=gpu --gres=gpu:1 --time=00:20:00
```

### AutoLoad in Jupyter Lab
```
%load_ext autoreload
%autoreload 2
```

### This is a tutorial on using the FCOS model for inference and plotting out the model's predictions.
https://learnopencv.com/fcos-anchor-free-object-detection-explained/

### Inputs for training
The input to the model is expected to be a list of tensors, each of shape `[C, H, W]`, one for each image, and should be in 0-1 range. Different images can have different sizes.

During training, the model expects both the input tensors and targets (list of dictionary), containing:

1. boxes (`FloatTensor[N, 4]`): the ground-truth boxes in `[x1, y1, x2, y2]` format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.

2. labels (`Int64Tensor[N]`): the class label for each ground-truth box

The model returns a `Dict[Tensor]` during training, containing the classification and regression losses.

### Input for inference
During inference, the model requires only the input tensors, and returns the post-processed predictions as a `List[Dict[Tensor]]`, one for each input image. The fields of the Dict are as follows, where N is the number of detections:

1. boxes (`FloatTensor[N, 4]`): the predicted boxes in `[x1, y1, x2, y2]` format, with `0 <= x1 < x2 <= W` and `0 <= y1 < y2 <= H`.

2. labels (`Int64Tensor[N]`): the predicted labels for each detection

3. scores (`Tensor[N]`): the scores of each detection