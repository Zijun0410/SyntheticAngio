#!/bin/bash

DATE=20211102
sbatch scripts/${DATE}/config_densenet121_unet.sh
sleep 1
sbatch scripts/${DATE}/config_inceptionresnet2_unet.sh
sleep 1
sbatch scripts/${DATE}/config_resnet101_unet.sh
sleep 1
sbatch scripts/${DATE}/config_unet.sh
sleep 1