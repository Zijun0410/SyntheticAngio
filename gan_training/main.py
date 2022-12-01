from args import *
import argparse
import torch
import numpy as np

def training_gan():
	pass

if __name__ == '__main__':
	args = argparse.ArgumentParser('GAN')

    args.add_argument('-gt', '--generator_type', default='unet', type=str,
                      help='The type of the generator')
    args.add_argument('-gd', '--generator_depth', default=3, type=int,
                      help='The depth of the generator')

    args.add_argument('-dt', '--disciminator_type', default='resnet18', type=str,
                      help='The type of the disciminator') 

    args.add_argument('-d', '--dataset', default='all', type=str,
                      help='The dataset type')

    # args.add_argument('-a', '--availability', default=1.0, type=float,
    #                   help='The availability of priviledged infor') 
    # args.add_argument('-s', '--seed', default=1, type=int,
    #                   help='The batch index')

    parser = argparse.ArgumentParser('BYOL', parents=[moswl_utils.get_args_parser()])
    args = parser.parse_args()

	main()