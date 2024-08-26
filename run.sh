#!/bin/bash

python train.py --kwargs resize=True hflip=True normalize=True model_name="efficientnet_b3" batch_size=32 data_path="/home/ubuntu/IDNet/split_normal/fin2_sidtd/"
python train.py --kwargs resize=True hflip=True normalize=False model_name="efficientnet_b3" batch_size=32 data_path="/home/ubuntu/IDNet/split_normal/fin2_sidtd/"
python train.py --kwargs resize=True hflip=False normalize=False model_name="efficientnet_b3" batch_size=32 data_path="/home/ubuntu/IDNet/split_normal/fin2_sidtd/"
python train.py --kwargs resize=False hflip=False normalize=False model_name="efficientnet_b3" batch_size=2 data_path="/home/ubuntu/IDNet/split_normal/fin2_sidtd/"

python train.py --kwargs resize=True hflip=True normalize=True model_name="efficientnet_b3" batch_size=32 data_path="/home/ubuntu/IDNet/split_normal/fin1_fixed/"
python train.py --kwargs resize=True hflip=True normalize=False model_name="efficientnet_b3" batch_size=32 data_path="/home/ubuntu/IDNet/split_normal/fin1_fixed/"
python train.py --kwargs resize=True hflip=False normalize=False model_name="efficientnet_b3" batch_size=32 data_path="/home/ubuntu/IDNet/split_normal/fin1_fixed/"
python train.py --kwargs resize=False hflip=False normalize=False model_name="efficientnet_b3" batch_size=2 data_path="/home/ubuntu/IDNet/split_normal/fin1_fixed/"
