# Sanity Checks for Lottery Tickets: Does Your Winning Ticket Really Win the Jackpot?
simple version of using prune_utils for dynamic sparse training and lottery ticket hypothesis





# CIFAR-10 and CIFAR-100

## Requirements

python >= 3.6

PyTorch >= 1.6

TorchVision >= 0.7

Other required dependency: `numpy`, `pyyaml`, `matplotlib`, `tensorboardX`, `opencv-python` , `sklearn`, `scikit-image`.

## Main pipeline

We prune globally. The layers that are considered "global" is defined in the corresponding `.yaml` files. Once we prune globally, 
the sparsity ratio in the `.yaml` file will be override by the global sparsity.
There are four necessary settings for the LTH experimens (using resnet-20 as example):

- pretraining

    - pretrain a network from scratch, resulting a dense model.
    
    ```
      cd scripts/resnet20/pretrain
      bash run.sh
    ```

- iterative magnitude-based pruning (`IMP`)

    - Prune model iteratively. At each round the initial weights are rewind to the same initial point as pretraining. In this case, 
      specify the same `seed` used in the pretraining, which will give you the same initialization. You can also `--resume` a 
      pre-saved initia model as the initial point in case different servers may produce varied results using same seed.
    - Each round prunes 20% of the remaining weights.
    
    ```
      cd scripts/resnet20/iterative_pruning_lth_setting
      python run_iterative_lth_lr0.1.py
    ```

- oneshot pruning + retraining (essentially the so called lr-rewinding technique)
  
    - Oneshot prune the pretrained model by magnitude at the beginning, followed by retraining the remaining weights.
    
    ```
      cd scripts/resnet20/oneshot_pruning_retraining
      bash run_oneshot_lr_0.1.sh
    ```


- oneshot magnitude-based pruning (`OMP`)
    
    - This setting is different from "oneshot pruning + retraining". In "oneshot pruning + retraining", the sparse weights are continuously trained 
      from pretrained value (no rewinding), while in `OMP`, the LTH setting is: 1) using the mask obtained from oneshot pruning on the 
      pretrained model, and 2) using the same **initial** weights as pretraining for the sparse weight training.
    
    ```
      cd scripts/resnet20/oneshot_pruning_lth_setting
      bash run_winning_lr_0.1.sh
    ```

# ImageNet-1k

## Requirements

For easy implementation, we suggest to use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) for the running environments.
We have pre-built the ready-to-run nvidia-docker image here.

- Load pre-built docker images (download or build): 
  
    `docker load -i nvidia_rn50.tar`


- Rename the docker image: 
  
    `docker image tag 4c5875fdd48859f69015c7ec7183e5d2e706ffe7dabcad177e39e041673dba82 nvidia_rn50:latest`


- Start nvidia-docker interactive session: 
  
    `nvidia-docker run --rm -it -v /path/to/your/imagenet/:/data/imagenet -v /path/to/your/project:/workspace/rn50 --ipc=host nvidia_rn50`



