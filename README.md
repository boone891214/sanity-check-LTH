# Sanity Checks for Lottery Tickets: Does Your Winning Ticket Really Win the Jackpot?
Sample code use for NeurIPS 2021 paper:
[Sanity Checks for Lottery Tickets: Does Your Winning Ticket Really Win the Jackpot?](https://arxiv.org/abs/2107.00166)





# CIFAR-10 and CIFAR-100

## Requirements

python >= 3.6

PyTorch >= 1.6

TorchVision >= 0.7

Other required dependency: `numpy`, `pyyaml`, `matplotlib`, `tensorboardX`, `opencv-python` , `sklearn`, `scikit-image`.

## Main pipeline

We prune globally. The layers that are considered "global" is defined in the corresponding `.yaml` files. Once we prune globally, 
the sparsity ratio in the `.yaml` file will be override by the global sparsity.
There are five necessary settings for the LTH experimens (using resnet-20 as example):

- pretraining

    - pretrain a network from scratch, resulting a dense model.
    
    ```
      cd scripts/resnet20/pretrain
      bash run.sh
    ```

- iterative magnitude-based pruning (`LT-IMP`)

    - Prune model iteratively. At each round the initial weights are rewind to the same initial point as pretraining. In this case, 
      specify the same `seed` used in the pretraining, which will give you the same initialization. You can also `--resume` a 
      pre-saved initia model as the initial point in case different servers may produce varied results using same seed.
    - Each round prunes 20% of the remaining weights.
    
    ```
      cd scripts/resnet20/iterative_pruning_lth_setting
      python run_imp_lr0.1.py
    ```

- oneshot pruning + retraining (essentially the so called lr-rewinding technique)
  
    - Oneshot prune the pretrained model by magnitude at the beginning, followed by retraining the remaining weights.
    
    ```
      cd scripts/resnet20/oneshot_pruning_retraining
      bash run_prune_lr_0.1.sh
    ```


- oneshot magnitude-based pruning (`LT-OMP`)
   
    - Must run oneshot pruning + retraining for 1 epochs to obtain the model with `OMP` masks.
   
    - This setting is different from "oneshot pruning + retraining". In "oneshot pruning + retraining", the sparse weights are continuously trained 
      from pretrained value (lr-rewinding), while in `OMP`, the LTH setting is: 1) using the mask obtained from oneshot pruning on the 
      pretrained model, and 2) using the same **initial** weights as pretraining for the sparse weight training.
    
    ```
      cd scripts/resnet20/oneshot_pruning_lth_setting
      bash run_omp_lr_0.1.sh
    ```


- ramdom re-initialization (`RR-IMP/OMP`)
   
    - For `RR-OMP`, similar to the scripts of `LT-OMP`, with different seed (or load different weights). For `RR-IMP`, change the seed
      and mask model path of `--sp-pre-defined-mask-dir` to your `LT-IMP` checkpoints.



# ImageNet-1k

## Requirements

For easy implementation, we suggest to use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) with CUDA-11 for the training environments.
We have pre-built the ready-to-run nvidia-docker image [here](https://drive.google.com/file/d/1kEXD8ZHXEoHIMSpAFKaKZh2SExoWHONy/view?usp=sharing).

- Load pre-built docker images (download or build): 
  
    `docker load -i nvidia_rn50.tar`


- Rename the docker image: 
  
    `docker image tag 4c5875fdd48859f69015c7ec7183e5d2e706ffe7dabcad177e39e041673dba82 nvidia_rn50:latest`


- Start nvidia-docker interactive session: 
  
    `nvidia-docker run --rm -it -v /path/to/your/imagenet/:/data/imagenet -v /path/to/your/project:/workspace/rn50 --ipc=host nvidia_rn50`


## Main pipeline

Similar with CIFAR experiments, there are five necessary settings for the LTH experimens (using resnet-50 as example):

- pretraining
    
    ```
      cd scripts/LTH/resnet50/pretraining
      bash run.sh
    ```

- iterative magnitude-based pruning (`LT-IMP`)
    
    ```
      cd scripts/LTH/resnet50/iterative_pruning_lth_setting
      python run_imp.py
    ```

- oneshot pruning + retraining (essentially the so called lr-rewinding technique)
- 
    ```
      cd scripts/LTH/resnet50/oneshot_pruning_retraining
      bash run_prune.sh
    ```


- oneshot magnitude-based pruning (`LT-OMP`)
   
    ```
      cd scripts/LTH/resnet50/oneshot_pruning_lth_setting
      bash run_omp.sh
    ```


- ramdom re-initialization (`RR-IMP/OMP`)
   
    - For `RR-OMP`, similar to the scripts of `LT-OMP`, with different seed (or load different weights). For `RR-IMP`, change the seed
      and mask model path of `--sp-pre-defined-mask-dir` to your `LT-IMP` checkpoints.





# Citation
if you find this repo is helpful, please cite
```
@article{ma2021sanity,
  title={Sanity Checks for Lottery Tickets: Does Your Winning Ticket Really Win the Jackpot?},
  author={Ma, Xiaolong and Yuan, Geng and Shen, Xuan and Chen, Tianlong and Chen, Xuxi and Chen, Xiaohan and Liu, Ning and Qin, Minghai and Liu, Sijia and Wang, Zhangyang and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

