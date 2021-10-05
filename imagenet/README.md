# Requirements

CUDA >= 11.0

python >= 3.6

PyTorch >= 1.6

TorchVision >= 0.7

Use NVIDIA docker to execute training scripts. All required 

The NVIDIA docker container can be obtained from open source: https://github.com/NVIDIA/nvidia-docker

Please refer to this tutorial to install nvidia docker command: https://cnvrg.io/how-to-setup-docker-and-nvidia-docker-2-0-on-ubuntu-18-04/

Download our pre-build docker image for this repository [here](https://drive.google.com/file/d/1kEXD8ZHXEoHIMSpAFKaKZh2SExoWHONy/view?usp=sharing).

Other required dependency: `numpy`, `pyyaml`, `matplotlib`, `tensorboardX`, `opencv-python` , `sklearn`, `scikit-image`.


# Docker

- Load pre-built docker images (download or build): 
  
    `docker load -i nvidia_rn50.tar`


- Rename the docker image: 
  
    `docker image tag 4c5875fdd48859f69015c7ec7183e5d2e706ffe7dabcad177e39e041673dba82 nvidia_rn50:latest`


- Start nvidia-docker: 
  
    `nvidia-docker run --rm -it -v /path/to/your/imagenet/:/data/imagenet -v /path/to/your/project:/workspace/rn50 --ipc=host nvidia_rn50`

# ADMM pruning and retraining

We prune layer-wisely. The layers that are considered to be pruned are defined in the corresponding `.yaml` files. Once pruned, 
the sparsity ratio in the `.yaml` file will be used.

We support three types of sparsity schemes, `pattern`, `connectivity`, and `block` sparsity. To combine different sparsity together 
(e.g., pattern+connectivity), just specify sparsity type as `pattern+connectivity` in training scripts.

To prune a model with desired sparsity, there are three steps: pretrain, ADMM prune, and retrain.


### Pretrain

Pretrain a network from scratch, resulting a dense model. We include `resnet50` and `vgg16` in this repository as examples.

```
  cd scripts/ADMM/resnet50/pretrain
  bash run.sh
```

**Command explain:**

`ARCH`: Network architectures to be used (e.g. resnet50, vgg16, mobilenetv2, etc.).

`WIDTH`: Width multiplier of the resnet architecture, default is `64-128-256-512-64`.

`INIT_LR`: Initial learning rate.

`LR_SCHEDULE`: Learning rate scheduler, step, cosine, etc.

`GLOBAL_BATCH_SIZE`: The optimizer batch size.

`LOCAL_BATCH_SIZE`: The per-gpu batch size.

`EPOCHS`: Total training epochs.

`WARMUP`: The warm up epochs.

`AMP`: Using the nvidia APEX Mixed Precision for accelerating training.

`SEED`: random seed.

`LOAD_CKPT`: The path of the checkpoint to be loaded. If not found, the training will *NOT* stop, and it will use random
initialization to start.

`SAVE_FOLDER`: The directory that models are saved in this training process.


***  


### ADMM prune and retrain

Using ADMM regularization to solve the pruning problem iteratively. At the end of the ADMM training, we hard prune the 
model and obtain a sparse model with desired sparsity scheme and ratio. After ADMM training, the model is retrained
immediately to retore accuracy, and the sparse mask that is obtained will be retained.


```
  cd scripts/ADMM/resnet50/prune
  bash run.sh
```

**Extra command explain:**

`SPARSITY_TYPE`: Define sparsity type. Use `+` to combine different sparsity types.

`CONFIG_FILE`: The `.yaml` file that specify layer-wise pruning configuration.

`PRUNE_ARGS`: The pruning-related arguments (e.g., pruning initialization, ADMM, retrain, etc.).



