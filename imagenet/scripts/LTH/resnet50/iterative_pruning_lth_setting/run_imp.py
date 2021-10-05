import os
from time import sleep
import argparse
import numpy as np
import time

os.chdir(r"/workspace/rn50")
print("Current Working Directory ", os.getcwd())
time.sleep(2)



arch = "resnet50"
sparsity_type = "irregular_global"
imp_ratio = "0.2"

config_file = "./profiles/resnet_LTH/{}/irregular/resnet_global.yaml".format(arch)

lr = "0.2048"
lr_scheduler = "cosine"
epochs = "90"
batch_size = "1024"
seed = "914"
warmup = "5"
remark = "run_1"


global_workspace = "checkpoints/{}/LTH/iterative_pruning_rewinding/ep{}/lr{}/cosine_bs_{}_{}ep_warmup/imp_ratio_{}/".format(arch, epochs, lr, batch_size, warmup, imp_ratio)
local_workspace = ""



# first round command
load_weight_path = "checkpoints/{}/LTH/pretraining/ep{}/lr{}/cosine_bs_{}_{}ep_warmup/model_best.pth.tar".format(arch, epochs, lr, batch_size, warmup)
local_workspace = os.path.join(global_workspace, "round_0_sp0.0/")
prune_args = "--sp-retrain --sp-prune-before-retrain --sp-global-weight-sparsity={} --imp --imp-ratio={} --imp-round=0".format(imp_ratio, imp_ratio)
cmd = "python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu " \
      "--raport-file raport.json -j8 -p 100 --seed {} --lr {} --optimizer-batch-size {} " \
      "--warmup {} --arch {} -c fanin --label-smoothing 0.1 --lr-schedule {} --mom 0.875 " \
      "--wd 3.0517578125e-05 --workspace ./ -b 128 --amp --static-loss-scale 128 --epochs 1 " \
      "--mixup 0.2 --widths=64-128-256-512-64 --sp-config-file={} --sp-admm-sparsity-type={} {} " \
      "--checkpoint-dir {} --log-filename={}log.txt --resume {} " \
      "--restart-training".format(seed, lr, batch_size, warmup, arch, lr_scheduler, config_file,
                                             sparsity_type, prune_args, local_workspace, local_workspace, load_weight_path)


print(cmd)
os.system(cmd)



# rest rounds commands
# load_weight_path = "./checkpoints/{}/LTH/pretraining/ep90/lr{}/cosine_bs_1024_5ep_warmup/checkpoint-5.pth.tar".format(arch, lr) # for weight rewinding, load weights at early stage
load_weight_path = "xxxx" # for lottery ticket initialization, load init weight or use same seed for same init weight

total_round = 12

for round in range(1, total_round):
    ''' the main code use round value to determine sparsity ratio to prune AFTER each retraining step '''

    # define current workspace folder name
    sp = 1 - (1 - float(imp_ratio)) ** round
    sp = np.round(sp, 3)

    # load previous round model that is hard pruned without retraining
    sp_prev = 1 - (1 - float(imp_ratio)) ** (round - 1)
    sp_prev = np.round(sp_prev, 3)

    prune_args = "--sp-retrain --retrain-mask-pattern=pre_defined --imp --imp-ratio={} --imp-round={}".format(imp_ratio, str(round))

    local_workspace = os.path.join(global_workspace, "round_{}_sp{}/".format(round, sp))

    load_mask_path = os.path.join(global_workspace, "round_{}_sp{}/checkpoint_{}_harden_for_next_round.pth.tar".format(round - 1, sp_prev, seed))

    cmd = "python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu " \
          "--raport-file raport.json -j8 -p 100 --seed {} --lr {} --optimizer-batch-size {} " \
          "--warmup {} --arch {} -c fanin --label-smoothing 0.1 --lr-schedule {} --mom 0.875 " \
          "--wd 3.0517578125e-05 --workspace ./ -b 128 --amp --static-loss-scale 128 --epochs {} " \
          "--mixup 0.2 --widths=64-128-256-512-64 --sp-config-file={} --sp-admm-sparsity-type={} {} " \
          "--checkpoint-dir {} --log-filename={}log.txt --resume {} " \
          "--sp-pre-defined-mask-dir={}".format(seed, lr, batch_size, warmup, arch, lr_scheduler, epochs, config_file,
                                      sparsity_type, prune_args, local_workspace, local_workspace, load_weight_path, load_mask_path)


    print(cmd)
    os.system(cmd)




