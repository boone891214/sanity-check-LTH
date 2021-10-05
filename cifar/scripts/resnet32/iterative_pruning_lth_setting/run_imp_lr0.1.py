import os
import time
import argparse
import numpy as np

# the root working directory of the project
os.chdir(r"/data/xiaolong/code/cifar_DST_LTH_s")
print("Current Working Directory ", os.getcwd())
time.sleep(2)

GPU_ID = 7

arch = "resnet"
depth = "32"
dataset = "cifar10"
sparsity_type = "irregular_global"
imp_ratio = "0.2"

config_file = "./profiles/{}{}_LTH/irregular/resnet_global.yaml".format(arch, depth)

lr = "0.1"
lr_scheduler = "default"
epochs = "160"
batch_size = "64"
seed = "914"
warmup = ""
remark = "run_1"


global_workspace = "checkpoints/{}/{}{}/LTH/iterative_pruning_rewinding/ep{}/lr{}/step_80_120_bs_{}_no_warmup/imp_ratio_{}/".format(dataset, arch, depth, epochs, lr, batch_size, imp_ratio)
local_workspace = ""


# first round command: load a pretrained model, one-shot prune to 'imp_ratio' sparsity
load_weight_path = "checkpoints/{}/{}{}/LTH/pretraining/ep160/lr{}/step_80_120_bs_{}_no_warmup/seed914_64_lr_0.1_resnet20_cifar10_acc_92.530_sgd_lr0.1_default_epoch132.pt".format(dataset, arch, depth, lr, batch_size)
local_workspace = os.path.join(global_workspace, "round_0_sp0.0/")
prune_args = "--sp-retrain --sp-prune-before-retrain --sp-global-weight-sparsity={} --imp --imp-ratio={} --imp-round=0".format(imp_ratio, imp_ratio)
cmd = "CUDA_VISIBLE_DEVICES={} python3 -u main_prune_train.py --arch {} " \
          "--depth {} " \
          "--dataset {} " \
          "--optmzr sgd {} " \
          "--batch-size {} " \
          "--lr {} " \
          "--lr-scheduler {} " \
          "--resume {} " \
          "--save-model {} " \
          "--epochs 1 --seed {} " \
          "--remark {} " \
          "{} " \
          "--sp-admm-sparsity-type={} " \
          "--sp-config-file={} " \
          "--log-filename={}log.txt ".format(GPU_ID, arch, depth, dataset, warmup, batch_size, lr, lr_scheduler,
                                             load_weight_path, local_workspace, seed, remark, prune_args,
                                             sparsity_type, config_file, local_workspace)


print(cmd)
os.system(cmd)


# rest rounds commands, load initial model at every round
load_weight_path = "xxxx.pth.tar"

total_round = 15

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


    cmd = "CUDA_VISIBLE_DEVICES={} python3 -u main_prune_train.py --arch {} " \
          "--depth {} " \
          "--dataset {} " \
          "--optmzr sgd {} " \
          "--batch-size {} " \
          "--lr {} " \
          "--lr-scheduler {} " \
          "--resume {} " \
          "--save-model {} " \
          "--epochs {} --seed {} " \
          "--remark {} " \
          "{} " \
          "--sp-admm-sparsity-type={} " \
          "--sp-config-file={} " \
          "--log-filename={}log.txt " \
          "--sp-pre-defined-mask-dir={} ".format(GPU_ID, arch, depth, dataset, warmup, batch_size, lr, lr_scheduler,
                                                load_weight_path, local_workspace, epochs, seed, remark,
                                                prune_args, sparsity_type, config_file, local_workspace, load_mask_path)


    print(cmd)
    os.system(cmd)




