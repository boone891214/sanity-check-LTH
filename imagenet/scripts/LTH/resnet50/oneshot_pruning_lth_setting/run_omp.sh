# magnitude-based 1 shot retraining
ARCH=${1:-"resnet50"} # resnet50 or mobilenetv2
SPARSITY_TYPE=${2:-"irregular_global"} 
SP=${16:-"0.738"}
CONFIG_FILE=${3:-"./profiles/resnet_LTH/${ARCH}/irregular/resnet_global.yaml"}
GLOBAL_BATCH_SIZE=${8:-"1024"}
LOCAL_BATCH_SIZE=${9:-"128"}
EPOCHS=${10:-"90"}
WARMUP=${11:-"5"}
WIDTH=${12:-"64-128-256-512-64"}
AMP=${13:-"--amp"}
SEED=${14:-"914"}
LR_SCHEDULE=${15:-"cosine"}
LOAD_CKPT=${5:-"./checkpoints/${ARCH}/LTH/initial_weight/seed_${SEED}.pth.tar"}


cd ../../../..


INIT_LR=${7:-"0.2048"}


SAVE_FOLDER=${6:-"./checkpoints/${ARCH}/LTH/winning_ticket_oneshot/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
PRUNE_ARGS=${4:-"--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=./checkpoints/${ARCH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup/model_best.pth.tar"}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT} --restart-training


INIT_LR=${7:-"0.4"}


SAVE_FOLDER=${6:-"./checkpoints/${ARCH}/LTH/winning_ticket_oneshot/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
PRUNE_ARGS=${4:-"--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=./checkpoints/${ARCH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup/model_best.pth.tar"}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT} --restart-training


INIT_LR=${7:-"1.024"}

SAVE_FOLDER=${6:-"./checkpoints/${ARCH}/LTH/winning_ticket_oneshot/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
PRUNE_ARGS=${4:-"--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=./checkpoints/${ARCH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup/model_best.pth.tar"}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT} --restart-training


INIT_LR=${7:-"1.6"}

SAVE_FOLDER=${6:-"./checkpoints/${ARCH}/LTH/winning_ticket_oneshot/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
PRUNE_ARGS=${4:-"--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=./checkpoints/${ARCH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup/model_best.pth.tar"}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT} --restart-training



INIT_LR=${7:-"2.048"}

SAVE_FOLDER=${6:-"./checkpoints/${ARCH}/LTH/winning_ticket_oneshot/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
PRUNE_ARGS=${4:-"--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=./checkpoints/${ARCH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup/model_best.pth.tar"}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup 8 --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.3 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT} --restart-training







# magnitude-based 1 shot retraining
ARCH=${1:-"resnet50"} # resnet50 or mobilenetv2
SPARSITY_TYPE=${2:-"irregular_global"} 
SP=${16:-"0.59"}
CONFIG_FILE=${3:-"./profiles/resnet_LTH/${ARCH}/irregular/resnet_global.yaml"}
GLOBAL_BATCH_SIZE=${8:-"1024"}
LOCAL_BATCH_SIZE=${9:-"128"}
EPOCHS=${10:-"90"}
WARMUP=${11:-"5"}
WIDTH=${12:-"64-128-256-512-64"}
AMP=${13:-"--amp"}
SEED=${14:-"914"}
LR_SCHEDULE=${15:-"cosine"}
LOAD_CKPT=${5:-"./checkpoints/${ARCH}/LTH/initial_weight/seed_${SEED}.pth.tar"}




INIT_LR=${7:-"0.2048"}


SAVE_FOLDER=${6:-"./checkpoints/${ARCH}/LTH/winning_ticket_oneshot/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
PRUNE_ARGS=${4:-"--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=./checkpoints/${ARCH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup/model_best.pth.tar"}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT} --restart-training


INIT_LR=${7:-"0.4"}


SAVE_FOLDER=${6:-"./checkpoints/${ARCH}/LTH/winning_ticket_oneshot/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
PRUNE_ARGS=${4:-"--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=./checkpoints/${ARCH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup/model_best.pth.tar"}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT} --restart-training


INIT_LR=${7:-"1.024"}

SAVE_FOLDER=${6:-"./checkpoints/${ARCH}/LTH/winning_ticket_oneshot/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
PRUNE_ARGS=${4:-"--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=./checkpoints/${ARCH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup/model_best.pth.tar"}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT} --restart-training


INIT_LR=${7:-"1.6"}

SAVE_FOLDER=${6:-"./checkpoints/${ARCH}/LTH/winning_ticket_oneshot/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
PRUNE_ARGS=${4:-"--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=./checkpoints/${ARCH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup/model_best.pth.tar"}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup 8 --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.3 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT} --restart-training



INIT_LR=${7:-"2.048"}

SAVE_FOLDER=${6:-"./checkpoints/${ARCH}/LTH/winning_ticket_oneshot/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
PRUNE_ARGS=${4:-"--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=./checkpoints/${ARCH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup/model_best.pth.tar"}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup 8 --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.3 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT} --restart-training





# magnitude-based 1 shot retraining
ARCH=${1:-"resnet50"} # resnet50 or mobilenetv2
SPARSITY_TYPE=${2:-"irregular_global"} 
SP=${16:-"0.832"}
CONFIG_FILE=${3:-"./profiles/resnet_LTH/${ARCH}/irregular/resnet_global.yaml"}
GLOBAL_BATCH_SIZE=${8:-"1024"}
LOCAL_BATCH_SIZE=${9:-"128"}
EPOCHS=${10:-"90"}
WARMUP=${11:-"5"}
WIDTH=${12:-"64-128-256-512-64"}
AMP=${13:-"--amp"}
SEED=${14:-"914"}
LR_SCHEDULE=${15:-"cosine"}
LOAD_CKPT=${5:-"./checkpoints/${ARCH}/LTH/initial_weight/seed_${SEED}.pth.tar"}




INIT_LR=${7:-"0.2048"}


SAVE_FOLDER=${6:-"./checkpoints/${ARCH}/LTH/winning_ticket_oneshot/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
PRUNE_ARGS=${4:-"--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=./checkpoints/${ARCH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup/model_best.pth.tar"}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT} --restart-training


INIT_LR=${7:-"0.4"}


SAVE_FOLDER=${6:-"./checkpoints/${ARCH}/LTH/winning_ticket_oneshot/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
PRUNE_ARGS=${4:-"--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=./checkpoints/${ARCH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup/model_best.pth.tar"}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT} --restart-training


INIT_LR=${7:-"1.024"}

SAVE_FOLDER=${6:-"./checkpoints/${ARCH}/LTH/winning_ticket_oneshot/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
PRUNE_ARGS=${4:-"--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=./checkpoints/${ARCH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup/model_best.pth.tar"}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT} --restart-training


INIT_LR=${7:-"1.6"}

SAVE_FOLDER=${6:-"./checkpoints/${ARCH}/LTH/winning_ticket_oneshot/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
PRUNE_ARGS=${4:-"--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=./checkpoints/${ARCH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup/model_best.pth.tar"}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup 8 --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.3 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT} --restart-training



INIT_LR=${7:-"2.048"}

SAVE_FOLDER=${6:-"./checkpoints/${ARCH}/LTH/winning_ticket_oneshot/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
PRUNE_ARGS=${4:-"--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=./checkpoints/${ARCH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup/model_best.pth.tar"}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup 8 --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.3 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT} --restart-training
