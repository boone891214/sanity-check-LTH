# load a pre-defined mask (from one_shot_pruning_retraining.sh results) and apply it on initial weights (use same seed with pretraining to get same initial weights)
SEED="914"

DATASET="cifar10"
ARCH="resnet"
DEPTH="32"

INIT_LR="0.1"
GLOBAL_BATCH_SIZE="64"
EPOCHS="160"
WARMUP=""
SCHEDULER="default"

SP="0.956"
SPARSITY_TYPE="irregular_global"
CONFIG_FILE="./profiles/${ARCH}${DEPTH}_LTH/irregular/resnet_global.yaml"
PRUNE_ARGS="--sp-retrain --retrain-mask-pattern=pre_defined --sp-pre-defined-mask-dir=checkpoints/${DATASET}/${ARCH}${DEPTH}/LTH/one_shot_pruning/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/step_80_120_bs_${GLOBAL_BATCH_SIZE}_no_warmup/checkpoint_${SEED}.pth.tar"


REMARK="run1"
LOAD_CKPT="XXXXX.pth.tar"
SAVE_FOLDER="checkpoints/${DATASET}/${ARCH}${DEPTH}/LTH/winning_ticket/ep${EPOCHS}/lr${INIT_LR}/sp_${SP}_global/step_80_120_bs_${GLOBAL_BATCH_SIZE}_no_warmup/"


cd ../../..


CUDA_VISIBLE_DEVICES=3 python3 -u main_prune_train.py --arch ${ARCH} --depth ${DEPTH} --dataset ${DATASET} --optmzr sgd ${WARMUP} --batch-size ${GLOBAL_BATCH_SIZE} --lr ${INIT_LR} --lr-scheduler ${SCHEDULER} --resume ${LOAD_CKPT} --save-model ${SAVE_FOLDER} --epochs ${EPOCHS} --seed ${SEED} --remark ${REMARK} ${PRUNE_ARGS} --sp-admm-sparsity-type=${SPARSITY_TYPE} --sp-config-file=${CONFIG_FILE} --log-filename=${SAVE_FOLDER}/log.txt 
