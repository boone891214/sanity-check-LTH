# pretrain
DATASET="cifar10"
ARCH="resnet"
DEPTH="20"

INIT_LR="0.1"
GLOBAL_BATCH_SIZE="64"
EPOCHS="160"
WARMUP=""
SCHEDULER="default"

SP=""
SPARSITY_TYPE=""
CONFIG_FILE=""
PRUNE_ARGS=""

SEED="914"
REMARK="run1"
LOAD_CKPT="XXXXX.pth.tar"
SAVE_FOLDER="checkpoints/${DATASET}/${ARCH}${DEPTH}/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/step_80_120_bs_${GLOBAL_BATCH_SIZE}_no_warmup/"


cd ../../..


CUDA_VISIBLE_DEVICES=7 python3 -u main_prune_train.py --arch ${ARCH} --depth ${DEPTH} --dataset ${DATASET} --optmzr sgd ${WARMUP} --batch-size ${GLOBAL_BATCH_SIZE} --lr ${INIT_LR} --lr-scheduler ${SCHEDULER} --resume ${LOAD_CKPT} --save-model ${SAVE_FOLDER} --epochs ${EPOCHS} --seed ${SEED} --remark ${REMARK} ${PRUNE_ARGS} --sp-admm-sparsity-type=${SPARSITY_TYPE} --sp-config-file=${CONFIG_FILE} --log-filename=${SAVE_FOLDER}/log.txt
