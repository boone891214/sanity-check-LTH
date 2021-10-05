# magnitude-based 1 shot pruning+retraining (oneshot prune on pretrained weights and retraining)
DATASET="cifar10"
ARCH="resnet"
DEPTH="20"

INIT_LR="0.1"
GLOBAL_BATCH_SIZE="64"
EPOCHS="160"
WARMUP=""
SCHEDULER="default"

SP="0.956"
SPARSITY_TYPE="irregular_global"
CONFIG_FILE="./profiles/${ARCH}${DEPTH}_LTH/irregular/resnet_global.yaml"
PRUNE_ARGS="--sp-retrain --sp-prune-before-retrain --sp-global-weight-sparsity=${SP}"

SEED="914"
REMARK="run1"
LOAD_CKPT="checkpoints/cifar10/resnet20/LTH/pretraining/ep160/lr0.1/step_80_120_bs_64_no_warmup/cifar10_resnet20_acc_46.460_sgd_lr0.1_default_epoch1_seed914_run1.pt"
SAVE_FOLDER="checkpoints/${DATASET}/${ARCH}${DEPTH}/LTH/one_shot_pruning/ep160/lr${INIT_LR}/sp_${SP}_global/step_80_120_bs_${GLOBAL_BATCH_SIZE}_no_warmup/"


cd ../../..

CUDA_VISIBLE_DEVICES=0 python3 -u main_prune_train.py --arch ${ARCH} --depth ${DEPTH} --dataset cifar10 --optmzr sgd ${WARMUP} --batch-size ${GLOBAL_BATCH_SIZE} --lr ${INIT_LR} --lr-scheduler ${SCHEDULER} --resume ${LOAD_CKPT} --save-model ${SAVE_FOLDER} --epochs ${EPOCHS} --seed ${SEED} --remark ${REMARK} ${PRUNE_ARGS} --sp-admm-sparsity-type=${SPARSITY_TYPE} --sp-config-file=${CONFIG_FILE} --log-filename=${SAVE_FOLDER}/log.txt


