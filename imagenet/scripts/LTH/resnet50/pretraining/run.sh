# magnitude-based 1 shot retraining
ARCH=${1:-"resnet50"} # resnet50 or mobilenetv2
SPARSITY_TYPE=${2:-"irregular"} #irregular or N:M-prune-pattern or 4:2-H-V-balanced or block or gather_scatter
CONFIG_FILE=${3:-""}
PRUNE_ARGS=${4:-""}
LOAD_CKPT=${5:-"xxxxx"}

GLOBAL_BATCH_SIZE=${8:-"1024"}
LOCAL_BATCH_SIZE=${9:-"128"}
EPOCHS=${10:-"90"}
WARMUP=${11:-"5"}
WIDTH=${12:-"64-128-256-512-64"}
AMP=${13:-"--amp"}
SEED=${14:-"914"}
LR_SCHEDULE=${15:-"cosine"}


cd ../../../..


INIT_LR=${7:-"0.2048"}


SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
mkdir -p ${SAVE_FOLDER}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}


INIT_LR=${7:-"0.4"}
SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
mkdir -p ${SAVE_FOLDER}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}


INIT_LR=${7:-"1.024"}
SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
mkdir -p ${SAVE_FOLDER}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}


INIT_LR=${7:-"1.6"}
SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
mkdir -p ${SAVE_FOLDER}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}


INIT_LR=${7:-"2.048"}
SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
mkdir -p ${SAVE_FOLDER}
python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}




#
## small dense
#
#ARCH=${1:-"resnet50"} # resnet50 or mobilenetv2
#SPARSITY_TYPE=${2:-"irregular"} #irregular or N:M-prune-pattern or 4:2-H-V-balanced or block or gather_scatter
#CONFIG_FILE=${3:-""}
#PRUNE_ARGS=${4:-""}
#LOAD_CKPT=${5:-"xxxxx"}
#
#GLOBAL_BATCH_SIZE=${8:-"1024"}
#LOCAL_BATCH_SIZE=${9:-"128"}
#EPOCHS=${10:-"90"}
#WARMUP=${11:-"5"}
#
#AMP=${13:-"--amp"}
#SEED=${14:-"914"}
#LR_SCHEDULE=${15:-"cosine"}
#
#
#cd ../../../..
#
#INIT_LR=${7:-"0.2048"}
#
##WIDTH=${12:-"19-38-77-120-64"}
##SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.914/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
##mkdir -p ${SAVE_FOLDER}
##python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#WIDTH=${12:-"26-51-110-182-26"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.832/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#WIDTH=${12:-"32-32-128-256-32"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.738/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#WIDTH=${12:-"38-77-168-312-38"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.59/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#
#
#INIT_LR=${7:-"0.4"}
#
#WIDTH=${12:-"19-38-77-120-64"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.914/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#WIDTH=${12:-"26-51-110-182-26"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.832/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#WIDTH=${12:-"32-32-128-256-32"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.738/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#WIDTH=${12:-"38-77-168-312-38"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.59/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#
#
#INIT_LR=${7:-"1.024"}
#
#WIDTH=${12:-"19-38-77-120-64"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.914/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#WIDTH=${12:-"26-51-110-182-26"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.832/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#WIDTH=${12:-"32-32-128-256-32"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.738/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#WIDTH=${12:-"38-77-168-312-38"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.59/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}




#
#INIT_LR=${7:-"1.6"}
#
#WIDTH=${12:-"19-38-77-120-64"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.914/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#WIDTH=${12:-"26-51-110-182-26"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.832/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#WIDTH=${12:-"32-32-128-256-32"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.738/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#WIDTH=${12:-"38-77-168-312-38"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.59/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#
#
#
#
#
#INIT_LR=${7:-"2.048"}
#
#WIDTH=${12:-"19-38-77-120-64"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.914/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#WIDTH=${12:-"26-51-110-182-26"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.832/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#WIDTH=${12:-"32-32-128-256-32"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.738/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
#WIDTH=${12:-"38-77-168-312-38"}
#SAVE_FOLDER=${6:-"./checkpoints/resnet50/LTH/pretraining/ep${EPOCHS}/lr${INIT_LR}/${WIDTH}_0.59/cosine_bs_${GLOBAL_BATCH_SIZE}_${WARMUP}ep_warmup"}
#mkdir -p ${SAVE_FOLDER}
#python ./multiproc.py --nproc_per_node 8 ./main.py /data/imagenet --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --seed ${SEED} --lr ${INIT_LR} --optimizer-batch-size ${GLOBAL_BATCH_SIZE} --warmup ${WARMUP} --arch ${ARCH} -c fanin --label-smoothing 0.1 --lr-schedule ${LR_SCHEDULE} --mom 0.875 --wd 3.0517578125e-05 --workspace ./ -b ${LOCAL_BATCH_SIZE} ${AMP} --static-loss-scale 128 --epochs ${EPOCHS} --mixup 0.2 --widths=${WIDTH} --sp-config-file=${CONFIG_FILE} --sp-admm-sparsity-type=${SPARSITY_TYPE} ${PRUNE_ARGS} --checkpoint-dir ${SAVE_FOLDER} --log-filename=${SAVE_FOLDER}/log.txt --resume ${LOAD_CKPT}
#
#
