EXP_NAME=T00052

ROOT_DIR=/home/demiguo/nerf_pl

LOG_DIR=${ROOT_DIR}/logs/${EXP_NAME}
DATA_DIR=${ROOT_DIR}/data/brandenburg_gate
DATASET=phototourism

GPU=8
BOX_WARP=2.00
IMG_DS=8
MODEL=NeRFCube
N_FEAT=48
BACKBONE_RES=128
BS=1024
LR=0.0001000
EPOCHS=20
USE_XYZ_NET=1
N_XYZ_DIM=128
MODE=default

cd ${ROOT_DIR}
mkdir -p ${LOG_DIR}

CUDA_VISIBLE_DEVICES=${GPU} python train.py --dataset_name ${DATASET} --img_downscale ${IMG_DS} --use_cache --N_importance 64 --N_samples 64 --beta_min 0.03 --N_vocab 1500 --num_epochs ${EPOCHS} --batch_size ${BS} --optimizer adam --lr ${LR} --lr_scheduler cosine --exp_name ${EXP_NAME} --root_dir ${DATA_DIR}  --model ${MODEL} --n_features ${N_FEAT} --backbone_res ${BACKBONE_RES} --box_warp ${BOX_WARP} --use_xyz_net ${USE_XYZ_NET} --n_xyz_dim ${N_XYZ_DIM} --mode ${MODE} 1> ${LOG_DIR}/log.out 2> ${LOG_DIR}/log.err


