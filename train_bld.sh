export CUDA_VISIBLE_DEVICES=0,1,2,3
BLD_TRAINING="/data/dataset/dataset_low_res"
BLD_TRAINLIST="lists/blendedmvs/train.txt"
BLD_TESTLIST="lists/blendedmvs/val.txt"
BLD_CKPT_FILE="./checkpoints/model_dtu.ckpt"  # dtu pretrained model

exp="EIA-MVS-BLD-TEST"
PY_ARGS=${@:2}

BLD_LOG_DIR="./checkpoints/bld/"$exp 
if [ ! -d $BLD_LOG_DIR ]; then
    mkdir -p $BLD_LOG_DIR
fi



python -u train_bld.py --fpn_base_channel=8 --epochs=10 --logdir $BLD_LOG_DIR --dataset=blendedmvs_aug --batch_size=4 --trainpath=$BLD_TRAINING --summary_freq 100 --loadckpt $BLD_CKPT_FILE\
        --ndepths 32,16,8,4 --depth_inter_r 2.0,1.0,1.0,0.5 --group_cor_dim 4,4,4,4 --group_cor --inverse_depth --rt --attn_temp 2 --trainlist $BLD_TRAINLIST --testlist $BLD_TESTLIST  $PY_ARGS | tee -a $BLD_LOG_DIR/log.txt


