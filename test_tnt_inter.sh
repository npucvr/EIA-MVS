export CUDA_VISIBLE_DEVICES=7

TNT_INTER_TESTPATH="/data/local_userdata/wangshaoqian/TankandTemples/TankandTemples/intermediate/"
TNT_INTER_TESTLIST="lists/tnt/inter.txt"

TNT_CKPT_FILE="./checkpoints/model_tank.ckpt" # dtu pretrained model (only for Horse)

exp="EIA-MVS-TANK-TEST"
PY_ARGS=${@:2}

TNT_LOG_DIR="./checkpoints/tnt/"$exp 
if [ ! -d $TNT_LOG_DIR ]; then
    mkdir -p $TNT_LOG_DIR
fi
TNT_OUT_DIR="/data/local_userdata/wangshaoqian/"$exp

if [ ! -d $TNT_OUT_DIR ]; then
    mkdir -p $TNT_OUT_DIR
fi


python -u test_dypcd_tnt_inter.py --dataset=tanks --fpn_base_channel=8 --batch_size=1 --testpath=$TNT_INTER_TESTPATH  --testlist=$TNT_INTER_TESTLIST --loadckpt $TNT_CKPT_FILE --interval_scale 1.06 --outdir $TNT_OUT_DIR\
            --ndepths 32,16,8,4 --depth_inter_r 2.0,1.0,1.0,0.5 --group_cor_dim 4,4,4,4 --num_view=11  --group_cor --attn_temp 2 --inverse_depth $PY_ARGS | tee -a $TNT_LOG_DIR/log_test.txt


