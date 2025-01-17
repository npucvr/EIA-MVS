export CUDA_VISIBLE_DEVICES=5
DTU_TESTPATH="/home/wangshaoqian/data/DTU/dtu_testing/"

DTU_TESTLIST="lists/dtu/test.txt"
DTU_CKPT_FILE='./checkpoints/model_dtu.ckpt' # dtu pretrained model

exp="EIA-MVS-TEST"


PY_ARGS=${@:2}

DTU_LOG_DIR="./checkpoints/dtu/"$exp 
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi
DTU_OUT_DIR="./outputs/dtu/"$exp

if [ ! -d $DTU_OUT_DIR ]; then
    mkdir -p $DTU_OUT_DIR
fi

python -u test_dtu.py --dataset=general_eval4 --fpn_base_channel=8 --batch_size=1 --testpath=$DTU_TESTPATH  --testlist=$DTU_TESTLIST --loadckpt $DTU_CKPT_FILE --interval_scale 1.06 --outdir $DTU_OUT_DIR\
            --ndepths 32,16,8,4 --depth_inter_r 2.0,1.0,1.0,0.5 --group_cor_dim 4,4,4,4 --conf 0.55 --group_cor --attn_temp 2 --inverse_depth $PY_ARGS | tee -a $DTU_LOG_DIR/log_test.txt
