#! usr/bin/env bash

# todo anatomy: knee pd ->brain t2

args=(
    --shift n_views
    --pretrain_dataset "mayo"
    --supervised_dataset_name "mayo"

    # --pretrain_dataset "deeplesion"
    # --supervised_dataset_name "deeplesion"


    --n_views 40
    --real_n_views 40
    
    --epochs 20
    --lr 1e-3
    --batch_size 1
    )

# python ./training_ct/ct_py/learn.py "${args[@]}" --epochs 19 # 20 epoch 💎
# python ./training_ct/ct_py/ttt_train.py "${args[@]}" --epochs 20 # 20 epoch 💎

# python ./training_ct/ct_py/our_s1_train.py "${args[@]}" --epochs 21 # 20 epoch 💎

python ./training_ct/ct_py/metainvnet_train.py "${args[@]}" # 20 epoch 💎

# TTA
args=(
    --shift n_views
    --pretrain_dataset "mayo"
    --supervised_dataset_name "mayo"
    # --pretrain_dataset "deeplesion"
    # --supervised_dataset_name "deeplesion"
    
    # --n_views 120
    --n_views 70
    --real_n_views 70
    --epochs 20
    --lr 1e-3
    --batch_size 1
    )

# todo ei

# python ./training_ct/ct_py/EI.py "${args[@]}" --epochs 13 # 20 epoch 💎

# python ./training_ct/ct_py/ttt_tta.py "${args[@]}" # 20 epoch 💎

# python ./training_ct/ct_py/our_s2_tta.py "${args[@]}"  # 20 epoch 💎
