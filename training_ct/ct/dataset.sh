#! usr/bin/env bash

# todo anatomy: knee pd ->brain t2

args=(
    --shift dataset
    --pretrain_dataset "deeplesion"
    --supervised_dataset_name "spine"

    # --pretrain_dataset "deeplesion"
    # --supervised_dataset_name "deeplesion"

    --epochs 20
    --lr 1e-3
    --batch_size 1
    )


# python ./training_ct/ct_py/learn.py "${args[@]}" # 20 epoch 💎
# python ./training_ct/ct_py/ttt_train.py "${args[@]}" # 20 epoch 💎
# python ./training_ct/ct_py/our_s1_train.py "${args[@]}" # 20 epoch 💎

python ./training_ct/ct_py/metainvnet_train.py "${args[@]}" # 20 epoch 💎

# TTA
args=(
    --shift dataset
    --pretrain_dataset "deeplesion"
    # --supervised_dataset_name "mayo"
    --supervised_dataset_name "spine"
    # --pretrain_dataset "deeplesion"
    # --supervised_dataset_name "deeplesion"

    
    --epochs 20
    --lr 1e-3
    --batch_size 1
    )

# todo ei

# python ./training_ct/ct_py/EI.py "${args[@]}" # 20 epoch 💎

# python ./training_ct/ct_py/ttt_tta.py "${args[@]}" # 20 epoch 💎

# python ./training_ct/ct_py/our_s2_tta.py "${args[@]}" --lr 5e-4 # 20 epoch 💎
