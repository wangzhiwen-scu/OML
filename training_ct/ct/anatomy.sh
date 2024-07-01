#! usr/bin/env bash

# python ./training_ct/ct_ei_tta.py "${args[@]}" 
# python ./training_ct/new_ct_ei_tta.py "${args[@]}" 
# python ./training_ct/EI_ctlib.py "${args[@]}" 
# python ./training_ct/EI_wtct.py "${args[@]}" #

# todo anatomy: knee pd ->brain t2

args=(
    --shift anatomy
    --pretrain_dataset "mayo"
    --supervised_dataset_name "cta"
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
    --shift anatomy
    --pretrain_dataset "mayo"
    --supervised_dataset_name "cta"
    # --pretrain_dataset "deeplesion"
    # --supervised_dataset_name "deeplesion"

    
    --epochs 20
    --lr 5e-4
    --batch_size 1
    )

# todo ei

# python ./training_ct/ct_py/EI.py "${args[@]}" --lr 1e-3 # 20 epoch 💎

# python ./training_ct/ct_py/ttt_tta.py "${args[@]}"  --lr 1e-3 # 20 epoch 💎

# python ./training_ct/ct_py/our_s2_tta.py "${args[@]}" # 20 epoch 💎