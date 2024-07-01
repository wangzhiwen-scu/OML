

#! usr/bin/env bash

# todo dataset: knee pd ->brain t2


args=(
    --shift "dataset"
    --pretrain_dataset "ixi_t1_periodic_slight_sagittal"
    --supervised_dataset_name "cc_data2_brain_t1"

    --epochs 15 
    --lr 5e-4
    --batch_size 1
    )

# todo our
# python ./training/mri_py/our_s1_train.py "${args[@]}"  --lr 1e-3 # loss stop ❤

# todo TTT
# python ./training/mri_py/main_ttt_train.py "${args[@]}"  --lr 1e-3  # loss stop ❤

# todo md-recon
# python ./training/mri_py/main_mdrecon.py "${args[@]}"  --lr 1e-4 # loss stop ❤

python ./training/mri_py/main_metainvnet.py "${args[@]}"  --lr 1e-3 --epochs 20 # loss stop ❤

# TTA
args=(
    --shift "dataset"
    --pretrain_dataset "ixi_t1_periodic_slight_sagittal"
    --supervised_dataset_name "cc_data2_brain_t1"

    --epochs 10 
    --lr 1e-4
    --batch_size 1
    )

# todo our
# python ./training/mri_py/our_s2_tta.py "${args[@]}" --lr 1e-5 # loss stop ❤

# todo ei
# python ./training/mri_py/main_ei_tta.py "${args[@]}" --lr 5e-4 --epochs 15 # loss stop ❤

# todo TTT
# python ./training/mri_py/main_ttt_tta.py "${args[@]}" --lr 1e-4 # loss stop ❤
