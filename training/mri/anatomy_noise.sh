#! usr/bin/env bash

# todo anatomy: knee pd ->brain t2

# TTA --sigma 0.01 0.02 0.05
args=(
    --shift "anatomy"
    --pretrain_dataset "fastmri_knee_pd"
    --supervised_dataset_name "fastmri_brain_t2"
    --sigma 0.01
    --epochs 10 
    --lr 1e-4
    --batch_size 1
    )
device=0

# todo our
# python ./training/mri_py_noise/noise_our_s2_tta.py "${args[@]}" --lr 1e-5 # loss stop ❤

# todo ei
# python ./training/mri_py_noise/noise_ei_tta.py "${args[@]}" --lr 5e-4 --epochs 20 # loss stop ❤

# todo TTT
# python ./training/mri_py_noise/noise_ttt_tta.py "${args[@]}" --lr 1e-4 # loss stop ❤


args=(
    --shift "anatomy"
    --pretrain_dataset "fastmri_knee_pd"
    --supervised_dataset_name "fastmri_brain_t2"
    --sigma 0.02
    --epochs 10 
    --lr 1e-4
    --batch_size 1
    )
device=0

# # todo our
# python ./training/mri_py_noise/noise_our_s2_tta.py "${args[@]}" --lr 1e-5 # loss stop ❤

# # todo ei
# python ./training/mri_py_noise/noise_ei_tta.py "${args[@]}" --lr 5e-4 --epochs 20 # loss stop ❤

# # todo TTT
# python ./training/mri_py_noise/noise_ttt_tta.py "${args[@]}" --lr 1e-4 # loss stop ❤


args=(
    --shift "anatomy"
    --pretrain_dataset "fastmri_knee_pd"
    --supervised_dataset_name "fastmri_brain_t2"
    --sigma 0.05
    --epochs 10 
    --lr 1e-4
    --batch_size 1
    )
device=0

# # todo our
python ./training/mri_py_noise/noise_our_s2_tta.py "${args[@]}" --lr 1e-5 # loss stop ❤

# # todo ei
python ./training/mri_py_noise/noise_ei_tta.py "${args[@]}" --lr 5e-4 --epochs 20 # loss stop ❤

# # todo TTT
python ./training/mri_py_noise/noise_ttt_tta.py "${args[@]}" --lr 1e-4 # loss stop ❤