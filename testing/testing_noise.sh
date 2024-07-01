

#! usr/bin/env bash

# todo anatomy: knee pd ->brain t2
args=(
    --shift "anatomy"
    --pretrain_dataset "fastmri_knee_pd"
    --supervised_dataset_name "fastmri_brain_t2"
    --sigma 0.01
    --epochs 10 
    --lr 1e-4
    --batch_size 1
    )

python ./testing/figr1_noise.py "${args[@]}"


args=(
    --shift "anatomy"
    --pretrain_dataset "fastmri_knee_pd"
    --supervised_dataset_name "fastmri_brain_t2"
    --sigma 0.02
    --epochs 10 
    --lr 1e-4
    --batch_size 1
    )

# python ./testing/figr1_noise.py "${args[@]}"

args=(
    --shift "anatomy"
    --pretrain_dataset "fastmri_knee_pd"
    --supervised_dataset_name "fastmri_brain_t2"
    --sigma 0.05
    --epochs 10 
    --lr 1e-4
    --batch_size 1
    )

# python ./testing/figr1_noise.py "${args[@]}"