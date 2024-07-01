#! usr/bin/env bash

# todo anatomy: knee pd ->brain t2
args=(
    --shift "anatomy"
    --pretrain_dataset "mayo"
    --supervised_dataset_name "cta"

    --epochs 20
    --lr 5e-4
    --batch_size 1
    )

# python ./testing_r1/fig2_ct_r1.py "${args[@]}"


# todo dataset: knee pd ->brain t2
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
# python ./testing_r1/fig2_ct_r1.py "${args[@]}"

# todo modality: brain t2 -> brain t1

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

python ./testing_r1/fig2_ct_r1.py "${args[@]}"
