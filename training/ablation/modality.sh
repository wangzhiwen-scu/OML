

#! usr/bin/env bash

# todo modality: brain t2 -> brain t1


args=(
    --shift "dataset"
    # --supervised_dataset_name "ixi_t1_periodic_slight_sagittal"
    
    # --supervised_dataset_name "fastmri_knee_pdfs"

    # --supervised_dataset_name "fastmri_brain_t1"

    --supervised_dataset_name "fastmri_brain_t2"

    --epochs 20 
    --lr 5e-4
    --batch_size 1
    )

# python ./training/ablation_py/tta_sia_stage1.py "${args[@]}"  --lr 1e-3 # loss stop ❤


# python ./training/ablation_py/tta_stage1.py "${args[@]}"  --lr 1e-3 # loss stop ❤

# TTA
args=(
    --shift "dataset"
    --pretrain_dataset "fastmri_brain_t2"

    # --supervised_dataset_name "cc_data2_brain_t1"

    # --supervised_dataset_name "fastmri_brain_pd"
    # --supervised_dataset_name "fastmri_knee_pdfs"
    # --supervised_dataset_name "fastmri_brain_t2"
    --supervised_dataset_name "fastmri_brain_t1"
    # --supervised_dataset_name "fastmri_knee_t1"
    # --acc 2
    --epochs 10 
    --lr 1e-4
    --batch_size 1
    )

# python ./training/ablation_py/tta_sia_stage2.py "${args[@]}" --lr 1e-5 # loss stop ❤

# python ./training/ablation_py/tta_stage2.py "${args[@]}" --lr 1e-5 # loss stop ❤

python ./training/ablation_py/upperbound.py "${args[@]}" --lr 1e-3 # loss stop ❤

