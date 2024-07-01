

#! usr/bin/env bash

# todo anatomy: knee pd ->brain t2
args=(
    --shift "anatomy"
    --pretrain_dataset "fastmri_knee_pd"
    --supervised_dataset_name "fastmri_brain_t2"
    --epochs 10 
    --lr 1e-4
    --batch_size 1
    )

# python ./testing/fig3_mri_ablation.py "${args[@]}"
python ./testing/fig3_mri_ablation_r1.py "${args[@]}"


# todo dataset: knee pd ->brain t2
args=(
    --shift dataset
    --pretrain_dataset "ixi_t1_periodic_slight_sagittal"
    --supervised_dataset_name "cc_data2_brain_t1"
    
    --epochs 10 
    --lr 1e-4
    --batch_size 1
    )

# python ./testing/fig1_mri.py "${args[@]}"

# todo modality: brain t2 -> brain t1

args=(
    --shift modality
    --pretrain_dataset "fastmri_brain_t2"
    --supervised_dataset_name "fastmri_brain_t1"
    --epochs 20 
    --lr 1e-4
    --batch_size 1
    )

# python ./testing/fig1_mri.py "${args[@]}"

# todo ratio: brain t2 -> brain t1

args=(

    --shift "ratio"
    --pretrain_dataset "fastmri_brain_t2"

    # --supervised_dataset_name "cc_data2_brain_t1"

    # --supervised_dataset_name "fastmri_brain_pd"
    # --supervised_dataset_name "fastmri_knee_pdfs"
    --supervised_dataset_name "fastmri_brain_t2"
    # --supervised_dataset_name "fastmri_brain_t1"
    # --supervised_dataset_name "fastmri_knee_t1"
    --acc 4
    --epochs 20 
    --lr 1e-4
    --batch_size 1
    )

# python ./testing/fig1_mri.py "${args[@]}"