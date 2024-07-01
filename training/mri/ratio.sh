

#! usr/bin/env bash

# todo ratio : brain t2 -> brain t1


args=(
    --shift "ratio"
    --pretrain_dataset "fastmri_brain_t2"

    # --supervised_dataset_name "ixi_t1_periodic_slight_sagittal"
    
    # --supervised_dataset_name "fastmri_knee_pdfs"

    # --supervised_dataset_name "fastmri_brain_t1"

    --supervised_dataset_name "fastmri_brain_t2"
    --acc 10
    --epochs 30 
    --lr 5e-4
    --batch_size 1
    )

# todo our
# python ./training/mri_py/our_s1_train.py "${args[@]}"  --lr 5e-4 # loss stop ❤

# todo TTT
# python ./training/mri_py/main_ttt_train.py "${args[@]}"  --lr 1e-3 --epochs 30 # loss stop ❤

# todo md-recon
# python ./training/mri_py/main_mdrecon.py "${args[@]}"  --lr 1e-3 --epochs 15  # loss stop ❤

# python ./training/mri_py/main_metainvnet.py "${args[@]}"  --lr 1e-3 --epochs 8 # loss stop ❤
python ./training/mri_py/main_metainvnet.py "${args[@]}"  --lr 1e-3 --epochs 2 --load_model 1 # loss stop ❤

# TTA
args=(
    --shift "ratio"

    --pretrain_dataset "fastmri_brain_t2"
    # --supervised_dataset_name "cc_data2_brain_t1"
    # --supervised_dataset_name "fastmri_brain_pd"
    # --supervised_dataset_name "fastmri_knee_pdfs"
    --supervised_dataset_name "fastmri_brain_t2"
    # --supervised_dataset_name "fastmri_brain_t1"
    # --supervised_dataset_name "fastmri_knee_t1"
    # --acc 2
    --acc 4
    --epochs 20
    --lr 1e-4
    --batch_size 1
    )

# todo our
# python ./training/mri_py/our_s2_tta.py "${args[@]}" --lr 5e-6  --epochs 15 # loss stop ❤ --lr 1e-6 

# todo ei
# python ./training/mri_py/main_ei_tta.py "${args[@]}" --lr 5e-4 --epochs 20 # loss stop ❤

# todo TTT
# python ./training/mri_py/main_ttt_tta.py "${args[@]}" --lr 1e-4 # loss stop ❤
