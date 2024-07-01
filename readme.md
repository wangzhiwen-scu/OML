# OML
Test-time adaption for medical imaging (such as MRI and CT).

This is the official PyTorch implementation of our manuscript:

> [**Test-Time Adaptation via Orthogonal Meta-Learning for Medical Imaging**](xxx)

## Getting started

###  1. Clone the repository
```bash
git clone https://github.com/wangzhiwen-scu/OML.git
cd OML
```

### 2. Install dependencies

Here's a summary of the key dependencies.
- python 3.7
- pytorch 1.7.1
- cuda: 11.0

We recommend using [conda](https://docs.conda.io/en/latest/) to install all of the dependencies.

```bash
conda env create -f environment.yaml
```
To activate the environment, run:

```bash
conda activate OML
```

### 5. Training

Please see [training/mri/anatomy.sh](anatomy.sh) for an example of how to train OML.

### 6. Testing

```
bash ./testing/testing.sh
```

## Todo
Change radon and iradon to CT_LIBV2.

## Acknowledgement

Part of the data simulation are adapted from **EI**. 
Part of the MRI reconstruction network structures are adapted from **MD-Recon-Net**.
Part of the CT reconstruction network structures are adapted from **LEARN**.

+ MD-Recon-Net: [https://github.com/Deep-Imaging-Group/MD-Recon-Net](https://github.com/Deep-Imaging-Group/MD-Recon-Net)
+ LEARN: [https://github.com/maybe198376/LEARN](https://github.com/maybe198376/LEARN)
+ EI: [https://github.com/edongdongchen/EI](https://github.com/edongdongchen/EI).

Thanks a lot for their great works!

## contact
If you have any questions, please feel free to contact Wang Zhiwen {wangzhiwen_scu@163.com}.

 <!-- ## Citation

If you find this project useful, please consider citing:

```bibtex
@article{wang2024promoting,
  title={Promoting fast MR imaging pipeline by full-stack AI},
  author={Wang, Zhiwen and Li, Bowen and Yu, Hui and Zhang, Zhongzhou and Ran, Maosong and Xia, Wenjun and Yang, Ziyuan and Lu, Jingfeng and Chen, Hu and Zhou, Jiliu and others},
  journal={Iscience},
  volume={27},
  number={1},
  year={2024},
  publisher={Elsevier}
}
``` -->