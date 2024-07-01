This repository provides the official implementation of the **MetaInv-Net**.
# MetaInv-Net
**Haimiao Zhang, Baodong Liu, Hengyong Yu, Bin Dong**

MetaInv-Net: Meta Inversion Network for Sparse View CT Image Reconstruction

*IEEE Transactions on Medical Imaging*, 40(2), 621-634, 2020.



# Install

Here is the list of libraries you need to install to execute the code:

   - [astra-toolbox](https://github.com/astra-toolbox/astra-toolbox)=1.9.0.dev12=py_3.6_numpy_1.16
  - cudnn=7.6.4=cuda9.0_0
  - matplotlib=3.1.1=py36h5429711_0
  - numpy=1.16.5=py36h7e9f1db_0
  - [odl](https://github.com/odlgroup/odl)=0.7.0=py36_0
  - pydicom=1.3.0=py_0
  - python=3.6.9=h265db76_0
  - scikit-image=0.15.0=py36he6710b0_0
  - scipy=1.3.1=py36h7c811a0_0
  - pytorch=1.3.1

These packages can be installed via executing the code, i.e.,

```
conda install python
```



# Training

```python
python main.py
```



# Citation

```
@article{zhang2020metainv,
  title={MetaInv-Net: Meta Inversion Network for Sparse View CT Image Reconstruction},
  author={Zhang, Haimiao and Liu, Baodong and Yu, Hengyong and Dong, Bin},
  journal={IEEE Transactions on Medical Imaging},
  volume={40},
  number={2},
  pages={621--634},
  year={2020},
  publisher={IEEE}
}
```
