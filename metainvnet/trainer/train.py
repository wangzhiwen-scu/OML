from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from data_loader import DatasetLoader
from torch.utils.data.dataloader import DataLoader
from model import l2loss_mean
import torch
import numpy as np

import sys

sys.path.append('.')
from metainvnet.utils import ssim_loss

class Trainer():
    def __init__(self, args, model, tr_dset):
        self.args = args
        self.model = model
        self.epoch = args.epoch
        self.bat_size = args.tr_batch
        self.tr_dset = tr_dset

        self.proj, self.back_proj, _, self.op_norm = self.tr_dset.radon_op()

    def _set_optim(self):
        optimizer_cg = optim.Adam(self.model.net.parameters(), lr=1e-3)
        scheduler_cg = MultiStepLR(optimizer_cg, milestones=[30,40], gamma=0.9)
        return optimizer_cg, scheduler_cg

    def _set_dataloader(self):
        print('Load training data')
        self.img, self.u0, self.sino = self.tr_dset.get_imgs()
        DataSetLoader = DatasetLoader(self.img, self.u0, self.sino)
        DLoader = DataLoader(dataset=DataSetLoader, num_workers=4, drop_last=True, batch_size=self.bat_size)
        return DLoader

    def tr(self):
        self.optimizer, self.scheduler = self._set_optim()
        DLoader = self._set_dataloader()
        n_iter = 0
        for epoch in range(self.epoch):

            for n_count, data_batch in enumerate(DLoader):
                self.optimizer.zero_grad()
                batch_img, batch_u0, batch_sino = data_batch[0].cuda(), data_batch[1].cuda(), data_batch[2].cuda()

                batch_x_db = self.model.tr_model(batch_sino, batch_u0, batch_img, self.radon, self.iradon, self.op_norm)
                loss = self.loss_fun(batch_x_db, batch_img)
                loss.backward()
                self.optimizer.step()

        self.save_ckp(epoch)

    def loss_fun(self, db, sp):
        layer = len(db)

        loss = 0.0
        for ii in range(0, layer):
            loss =loss+ l2loss_mean(sp, db[ii]) * 1.1**ii

        for ii in range(0, layer):
            for jj in range(sp.shape[1]):
                img1=sp[jj,...]
                img2=db[ii][jj,...]

                ssim_value=1-ssim_loss(img1, img2,size=11,sigma = 1.5)
                loss=loss+ssim_value*100.0

        return loss

    def save_ckp(self, epoch):
        state = {'model': self.model.net.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict()}
        torch.save(state,'./results/epoch')

    def radon(self,img):
        if len(img.shape)==4:
            img=img.squeeze(1)
        return self.proj(img).unsqueeze(1)
    def iradon(self,sino):
        if len(sino.shape)==4:
            sino=sino.squeeze(1)
            
        return self.back_proj(sino/self.op_norm).unsqueeze(1)
