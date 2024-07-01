# https://github1s.com/jianzhangcs/ISTA-Net-PyTorch/blob/master/Train_MRI_CS_ISTA_Net_plus.py#L21
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

def rfft(input):
    fft = torch.rfft(input, 2, onesided=False)  # Real-to-complex Discrete Fourier Transform. normalized =False
    fft = fft.squeeze(1)
    fft = fft.permute(0, 3, 1, 2)
    return fft  

def fft(input):
    # (N, 2, W, H) -> (N, W, H, 2)
    # print(type(input))
    input = input.permute(0, 2, 3, 1)
    input = torch.fft(input, 2, normalized=False) # !

    # (N, W, H, 2) -> (N, 2, W, H)
    input = input.permute(0, 3, 1, 2)
    return input

def ifft(input):
    input = input.permute(0, 2, 3, 1)
    input = torch.ifft(input, 2, normalized=False) # !
    # (N, D, W, H, 2) -> (N, 2, D, W, H)
    input = input.permute(0, 3, 1, 2)
    return input

class FFT_Mask_ForBack(torch.nn.Module):
    def __init__(self):
        super(FFT_Mask_ForBack, self).__init__()

    def forward(self, x, mask):
        """_summary_

        Args:
            x (_type_): f_img
            mask (_type_): undersampling

        Returns:
            _type_: u_img
        """
        # f_k = rfft(x)
        f_k = fft(x)
        z_hat = ifft(f_k * mask) # z_hat = u_k
        zero_recon = torch.norm(z_hat, dim=1, keepdim=True)  # N1HW 默认求2范数：所有元素平方开根号 也就是 abs(complex_img?)
        return zero_recon

# Define ISTA-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        # self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        # self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))

    def forward(self, x, fft_forback, PhiTb, mask):
        """x, self.fft_forback, PhiTb, mask
        Phi: is oberservation matrix, which can be replaced by mask*fft or ifft

        Args:
            x (_type_): x(init) is Phib, is rec_u_img
            fft_forback (_type_): get a results: ifft(mask*fft(x))
            PhiTb (_type_): = PhiTy = zero_filling
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        # x(k) = x(k-1)-rho*PhiTPhix(k-1)+rho*PhiTy
        x = x - self.lambda_step * fft_forback(x, mask) # PhiTPhi, PhiTb即PhiTy即zero_filling_img, 
        x = x + self.lambda_step * PhiTb
        x_input = x

        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)  # 2层神经网络做为稀疏变换。

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))  #  软阈值

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = x_input + x_G

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)  # 这里是为了计算F F-1 可逆。
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]

# Define ISTA-Net-plus
class ISTANetplus(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANetplus, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.fft_forback = FFT_Mask_ForBack()

        for i in range(LayerNo):
            onelayer.append(BasicBlock())
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, PhiTb, mask):
        """PhiTb = iff(mask * f_k) = zero_filling_img
        mask = sampling_mask

        Args:
            PhiTb (_type_): _description_
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        

        x = PhiTb

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, self.fft_forback, PhiTb, mask)
            layers_sym.append(layer_sym)
        x_final = x
        return [x_final, layers_sym]

def ista_net_loss(x_output, loss_layers_sym, batch_x):
    # Compute and print loss
    layer_num = 9
    loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))
    loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
    for k in range(layer_num-1):
        loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gamma = torch.Tensor([0.01]).to(device)
    # loss_all = loss_discrepancy
    loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)
    return loss_all


def ista_net_plus_main(start_epoch, end_epoch, rand_loader, device, mask, layer_num, optimizer, log_file_name, model_dir):
    # Training loop
    model = 1
    for epoch_i in range(start_epoch+1, end_epoch+1):
        for data in rand_loader:

            batch_x = data
            batch_x = batch_x.to(device)
            batch_x = batch_x.view(batch_x.shape[0], 1, batch_x.shape[1], batch_x.shape[2])

            PhiTb = FFT_Mask_ForBack()(batch_x, mask)  # PhiTb is zero_filling_img
            [x_output, loss_layers_sym] = model(PhiTb, mask)

            # Compute and print loss
            loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))
            loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
            for k in range(layer_num-1):
                loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))
            gamma = torch.Tensor([0.01]).to(device)
            # loss_all = loss_discrepancy
            loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

        #     # Zero gradients, perform a backward pass, and update the weights.
        #     optimizer.zero_grad()
        #     loss_all.backward()
        #     optimizer.step()

        #     output_data = "[%02d/%02d] Total Loss: %.5f, Discrepancy Loss: %.5f,  Constraint Loss: %.5f\n" % (epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item(), loss_constraint)
        #     print(output_data)

        # output_file = open(log_file_name, 'a')
        # output_file.write(output_data)
        # output_file.close()

        # if epoch_i % 5 == 0:
        #     torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters