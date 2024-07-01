import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.models as models
import ctlib_v2
import numpy


# class bfan_ed(Function):
#
#     def forward(self, input_data, weight, prj, options):
#         dets = int(options[1])
#         dDet = options[5]
#         s2r = options[7]
#         d2r = options[8]
#         self.virdet = dDet * s2r / (s2r + d2r)
#         filter = torch.empty(2 * dets - 1)
#         pi = torch.acos(torch.tensor(-1.0))
#         for i in range(filter.size(0)):
#             x = i - dets + 1
#             if abs(x) % 2 == 1:
#                 filter[i] = -1 / (pi * pi * x * x * self.virdet * self.virdet)
#             elif x == 0:
#                 filter[i] = 1 / (4 * self.virdet * self.virdet)
#             else:
#                 filter[i] = 0
#         self.w = torch.arange((-dets / 2 + 0.5) * self.virdet, dets / 2 * self.virdet, self.virdet)
#         self.w = s2r / torch.sqrt(s2r ** 2 + self.w ** 2)
#         self.w = self.w.view(1, 1, 1, -1).cuda()
#         self.filter = self.filter.view(1, 1, 1, -1).cuda()
#         self.options = nn.Parameter(options, requires_grad=False)
#         self.coef = pi / options[0]
#         p = prj * self.virdet * self.w
#          p = torch.nn.functional.conv2d(p, self.filter, padding=(0, self.dets - 1))
#         recon = bprj_fun.apply(p, self.options)
#         recon = recon * self.coef
#         # p = p.squeeze()
#
#         return recon

def filtered(prj, options):
    dets = int(options[1])
    dDet = options[5]
    s2r = options[7].cuda()
    d2r = options[8]
    virdet = dDet * s2r / (s2r + d2r)
    filter = torch.empty(2 * dets - 1)
    pi = torch.acos(torch.tensor(-1.0))
    for i in range(filter.size(0)):
        x = i - dets + 1
        if abs(x) % 2 == 1:
            filter[i] = -1 / (pi * pi * x * x * virdet * virdet)
        elif x == 0:
            filter[i] = 1 / (4 * virdet * virdet)
        else:
            filter[i] = 0
    w = torch.arange((-dets / 2 + 0.5) * virdet, dets / 2 * virdet, virdet).cuda()
    w = s2r / torch.sqrt(s2r ** 2 + w ** 2)
    w = w.view(1, 1, 1, -1).cuda()
    filter = filter.view(1, 1, 1, -1).cuda()
    # self.options = nn.Parameter(options, requires_grad=False)
    coef = pi / options[0]
    p = prj * virdet * w * coef
    p = torch.nn.functional.conv2d(p, filter, padding=(0, dets - 1))
    p = p.squeeze()
    return p


class bprj_fun(Function):
    @staticmethod
    def forward(self, proj, options):
        self.save_for_backward(options)
        # print(self.saved_tensors)
        return ctlib_v2.fbpbackprojection(proj, options)

    @staticmethod
    def backward(self, grad_output):
        options = self.saved_tensors[0]
        grad_input = ctlib_v2.fbpprojection(grad_output.contiguous(), options)
        return grad_input, None


class prj_fun(Function):
    @staticmethod
    def forward(self, input_data, options):
        self.save_for_backward(options)
        temp_prj = ctlib_v2.fbpprojection(input_data, options)
        return temp_prj

    @staticmethod
    def backward(self, grad_output):
        # print(self.saved_tensors)
        options = self.saved_tensors[0]
        grad_input = ctlib_v2.fbpbackprojection(grad_output.contiguous(), options)
        return grad_input, None


class prj_module(nn.Module):
    def __init__(self):
        super(prj_module, self).__init__()
        self.weight_fed = nn.Parameter(torch.Tensor(1))
        self.w = nn.Parameter(torch.Tensor([[-0.8955, 0, -2.6443, 0.5331, 1.5326, 0, 0]]).t(), requires_grad=False)
        self.b = nn.Parameter(torch.Tensor([1.0363]), requires_grad=False)

        # self.w = nn.Parameter(torch.Tensor([[-1.6743, 0, 0, -1.2753, 0, -0.2185, 0]]).t(), requires_grad=False)
        # self.b = nn.Parameter(torch.Tensor([1.9305]), requires_grad=False)

    def forward(self, input_data, proj, options, gamma, beta, feature_vec):
        weight = self.weight_fed * gamma + beta
        weight.unsqueeze_(-1)
        maxval = torch.mm(feature_vec, self.w) + self.b
        le = weight.detach().le(maxval).float()
        w = le * weight + (1 - le) * maxval
        clamped_weight = w.clamp_min(0.0).view(input_data.size(0), 1, 1, 1)

        temp_prj = prj_fun.apply(input_data, options)
        temp = []
        for i in range(input_data.size(0)):
            temp.append(filtered(temp_prj[i] - proj[i], options[i]))
        intervening_res = bprj_fun.apply(temp, options)
        out = input_data - clamped_weight * intervening_res
        return out


# class prj_fun_(Function):
#     @staticmethod
#     def forward(self, input_data, weight, proj, options):
#         temp_prj = ctlib_v2.projection(input_data, options)
#         temp = []
#         for i in range(input_data.size(0)):
#             temp.append(temp_prj[i] - proj[i])
#         ###filter
#         intervening_res = []
#         for i in range(input_data.size(0)):
#             intervening_res[i] = bfan_ed(temp[i], options[i]).float()
#
#         #intervening_res = ctlib_v2.backprojection(temp, options).float()
#         self.save_for_backward(intervening_res, weight, options)
#         out = input_data - weight * intervening_res
#         return out
#
#     @staticmethod
#     def backward(self, grad_output):
#         intervening_res, weight, options = self.saved_tensors
# #        temp = ctlib_v2.projection(grad_output, options)
#         #temp = bfan_ed.backward(grad_output,options)
#         temp = ctlib_v2.backprojection(grad_output, options)
#
#         grad_input = grad_output - weight * temp
#         temp = intervening_res * grad_output
#         grad_weight = - temp.sum((2,3), keepdim=True)
#         return grad_input, grad_weight, None, None

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x, gamma_vec, beta_vec):
        gamma0 = gamma_vec[:, 0:64].view(x.size(0), 64, 1, 1)
        beta0 = beta_vec[:, 0:64].view(x.size(0), 64, 1, 1)
        gamma1 = gamma_vec[:, 64:128].view(x.size(0), 64, 1, 1)
        beta1 = beta_vec[:, 64:128].view(x.size(0), 64, 1, 1)
        gamma2 = gamma_vec[:, 128].view(x.size(0), 1, 1, 1)
        beta2 = beta_vec[:, 128].view(x.size(0), 1, 1, 1)

        out = self.conv1(x)
        out = out * gamma0 + beta0
        out = self.relu1(out)

        out = self.conv2(out)
        out = out * gamma1 + beta1
        out = self.relu2(out)

        out = self.conv3(out)
        out = out * gamma2 + beta2
        return out


class IterBlock(nn.Module):
    def __init__(self):
        super(IterBlock, self).__init__()
        self.block1 = prj_module()
        self.block2 = ConvBlock()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_data, proj, options, gamma_vec, beta_vec, feature_vec):

        tmp1 = self.block1(input_data, proj, options, gamma0, beta0, feature_vec)
        tmp2 = self.block2(input_data, gamma1, beta1)
        output = tmp1 + tmp2
        output = self.relu(output)
        return output


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, 260, bias=True)
        )

    def forward(self, x):
        out = self.model(x)
        out1 = out[:, :130]
        out2 = out[:, 130:]
        return out1, out2


class Learn(nn.Module):
    def __init__(self, block_num):
        super(Learn, self).__init__()
        self.model = nn.ModuleList([IterBlock() for i in range(block_num)])
        # self.MLP = MLP()

    def forward(self, input_data, proj, options, feature_vec):
        x = input_data
        # gamma, beta = self.MLP(feature_vec)
        for index, module in enumerate(self.model):
            x = module(x, proj, options, feature_vec)
        return x


if __name__ == "__main__":
    net = Learn(block_num=50)
    # for key in net.state_dict():
    #     print(key)

    for name, param in net.named_parameters():
        print(name)