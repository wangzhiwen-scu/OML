# https://github.com/Kthyeon/dnn_orthogonality/blob/main/regularization/orthogonal_weight.py
# Can We Gain More from Orthogonality Regularizations in Training Deep CNNs? ICLR, https://arxiv.org/abs/1810.09102.

import torch
from torch.autograd import Variable
from torch import cuda, nn, optim

def conv_ortho(weight, device):    
    cols = weight[0].numel()
    w1 = weight.view(-1, cols)
    wt = torch.transpose(w1, 0, 1)
    m = torch.matmul(wt, w1)
    ident = Variable(torch.eye(cols, cols)).to(device)

    w_tmp = (m-ident)
    sigma = torch.norm(w_tmp)
    
    return sigma


def my_ortho_norm(model):
    res = None
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # nn.init.orthogonal(m.weight)
            sigma = conv_ortho(m.weight, 'cuda')
            if res is None: 
                res = sigma
            else:
                res = res + sigma

    return res

def ct_ortho_reg(mdl, device, lamb_list=[0.0, 1.0, 0.0, 0.0], opt = 'both'):
# def ortho_reg(mdl, device, lamb_list=[0.0, 1.0, 1.0, 0.0], opt = 'both'):
# def ortho_reg(mdl, device, lamb_list=[0.0, 1.0, 1.0, 0.0], opt = 'both'):

    # Consider the below facotrs.
    # factor1: which kind layer (e.g., pointwise, depthwise, original, fc_layer)
    # factor2: power of regularization (i.e., lambda). Maybe, we should differ from each class of layer's lambda.
    # How do we handle these?
    # 'lamb_list' is a list of hyperparmeters for each class of layer. [origin_conv, pointwise, depthwise, fully coneected layer]
    # 'lamb_list' length is 4.
    # 'opt' : position of pointwise convolution. - expansion stage (exp), reduction stage (rec), both.
    
    assert type(lamb_list) is list, 'lamb_list should be list.'
    l2_reg = None

    for module in mdl.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if isinstance(module, nn.Conv2d):
                if module.weight.shape[2] == 1:
                    if opt == 'exp' and module.weight.shape[0]<module.weight.shape[1]:
                        continue
                    elif opt == 'rec' and module.weight.shape[0]>module.weight.shape[1]:
                        continue
                    # pointwise conv
                    W = module.weight
                    lamb = lamb_list[1]
                elif module.weight.shape[2] ==5 or module.weight.shape[2] ==3:
                    W = module.weight
                    if module.weight.shape[1] == 1:
                    # if module.weight.shape[1] == 2:
                        # Depthwise convolution.
                        lamb = lamb_list[2]
                    else:
                        # Original convolution. (Maybe including stem conv)
                        lamb = lamb_list[0]
            elif isinstance(module, nn.Linear):
                # fully connected layer
                W = module.weight
                lamb = lamb_list[3]
            else:
                continue

            cols = W[0].numel() 
            w1 = W.view(-1, cols) # out_channels x all
            wt = torch.transpose(w1, 0, 1)
            m = torch.matmul(wt, w1)

            w_tmp = (m-torch.diag(m))
#             height = w_tmp.size(0)
#             u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
#             v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
#             u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
#             sigma = torch.dot(u, torch.matmul(w_tmp, v))
            sigma = torch.norm(w_tmp)
            if l2_reg is None:
                l2_reg = lamb * (sigma)**2
                num = 1
            else:
                l2_reg += lamb * (sigma)**2
                num += 1
        else:
            continue

    return l2_reg / num


def ortho_reg(mdl, device, lamb_list=[0.0, 1.0, 0.0, 0.0], opt = 'both'):
# def ortho_reg(mdl, device, lamb_list=[0.0, 1.0, 1.0, 0.0], opt = 'both'):
# def ortho_reg(mdl, device, lamb_list=[0.0, 1.0, 1.0, 0.0], opt = 'both'):

    # Consider the below facotrs.
    # factor1: which kind layer (e.g., pointwise, depthwise, original, fc_layer)
    # factor2: power of regularization (i.e., lambda). Maybe, we should differ from each class of layer's lambda.
    # How do we handle these?
    # 'lamb_list' is a list of hyperparmeters for each class of layer. [origin_conv, pointwise, depthwise, fully coneected layer]
    # 'lamb_list' length is 4.
    # 'opt' : position of pointwise convolution. - expansion stage (exp), reduction stage (rec), both.
    
    assert type(lamb_list) is list, 'lamb_list should be list.'
    l2_reg = None

    for module in mdl.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if isinstance(module, nn.Conv2d):
                if module.weight.shape[2] == 1:
                    if opt == 'exp' and module.weight.shape[0]<module.weight.shape[1]:
                        continue
                    elif opt == 'rec' and module.weight.shape[0]>module.weight.shape[1]:
                        continue
                    # pointwise conv
                    W = module.weight
                    lamb = lamb_list[1]
                elif module.weight.shape[2] ==3:
                    W = module.weight
                    # if module.weight.shape[1] == 1:
                    if module.weight.shape[1] == 2:
                        # Depthwise convolution.
                        lamb = lamb_list[2]
                    else:
                        # Original convolution. (Maybe including stem conv)
                        lamb = lamb_list[0]
            elif isinstance(module, nn.Linear):
                # fully connected layer
                W = module.weight
                lamb = lamb_list[3]
            else:
                continue

            cols = W[0].numel() 
            w1 = W.view(-1, cols) # out_channels x all
            wt = torch.transpose(w1, 0, 1)
            m = torch.matmul(wt, w1)

            w_tmp = (m-torch.diag(m))
#             height = w_tmp.size(0)
#             u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
#             v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
#             u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
#             sigma = torch.dot(u, torch.matmul(w_tmp, v))
            sigma = torch.norm(w_tmp)
            if l2_reg is None:
                l2_reg = lamb * (sigma)**2
                num = 1
            else:
                l2_reg += lamb * (sigma)**2
                num += 1
        else:
            continue

    return l2_reg / num


def or_reg(mdl, device, lamb_list=[0.0, 1.0, 0.0, 0.0], opt = 'both'):
# def or_reg(mdl, device, lamb_list=[1.0, 1.0, 1.0, 1.0], opt = 'both'):
    # Consider the below facotrs.
    # factor1: which kind layer (e.g., pointwise, depthwise, original, fc_layer)
    # factor2: power of regularization (i.e., lambda). Maybe, we should differ from each class of layer's lambda.
    # How do we handle these?
    # 'lamb_list' is a list of hyperparmeters for each class of layer. [origin_conv, pointwise, depthwise, fully coneected layer]
    # 'lamb_list' length is 4.
    # 'opt' : position of pointwise convolution. - expansion stage (exp), reduction stage (rec), both.
    
    assert type(lamb_list) is list, 'lamb_list should be list.'
    l2_reg = None

    for module in mdl.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if isinstance(module, nn.Conv2d):
                if module.weight.shape[2] == 1:
                    if opt == 'exp' and module.weight.shape[0]<module.weight.shape[1]:
                        continue
                    elif opt == 'rec' and module.weight.shape[0]>module.weight.shape[1]:
                        continue
                    # pointwise conv
                    W = module.weight
                    lamb = lamb_list[1]
                elif module.weight.shape[2] ==3:
                    W = module.weight
                    if module.weight.shape[1] == 1:
                        # Depthwise convolution.
                        lamb = lamb_list[2]
                    else:
                        # Original convolution. (Maybe including stem conv)
                        lamb = lamb_list[0]
            elif isinstance(module, nn.Linear):
                # fully connected layer
                W = module.weight
                lamb = lamb_list[3]
            else:
                continue

            cols = W[0].numel() 
            w1 = W.view(-1, cols) # out_channels x all
            wt = torch.transpose(w1, 0, 1)
            m = torch.matmul(wt, w1)
            
#             w_tmp = (m - torch.diagflat(torch.diagonal(m)))
#             height = w_tmp.size(0)
#             u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
#             v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
#             u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
#             sigma = torch.dot(u, torch.matmul(w_tmp, v))
            row_sigma = torch.norm(m, p=1, dim=1)
            w_tmp = row_sigma - torch.ones_like(row_sigma)
            
            sigma = torch.norm(w_tmp, p=1)
            
            if l2_reg is None:
                l2_reg = lamb * (sigma)
                num = 1
            else:
                l2_reg += lamb * (sigma)
                num += 1
        else:
            continue

    return l2_reg / num

if __name__ == "__main__":
        # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(3, 4)
            self.fc2 = nn.Linear(4, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
        
    model = SimpleModel()

    # loss = ortho_reg(model, "cuda")
    # loss = or_reg(model, "cuda")
    loss = my_ortho_norm(model)
    

    x = 1