import torch
from thop import profile
import time
import sys

sys.path.append('.') # 
from model_backbones.recon_net import Hand_Tailed_Mask_Layer
from model_backbones.dualdomain import DualDoRecNet
from ct_dependencies.learn_ei import LEARN
from ct_dependencies.EI_dependencies.ct import CT

def compute_ct_metrics(model, input_tensor, fbp_mpg, device="cuda"):
    # Ensure model and tensor are on the correct device
    model.to(device)
    input_tensor = input_tensor.to(device)
    fbp_mpg = fbp_mpg.to(device)
    # input_tensor = (input_tensor, input_tensor, input_tensor[:,0:1,:,:])

    # FLOPs (G)
    # flops, params = profile(model, inputs=(input_tensor,))
    flops, params = profile(model, inputs=(input_tensor, fbp_mpg))
    flops_giga = flops / 1e9

    # Parameters (M)
    total_params = sum(p.numel() for p in model.parameters())
    params_mega = total_params / 1e6

    # Inference Time (ms)
    model.eval()
    inputs=(input_tensor, fbp_mpg)
    with torch.no_grad():
        start_time = time.time()
        _ = model(*inputs)
        end_time = time.time()

    elapsed_time_ms = (end_time - start_time) * 1000

    return flops_giga, params_mega, elapsed_time_ms

def compute_mri_metrics(model, input_tensor, device="cuda"):
    # Ensure model and tensor are on the correct device
    model.to(device)
    input_tensor = input_tensor.to(device)
    # input_tensor = (input_tensor, input_tensor, input_tensor[:,0:1,:,:])

    # FLOPs (G)
    # flops, params = profile(model, inputs=(input_tensor,))
    flops, params = profile(model, inputs=(input_tensor, input_tensor, input_tensor[:,0:1,:,:]))
    flops_giga = flops / 1e9

    # Parameters (M)
    total_params = sum(p.numel() for p in model.parameters())
    params_mega = total_params / 1e6

    # Inference Time (ms)
    model.eval()
    input_tensor3 = (input_tensor, input_tensor, input_tensor[:,0:1,:,:])
    with torch.no_grad():
        start_time = time.time()
        _ = model(*input_tensor3)
        end_time = time.time()

    elapsed_time_ms = (end_time - start_time) * 1000

    return flops_giga, params_mega, elapsed_time_ms

# Usage
# model = DualDoRecNet()  # Replace with your PyTorch model
# input_tensor = torch.randn(1, 2, 256, 256)  # Adjust according to your model's input shape
# flops, params, time_ms = compute_mri_metrics(model, input_tensor)

options = torch.tensor([60, 640, 256, 256, 0.006641, 0.012858, 0, 0.009817477*640/30,
        5.95, 4.906, 0, 0]).cuda()


model = LEARN(options, block_num=10)
input_tensor = torch.randn(1, 1, 256, 256)  # Adjust according to your model's input shape
physics = CT(256, 60, circle=False, device="cuda")
meas0 = physics.A(input_tensor)
# s_mpg = torch.log(physics.I0 / meas0)
fbp_mpg = physics.A_dagger(meas0)
flops, params, time_ms = compute_ct_metrics(model, input_tensor, meas0)

print(f"FLOPs: {flops:.2f} G")
print(f"Parameters: {params:.2f} M")
print(f"Inference Time: {time_ms:.2f} ms")
