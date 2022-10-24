import torch
import numpy as np
import time
import torch.nn.functional as F
import types

MEAN = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
STD = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
def MTA_loss(model, t_model, xs, labels, alpha, update_steps=2, gamma1=0.01, gamma2=0.01):
    xs_adv = xs.clone().detach().requires_grad_(True)
    for i in range(update_steps):
        outputs = model((xs_adv - MEAN) / STD)
        loss = F.cross_entropy(outputs, labels)
        grad = torch.autograd.grad(loss, xs_adv, retain_graph=True, create_graph=True)[0]


        abs_grad = torch.clamp(torch.abs(grad), min=1e-5)
        l1_norm = torch.sum(abs_grad, dim=[1, 2, 3], keepdim=True)
        grad_1 = grad / l1_norm

        mean_abs_grad = torch.mean(abs_grad, dim=[1, 2, 3], keepdim=True)
        norm_one_grad = grad / mean_abs_grad
        grad_atan = torch.atan(norm_one_grad) * 2 / 3.1415926

        grad_sign = torch.sign(grad)
        norm_grad = grad_1 + gamma1 * grad_sign + gamma2 * grad_atan
        xs_adv = xs_adv + alpha * norm_grad / update_steps 
        xs_adv = torch.clip(xs_adv, 0, 1)
    outputs = t_model((xs_adv - MEAN) / STD)
    return -F.cross_entropy(outputs, labels)
