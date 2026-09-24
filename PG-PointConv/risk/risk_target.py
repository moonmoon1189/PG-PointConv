import torch

def compute_risk_target(delta_h, w, obs_mask, H_max=0.015, W_min=0.80, gamma_h=5.0, gamma_w=2.0):
    h_pen = gamma_h * torch.clamp(delta_h - H_max, min=0) ** 2
    w_pen = gamma_w * torch.clamp(W_min - w, min=0) ** 2
    r_phys = torch.clamp(h_pen + w_pen, max=1.0) * obs_mask
    return r_phys.detach()