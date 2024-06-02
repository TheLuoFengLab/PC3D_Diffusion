import torch
import numpy as np
from typing import Tuple

@torch.jit.script
def quatnormalize(q: torch.Tensor) -> torch.Tensor:
    q = (1-2*(q[...,3:4]<0).to(q.dtype))*q
    return q / q.norm(p=2, dim=-1, keepdim=True)
    
@torch.jit.script
def axang2quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    theta = angle / 2
    axis = axis / (axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9))
    xyz = axis * torch.sin(theta)
    w = torch.cos(theta)
    return quatnormalize(torch.cat((xyz, w), -1))

@torch.jit.script
def quatdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w = (a*b).sum(-1).add_(1)
    xyz = torch.linalg.cross(a, b)
    q = torch.cat((xyz, w.unsqueeze_(-1)), -1)
    return quatnormalize(q)

@torch.jit.script
def wrap2pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def quat2axang(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w = q[..., 3]

    sin = torch.sqrt(1 - w * w)
    mask = sin > 1e-5

    angle = 2 * torch.acos(w)
    angle = wrap2pi(angle)
    axis = q[..., 0:3] / sin.unsqueeze_(-1)

    z_axis = torch.zeros_like(axis)
    z_axis[..., -1] = 1

    angle = torch.where(mask, angle, z_axis[...,0]).unsqueeze_(-1)
    axis = torch.where(mask.unsqueeze_(-1), axis, z_axis)
    return axis, angle

@torch.jit.script
def quat2expmap(q: torch.Tensor) -> torch.Tensor:
    ax, ang = quat2axang(q)
    return ax.mul_(ang)

@torch.jit.script
def expmap2axang(expmap):
    min_theta = 1e-5
    angle = torch.norm(expmap, dim=-1)
    angle_exp = torch.unsqueeze(angle, dim=-1)
    axis = expmap / angle_exp
    angle = wrap2pi(angle)
    default_axis = torch.zeros_like(expmap)
    default_axis[..., -1] = 1
    mask = angle > min_theta
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return axis, angle.unsqueeze(-1)

@torch.jit.script
def expmap2quat(expmap):
    axis, angle = expmap2axang(expmap)
    return axang2quat(axis, angle)

@torch.jit.script
def rotatepoint(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_r = q[...,3:4]
    q_xyz = q[...,:3]
    t = 2*torch.linalg.cross(q_xyz, v)
    return v + q_r * t + torch.linalg.cross(q_xyz, t)

@torch.jit.script
def quatmultiply(q0: torch.Tensor, q1: torch.Tensor):
    x0, y0, z0, w0 = torch.unbind(q0, -1)
    x1, y1, z1, w1 = torch.unbind(q1, -1)
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    return quatnormalize(torch.stack((x, y, z, w), -1))

@torch.jit.script
def quatconj(q: torch.Tensor):
    return torch.cat((-q[...,:3], q[...,-1:]), dim=-1)

@torch.jit.script
def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor):
    m = (fp[...,1:] - fp[...,:-1]) / (xp[...,1:] - xp[...,:-1])
    m = m.nan_to_num(nan=0., posinf=0., neginf=0.)
    b = fp[..., :-1] - m*xp[..., :-1]
    indicies = torch.sum(x[..., :, None] >= xp[..., None, :], -1) - 1
    idx = torch.clamp(indicies, 0, m.shape[-1] - 1)
    line_idx = torch.arange(len(indicies), device=indicies.device).view(-1, 1)
    res = m[line_idx, idx].mul(x) + b[line_idx, idx]
    res[indicies >= m.shape[-1]] = fp[-1]
    res[indicies < 0] = fp[0]
    return res

from collections import namedtuple
IGSO3 = namedtuple("IGSO3", "sigma omega pdf cdf score")
import sys
def igso3(*, sigmas, num_omega, series=20000):
    omega = torch.linspace(0, np.pi, num_omega+1, dtype=torch.float64)[1:]
    angle_density_unif = (1-torch.cos(omega))/np.pi
    pdf, cdf, score = [], [], []
    sigma = []
    prompt = "\r {:"+str(len(str(len(sigmas))))+"d}}/{}".format(len(sigmas))
    sys.stdout.write(prompt.format(0))
    for i, s in enumerate(sigmas):
        sys.stdout.write(prompt.format(i))
        o = omega.clone().detach().requires_grad_(True)
        t = s**2
        l = torch.arange(series)[None]
        f = ((2*l + 1) * torch.exp(-l*(l+1)*t/2) *
             torch.sin(o[:, None]*(l+1/2)) / torch.sin(o[:, None]/2)).sum(dim=-1)
        p = f*angle_density_unif
        c = p.cumsum(0) / num_omega * np.pi
        if i == 0 and c[-1].item() < 1-1e-6:
            raise ValueError("Discrete IGSO3 cannot converge. Please increase (1) the number of discrete omega, (2) the number of series or (3) the value of sigma.")
        dlp = torch.autograd.grad(torch.log(p).sum(), o)[0]
        pdf.append(p.detach().cpu().numpy())
        cdf.append(c.detach().cpu().numpy())
        score.append(dlp.detach().cpu().numpy())
        sigma.append(s)
    print()
    return IGSO3(sigma=sigma, omega=omega.detach().cpu().numpy(),
        pdf=pdf, cdf=cdf, score=score
    )
