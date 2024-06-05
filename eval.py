import os
import torch
import numpy as np
from model import Model
from vis_utils import auto_shrink, set_axes_equal
from torch_utils import seed
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import plot_cylinder

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--ckpt", type=str)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)


parser.add_argument("--n", type=int)
parser.add_argument("--length", type=float)
parser.add_argument("--diameter", type=float)
parser.add_argument("--a0", type=float)
parser.add_argument("--a1", type=float)
parser.add_argument("--a2", type=float)

if __name__ == "__main__":
    settings = parser.parse_args()
    from model import Model
    if settings.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = settings.device
    if os.path.isfile(settings.ckpt):
        ckpt = settings.ckpt
    elif os.path.exists(os.path.joint(settings.ckpt, "ckpt-best")):
        ckpt = os.path.joint(settings.ckpt, "ckpt-best")
    else:
        ckpt = os.path.joint(settings.ckpt, "ckpt")

    seed(settings.seed)
        
    model = Model([0.,0.,0.], [0.,0.,0.])
    model.to(device)
    state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict["model"])

    c = [settings.a0, settings.a1, settings.a2]
    if not model.fix_l: c.append(settings.length/2)
    if not model.fix_r: c.append(settings.diameter)
    c = torch.tensor([c], device=device)    # 1 x (3 or 4 or 5)

    p_t = torch.randn((1, settings.n, 3)).to(device) # 1 x n x 3
    u_t = torch.randn((1, settings.n, 3)).to(device) # 1 x n x 3
    u_t[..., 2] = 0
    u_t /= torch.linalg.norm(u_t, ord=2, dim=-1, keepdim=True)
    omega_t = torch.rand((1, settings.n, 1)).to(device) * (np.pi/2)
    d_t = omega_t*u_t

    x = model.sample(c, p_t, d_t)[0]
    ld = np.zeros((x.shape[0], 2))
    ld[:, 0] = settings.length/2
    ld[:, 1] = settings.diameter
    x = np.concatenate((x, ld), -1)
    fibers = auto_shrink(x)

    cm = plt.get_cmap("tab20")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot([0, 0], [100, 100], [0, 100], lw=0.75, c="#666", zorder=0)
    ax.plot([0, 100], [100, 100], [0, 0], lw=0.75, c="#666", zorder=0)
    ax.plot([100, 100], [100, 100], [0, 100], lw=0.75, c="#666", zorder=10000)
    ax.plot([0, 100], [100, 100], [100, 100], lw=0.75, c="#666", zorder=10000)
    ax.plot([0, 0], [0, 0], [0, 100], lw=0.75, c="#666", zorder=10000)
    ax.plot([0, 100], [0, 0], [0, 0], lw=0.75, c="#666", zorder=10000)
    ax.plot([100, 100], [0, 0], [0, 100], lw=0.75, c="#666", zorder=10000)
    ax.plot([0, 100], [0, 0], [100, 100], lw=0.75, c="#666", zorder=10000)
    ax.plot([0, 0], [0, 100], [100, 100], lw=0.75, c="#666", zorder=10000)
    ax.plot([0, 0], [0, 100], [0, 0], lw=0.75, c="#666", zorder=0)
    ax.plot([100, 100], [0, 100], [0, 0], lw=0.75, c="#666", zorder=10000)
    ax.plot([100, 100], [0, 100], [100, 100], lw=0.75, c="#666", zorder=10000)
    vf = 0
    for fid, (x, y, z, dx, dy, dz, ext, d) in enumerate(auto_shrink(fibers)):
        A2B = np.array([
            [0, 0, dx, x],
            [0, 0, dy, y],
            [0, 0, dz, z],
            [0, 0, 0, 1],
        ])
        c = cm(fid%len(cm.colors))
        plot_cylinder(ax=ax, length=ext*2, radius=d*0.5, thickness=d*0.499, A2B=A2B, color=c, alpha=1, wireframe=False)

    ax.view_init(elev=30, azim=-40)
    ax.set_xlim3d(0, 100)
    ax.set_ylim3d(0, 100)
    ax.set_zlim3d(0, 100)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    set_axes_equal(ax)

    plt.show()
    
