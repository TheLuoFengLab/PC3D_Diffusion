import torch
torch.set_float32_matmul_precision('high')
import numpy as np
from math_utils import axang2quat, rotatepoint, expmap2quat, quatconj, quatdiff, quat2axang, quatmultiply, quat2expmap, wrap2pi, interp, igso3
import math_utils

import math
class Model(torch.nn.Module):
    class Estimator(torch.nn.Module):
        def __init__(self, in_dim=6, cond_dim=3, embed_dim=512, out_dim=5, max_t=500, dropout=0, head=4, layers=8):
            super().__init__()
            t_dim = 2
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(in_dim, embed_dim*4),
                torch.nn.SiLU(),
                torch.nn.Linear(embed_dim*4, embed_dim)
            )
            self.embed_c = torch.nn.Sequential(
                torch.nn.Linear(cond_dim+t_dim, embed_dim*4),
                torch.nn.SiLU(),
                torch.nn.Linear(embed_dim*4, embed_dim*4),
                torch.nn.SiLU(),
                torch.nn.Linear(embed_dim*4, embed_dim)
            )
            self.trans = torch.nn.TransformerDecoder(
                torch.nn.TransformerDecoderLayer(embed_dim, head, embed_dim*4, dropout=dropout, batch_first=True,
                norm_first=True, activation=torch.nn.functional.relu), layers)
            self.decode = torch.nn.Sequential(
                torch.nn.Linear(embed_dim*2, embed_dim*4),
                torch.nn.SiLU(),
                torch.nn.Linear(embed_dim*4, embed_dim*4),
                torch.nn.SiLU(),
                torch.nn.Linear(embed_dim*4, out_dim)
            )
            position = torch.arange(max_t).view(max_t, 1)
            div_term = torch.exp(torch.arange(0, t_dim, 2) * (-math.log(10000.0) / t_dim))
            pe = torch.zeros(max_t, t_dim)
            pe[..., 0::2] = torch.sin(position * div_term)
            pe[..., 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)
    
        def forward(self, p, d, c, t, mask=None):
            x = torch.cat((p, d), -1)
            latent = self.embed(x)
            if self.pe.size(-1) == 2:
                c = torch.cat((c, self.pe[t]), -1)
            else:
                t = self.pe[t].view(-1, 1)
                c = torch.cat((c, t, torch.sin(t), torch.cos(t)), -1)
            cond = self.embed_c(c).unsqueeze(1)
            out = self.trans(tgt=latent, memory=cond, tgt_key_padding_mask=mask)
            out = torch.cat((out, cond.repeat(1, out.size(1), 1)), -1)
            return self.decode(out)

    def __init__(self, cond_normalizer_offset, cond_normalizer_scale, T=500,
        dropout=0, head=16, layers=32,
        beta0=1e-4, beta1=0.02, min_sigma=0.05, max_sigma=5,
        fix_l=True, fix_r=True,
    ):
        super().__init__()
        cond_dim = 3
        if not fix_l: cond_dim += 1
        if not fix_r: cond_dim += 1 
        self.fix_l = fix_l
        self.fix_r = fix_r
        self.estimate_eps = self.Estimator(out_dim=5, cond_dim=3, max_t=T, dropout=dropout, head=head, layers=layers)

        self.register_parameter("loss_logsigma", torch.nn.Parameter(torch.ones(2)))

        self.register_buffer("cond_normalizer_offset", torch.tensor(cond_normalizer_offset, dtype=torch.float32))
        self.register_buffer("cond_normalizer_scale", torch.tensor(cond_normalizer_scale, dtype=torch.float32))

        beta = torch.linspace(beta0**0.5, beta1**0.5, T) ** 2
        alpha = 1 - beta
        alpha_ = torch.cumprod(alpha, 0)

        self.register_buffer("sqrt_one_minus_alpha_", torch.sqrt(1 - alpha_).view(-1, 1, 1))
        self.register_buffer("sqrt_alpha_", torch.sqrt(alpha_).view(-1, 1, 1))
        self.register_buffer("beta",  beta.view(-1, 1))
        self.register_buffer("one_by_sqrt_alpha", 1/torch.sqrt(alpha).view(-1, 1, 1))
        self.register_buffer("beta_by_sqrt_one_minus_alpha_", (beta/self.sqrt_one_minus_alpha_.view(-1)).view(-1, 1, 1))
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_", alpha_)
        self.register_buffer("T", torch.tensor(T))

        import os, pickle
        so3_cache = "igso3_{:.2e}_{:.2e}_{:d}.pkl".format(min_sigma, max_sigma, T)
        so3_min_b = 1.
        so3_max_b= (max_sigma-min_sigma)*2 - so3_min_b
        t_ = np.linspace(0, 1, T)
        if os.path.exists(so3_cache):
            with open(so3_cache, "rb") as f:
                so3 = pickle.load(f)
        else:
            print("Generating IGSO3 cache... This may take a while.")
            sigma = min_sigma + t_*so3_min_b + 0.5*(t_**2) * (so3_max_b - so3_min_b)
            sigma = np.clip(sigma, min_sigma, max_sigma)
            so3 = igso3(sigmas=sigma, num_omega=5000, series=50000)
            with open(so3_cache, "wb") as f:
                pickle.dump(so3, f)
            print("IGSO cache dumped to {}".format(f))
        so3.omega[:] = so3.omega*0.5 # from [0, pi] to [0, pi/2]
        so3_g2 = 2*np.multiply(so3.sigma, so3_min_b + t_ * (so3_max_b - so3_min_b))
        self.register_buffer("so3_omega", torch.as_tensor(so3.omega, dtype=torch.float32))
        self.register_buffer("so3_cdf", torch.as_tensor(np.array(so3.cdf), dtype=torch.float32))
        self.register_buffer("so3_score", torch.as_tensor(np.array(so3.score), dtype=torch.float32))
        self.register_buffer("so3_sigma", torch.as_tensor(so3.sigma, dtype=torch.float32))
        self.register_buffer("so3_g2", torch.as_tensor(so3_g2, dtype=torch.float32))

    def normalize_d(self, d):
        sign = torch.sign(d)
        sign = sign[..., 0] + (sign[..., 0]==0)*(sign[..., 1]+(sign[..., 1]==0)*sign[..., 2])
        return d * sign.unsqueeze_(-1)

    def normalize_p(self, p):
        return p*0.02 - 1
    
    def normalize_l(self, l_half):
        return l_half*0.04
    
    def normalize_r(self, d):
        return d*0.1

    def unnormalize_p(self, p):
        return (p+1) * 50
    
    def normalize_c(self, c):
        return (c+self.cond_normalizer_offset) * self.cond_normalizer_scale

    def forward(self, x_0, seq_len, c):
        p = self.normalize_p(x_0[..., :3])
        d = self.normalize_d(x_0[..., 3:6])
        c = self.normalize_c(c)
        if not self.fix_l and not self.fix_r:
            l = self.normalize_l(x_0[..., 0, 6:7])
            r = self.normalize_r(x_0[..., 0, 7:8])
            c = torch.cat((c, l, r), -1)
        elif not self.fix_l:
            l = self.normalize_l(x_0[..., 0, 6:7])
            c = torch.cat((c, l), -1)
        elif not self.fix_r:
            r = self.normalize_r(x_0[..., 0, 7:8])
            c = torch.cat((c, r), -1)

        L = p.size(1)
        mask = torch.arange(L, device=seq_len.device).view(1, L).repeat(x_0.size(0), 1) >= seq_len.unsqueeze(-1)

        t, p_t, d_t, e_p, e_r = self.diffuse(p, d, normalize_p=False, normalize_d=False)
        e = self.estimate_eps(p_t, d_t, c, t, mask)
        p_, d_, e_p_, e_r_ = self.pr_from_est(e, p_t, d_t, t)

        loss_p = (e_p_ - e_p).square().sum(-1)
        loss_r = (e_r_[..., :2] - e_r[..., :2]/self.so3_sigma[t].view(-1, 1, 1)).square().sum(-1)

        with torch.no_grad():
            err_p = (p_ - p).square().sum(-1)
            err_r = torch.rad2deg(torch.acos((torch.nn.functional.normalize(d_, dim=-1)*d).sum(-1).abs().clip(0, 1)))
            err_r[mask] = 0
            err_p[mask] = 0
            n = mask.numel()-mask.sum()
            err_r = err_r.sum()/n
            err_p = err_p.sum()/n
        loss_r[mask] = 0
        loss_p[mask] = 0

        return loss_r, loss_p, err_r, err_p, mask

    @torch.no_grad()
    def diffuse(self, p, d, normalize_p=True, normalize_d=True, t=None):
        N = p.size(0)
        L = p.size(1)
        if normalize_p:
            p = self.normalize_p(p)
        if normalize_d:
            d = self.normalize_d(d)
        if t is None:
            t = torch.randint(0, self.T, (N, ), device=p.device)
        else:
            t = t.to(p.device)
        
        e_p = torch.randn_like(p)
        p_t = self.sqrt_alpha_[t] * p + self.sqrt_one_minus_alpha_[t] * e_p
        z = torch.rand(N, L, device=p.device)
        
        e_ang = interp(z, self.so3_cdf[t], self.so3_omega).unsqueeze_(-1)
        e_ax = torch.randn_like(d)
        # R0 = Rt @ Re
        e_ax[..., 2] = 0
        norm = torch.linalg.norm(e_ax, ord=2, axis=-1, keepdims=True)
        e_ax /= norm
        e_q = axang2quat(e_ax, e_ang)
        z_ref = torch.zeros_like(e_ax)
        z_ref[..., 2] = 1
        q = quatdiff(z_ref, d)
        d_t = rotatepoint(quatmultiply(q, quatconj(e_q)), z_ref) # R0 @ Re.T = Rt
        d_t = torch.where(norm < 1e-12, d, d_t)
        # the rotation angle (ignore rotation around the axis itself) can be restored by
        q_t = quatdiff(z_ref, d_t)
        r_0t = rotatepoint(quatconj(q_t), d)
        e_q = quatdiff(z_ref, r_0t)
        e_r = quat2expmap(e_q)
        return t, p_t, d_t, e_p, e_r

    def pr_from_est(self, e, p_t, d_t, t):
        e_p_ = e[..., :3]
        sqrt_alpha_ = self.sqrt_alpha_[t]
        sqrt_one_minus_alpha_ = self.sqrt_one_minus_alpha_[t]
        p_ = (p_t - sqrt_one_minus_alpha_ * e_p_)/sqrt_alpha_

        e_r_ = e[..., 3:5]
        e_q_ = expmap2quat(torch.cat((e_r_*self.so3_sigma[t].view(-1, 1, 1), torch.zeros_like(e_r_[..., :1])), -1))
        z_ref = torch.zeros_like(p_)
        z_ref[..., 2] = 1
        q_t = quatdiff(z_ref, d_t)
        d_ = rotatepoint(quatmultiply(q_t, e_q_), z_ref)

        d_ = self.normalize_d(d_)
        return p_, d_, e_p_, e_r_

    def loss(self, loss_r, loss_p, err_r, err_p, mask):
        n = (mask.numel()-mask.sum()).clip(min=1)
        loss_r = loss_r.sum()/n
        loss_p = loss_p.sum()/n

        w = torch.exp(self.loss_logsigma).square()
        loss = loss_r/w[0] + loss_p/w[1] + 2*self.loss_logsigma.sum()
        res = dict(
            loss_r = loss_r,
            loss_p = loss_p,
            err_r = err_r,
            err_p = err_p,
            loss = loss
        )
        return res
    
    def constraint(self, p_, d_, l_half, dia):
        if l_half is None: l_half = 25
        if dia is None: dia = 10
        l_half = l_half/50
        diameter = dia/50
        gap = diameter+0.1/50
        r = diameter*0.5

        d_ = d_ / torch.linalg.norm(d_, ord=2, axis=-1, keepdims=True)
        d2_ = d_.square()
        dist2x = r*(d2_[:,:,1]+d2_[:,:,2]).sqrt()
        dist2y = r*(d2_[:,:,2]+d2_[:,:,0]).sqrt()
        dist2z = r*(d2_[:,:,0]+d2_[:,:,1]).sqrt()
        dist2ax = torch.stack((dist2x, dist2y, dist2z), -1)
        t1 = (dist2ax-1 - p_)/d_
        t2 = (1-dist2ax - p_)/d_
        t11 = torch.maximum(t1, t2).min(-1).values.clip(max=l_half).unsqueeze(-1)
        t22 = torch.minimum(t1, t2).max(-1).values.clip(min=-l_half).unsqueeze(-1)
        length = t11 - t22

        p_ = p_ + 0.5*(t22+t11)*d_
        c0 = p_ + 0.5*(t22-t11)*d_

        length2 = length.square()
        r = c0[:, None, :, :] - c0[:, :, None, :]
        u = (length*d_).unsqueeze(2)
        v = (length*d_).unsqueeze(1)
        ru = (r*u).sum(-1)
        rv = (r*v).sum(-1)
        uu = length2
        uv = (u*v).sum(-1)
        vv = length2.view(-1, 1, length2.size(1))
        det = uu*vv - uv*uv
        invalid = det < 1e-6
        valid = ~invalid
        det[invalid] = 1

        s = (ru/uu)*invalid + ((((ru*uv-rv*uu)/det).clip(0, 1) * uv + ru)/uu) * valid
        s = s.clip(0, 1)
        t = ((s*uv -rv)/vv) * invalid + ((((ru*vv-rv*uv)/det).clip(0, 1) * uv - rv)/vv) * valid
        t = t.clip(0, 1)

        p1x = c0[:, :, None, :] + s.unsqueeze(-1)*u
        p2x = c0[:, None, :, :] + t.unsqueeze(-1)*v
        dist2 = (p1x-p2x).square().sum(-1)
        
        loss_dist = 1 - dist2 / gap**2
        dia = torch.arange(loss_dist.size(1))
        m = loss_dist < 0
        m[:,dia,dia] = True

        return loss_dist, m
        

    def sample(self, c, p=None, d=None, T=None, return_traj=False):
        N = c.size(0)
        c = self.normalize_c(c[..., :3])
        t = torch.full_like(c[..., :1], 1)
        if p is None:
            p = torch.randn((N, 30, 3))
        if d is None:
            d = torch.nn.functional.normalize(torch.randn((N, 30, 3)), axis=-1)
        if T is None:
            T = self.T.item()
        if hasattr(T, "__len__"):
            T = list(reversed(sorted(T)))
        else:
            T = list(reversed(range(T)))
        l_half = c[..., 3]
        dia = c[..., 4]
        if not self.fix_l and not self.fix_r:
            c = torch.cat((c[...,:3], self.normalize_l(l_half), self.normalize_r(dia)), -1)
        elif not self.fix_l:
            c = torch.cat((c[...,:3], self.normalize_l(l_half)), -1)
        elif not self.fix_r:
            c = torch.cat((c[...,:3], self.normalize_r(dia)), -1)
            
        if 0 not in T:
            T.insert(0, 0)

        z_ref = torch.zeros_like(d)
        z_ref[..., 2] = 1

        traj = []
        for i, t in enumerate(T):
            traj.append(torch.cat((self.unnormalize_p(p), d), -1).detach().cpu().numpy())
            with torch.no_grad():
                e = self.estimate_eps(p, d, c, [t])
                p_0, d_0, *_ = self.pr_from_est(e, p, d, t)
            
            if i > len(T) - 3:
                for _ in range(1000):
                    x = torch.cat((p_0, d_0), -1).requires_grad_()
                    p_, d_ = x[..., :3], x[..., 3:]
                    loss_distx, m = self.constraint(p_, d_, l_half, dia)
                    loss_distx[m] = 0
                    loss_dist = loss_distx.sum()
                    if loss_dist.item() == 0: break

                    g = torch.autograd.grad(loss_dist, x)[0]
                    x = x - 0.001*g
                    x = x.detach()
                    p_0, d_0 = x[:,:,:3], x[:,:,3:6]
                    d_0 = torch.nn.functional.normalize(d_0, dim=-1)

            with torch.no_grad():
                if i == len(T)-1:
                    p = p_0
                else:
                    t_1 = T[i+1]
                    # it must be t-1 in order to make it work directly using formula from DDPM
                    mu = self.alpha_[t_1]**0.5 * self.beta[t] * p_0 + self.alpha[t]**0.5 * (1-self.alpha_[t_1]) * p
                    mu = mu / (1-self.alpha_[t])
                    sigma = torch.sqrt((1 - self.alpha_[t_1]) / (1 - self.alpha_[t]) * self.beta[t])
                    p = mu + sigma * torch.randn_like(p)

                if i == len(T)-1:
                    d = d_0
                else:
                    q_t = quatdiff(z_ref, d)
                    e_q = quatdiff(z_ref, rotatepoint(quatconj(q_t), d_0))

                    ax, ang = quat2axang(e_q)
                    ang = wrap2pi(ang)
                    ax[ang.squeeze(-1) < 0] *= -1
                    ang = ang.abs()
                    m = ang > self.so3_omega[-1]
                    ang[m] = np.pi - ang[m]
                    ax[m.squeeze(-1)] *= -1

                    s = interp(ang.squeeze(-1), self.so3_omega.unsqueeze(0), self.so3_score[t]).view(*ang.shape)
                    dt = (t-t_1)/self.T
                    g2 = self.so3_g2[t]
                    da = g2*dt*s
                    de = -ax*da + (g2*dt)**0.5 * torch.randn_like(ax)
                    d = rotatepoint(quatmultiply(q_t, expmap2quat(de)), z_ref)
            
        x_0 = torch.cat((self.unnormalize_p(p), d), -1).detach().cpu().numpy()
        traj.append(x_0)
        if return_traj: return traj
        return x_0
            
