
import math, random
import numpy as np

PI = math.pi
PI2 = math.pi*2

X = 100 # x
Y = 100 # y
Z = 100 # z

L = 50  # length
D = 10  # diameter
N_FIBER = 10, 50

FIBER_GAP = 0.1  # minimal distance between two fibers excluding the radius of fibers
BOUND_GAP = 0.01 # minimal distance between a fiber to the boundary

def generate_fibers(N):
    proposal = []
    while len(proposal) < N:
        d = random.uniform(D[0], D[1]) if hasattr(D, "__len__") else D
        l = random.uniform(L[0], L[1]) if hasattr(L, "__len__") else L

        l_half = l*0.5
        r = d*0.5

        margin = r*(2**0.5)+BOUND_GAP
        x = margin+(X-margin*2)*random.uniform(0, 1)
        y = margin+(Y-margin*2)*random.uniform(0, 1)
        z = margin+(Z-margin*2)*random.uniform(0, 1)

        theta = PI2*random.uniform(0, 1)
        phi = math.acos(1-2*random.uniform(0, 1))

        s_phi = math.sin(phi)
        s_theta, c_theta = math.sin(theta), math.cos(theta)
        dx = s_phi * c_theta
        dy = s_phi * s_theta
        dz = math.cos(phi)
        fiber = (x, y, z, dx, dy, dz, l_half, d)
        if not collision_check(fiber, proposal): continue
        proposal.append(fiber)
    fibers = auto_shrink(proposal)
    return fibers

def auto_shrink(fibers, bound_x=X, bound_y=Y, bound_z=Z, bound_gap=BOUND_GAP):
    res = []
    for kkk, fiber in enumerate(fibers):
        x, y, z, dx, dy, dz, l_half, d = fiber[:8]
        assert x > 0 and x < bound_x
        assert y > 0 and y < bound_y
        assert z > 0 and z < bound_z

        r = d*0.5
        dist2x = r*(dz*dz+dy*dy)**0.5
        dist2y = r*(dx*dx+dz*dz)**0.5
        dist2z = r*(dx*dx+dy*dy)**0.5
        tx1 = (dist2x+bound_gap - x)/dx if dx else float("inf")
        ty1 = (dist2y+bound_gap - y)/dy if dy else float("inf")
        tz1 = (dist2z+bound_gap - z)/dz if dz else float("inf")
        tx2 = (bound_x-bound_gap - dist2x - x)/dx if dx else -float("inf")
        ty2 = (bound_y-bound_gap - dist2y - y)/dy if dy else -float("inf")
        tz2 = (bound_z-bound_gap - dist2z - z)/dz if dz else -float("inf")
        t  = min( l_half, max(tx1, tx2), max(ty1, ty2), max(tz1, tz2))
        t_ = max(-l_half, min(tx1, tx2), min(ty1, ty2), min(tz1, tz2))
        l = t-t_
        l_half = l*0.5
        c0 = [x+t*dx, y+t*dy, z+t*dz]
        c1 = [x+t_*dx, y+t_*dy, z+t_*dz]
        x, y, z = (c0[0]+c1[0])/2, (c0[1]+c1[1])/2, (c0[2]+c1[2])/2
        res.append((x, y, z, dx, dy, dz, l_half, d))
    return res
 
def collision_check(fiber, fibers, with_gap=FIBER_GAP):
    x, y, z, dx, dy, dz, l_half, d = fiber
    a0 = np.subtract((x,y,z), np.multiply(l_half, (dx, dy, dz)))
    u = np.multiply(2*l_half, (dx, dy, dz))
    for x1, y1, z1, dx1, dy1, dz1, l1_half, d1 in fibers:
        b0 = np.subtract((x1,y1,z1), np.multiply(l1_half, (dx1, dy1, dz1)))
        r = b0 - a0
        v = np.multiply(2*l1_half, (dx1, dy1, dz1))
        ru = (r*u).sum(-1)
        rv = (r*v).sum(-1)
        uu = (l_half*2)**2
        uv = (u*v).sum(-1)
        vv = (l1_half*2)**2
        det = uu*vv - uv*uv
        if det:
            s0 = np.clip((ru*vv-rv*uv)/det, 0, 1)
            t0 = np.clip((ru*uv-rv*uu)/det, 0, 1)
            s = np.clip((t0*uv + ru)/uu, 0, 1)
            t = np.clip((s0*uv - rv)/vv, 0, 1)
        else:
            s = np.clip(ru/uu, 0, 1)
            t = np.clip((s*uv - rv)/vv, 0, 1)
        p1x = a0+s*u
        p2x = b0+t*v
        dist2 = np.sum(np.square(p2x-p1x), -1)
        if with_gap:
            if dist2 < (max(d1,d)+with_gap)**2: 
                return False
        else:
            if dist2 < 0.25 * (d1+d)**2: 
                return False
    return True

def discretize_fibers(fibers):
    fiber_pts = []
    for fiber in fibers:
        x, y, z, l_half = fiber[0], fiber[1], fiber[2], fiber[6]
        dx, dy, dz, d = fiber[3], fiber[4], fiber[5], fiber[7]
        
        s_phi = (1-dz*dz)**0.5
        c_theta = dx/s_phi
        s_theta = dy/s_phi

        r = d*0.5
        r_by_sqrt2 = r/2**0.5

        c0 = [x-l_half*dx, y-l_half*dy, z-l_half*dz]
        c1 = [x+l_half*dx, y+l_half*dy, z+l_half*dz]

        n_x = -s_theta
        n_y = c_theta
        n_z = 0

        tan_x = -dz*n_y
        tan_y = dz*n_x
        tan_z = dx*n_y-dy*n_x

        ext_n_x = n_x*r
        ext_n_y = n_y*r
        ext_n_z = n_z*r

        ext_t_x = tan_x*r
        ext_t_y = tan_y*r
        ext_t_z = tan_z*r

        ext_nt_x = (n_x+tan_x)*r_by_sqrt2
        ext_nt_y = (n_y+tan_y)*r_by_sqrt2
        ext_nt_z = (n_z+tan_z)*r_by_sqrt2

        ext_nnt_x = (n_x-tan_x)*r_by_sqrt2
        ext_nnt_y = (n_y-tan_y)*r_by_sqrt2
        ext_nnt_z = (n_z-tan_z)*r_by_sqrt2

        pts = [
            [ ext_t_x,    ext_t_y,    ext_t_z],
            [ ext_nt_x,   ext_nt_y,   ext_nt_z],
            [ ext_n_x,    ext_n_y,    ext_n_z],
            [ ext_nnt_x,  ext_nnt_y,  ext_nnt_z],
            [-ext_t_x,   -ext_t_y,   -ext_t_z],
            [-ext_nt_x,  -ext_nt_y,  -ext_nt_z],
            [-ext_n_x,   -ext_n_y,   -ext_n_z],
            [-ext_nnt_x, -ext_nnt_y, -ext_nnt_z]
        ]
        p = []
        for c in [c0, c1]:
            for x, y, z in pts:
                x_, y_, z_ = c[0]+x, c[1]+y, c[2]+z
                p.append([x_, y_, z_])
        fiber_pts.append(p)
    return fiber_pts


def restore_shrunk_fibers(fibers, l_half, bound_x=X, bound_y=Y, bound_z=Z, bound_gap=BOUND_GAP):
    fibers = np.array(fibers)
    x = fibers[..., 0]
    y = fibers[..., 1]
    z = fibers[..., 2]
    dx = fibers[..., 3]
    dy = fibers[..., 4]
    dz = fibers[..., 5]
    l_h = fibers[..., 6]
    d = fibers[...,  7]
    
    r = d*0.5
    dist2x = r*(dz*dz+dy*dy)**0.5
    dist2y = r*(dx*dx+dz*dz)**0.5
    dist2z = r*(dx*dx+dy*dy)**0.5
    tx1 = (dist2x+bound_gap - x)/dx # x = bound_gap
    ty1 = (dist2y+bound_gap - y)/dy # y = bound_gap
    tz1 = (dist2z+bound_gap - z)/dz # z = bound_gap
    tx2 = (bound_x-bound_gap - dist2x - x)/dx # x = X-bound_gap
    ty2 = (bound_y-bound_gap - dist2y - y)/dy # y = Y-bound_gap
    tz2 = (bound_z-bound_gap - dist2z - z)/dz # z = Z-bound_gap

    dx1 = np.abs(np.abs(tx1)-l_h)
    dy1 = np.abs(np.abs(ty1)-l_h)
    dz1 = np.abs(np.abs(tz1)-l_h)
    dx2 = np.abs(np.abs(tx2)-l_h)
    dy2 = np.abs(np.abs(ty2)-l_h)
    dz2 = np.abs(np.abs(tz2)-l_h)

    l_m = l_h < l_half
    mx1 = np.logical_and(np.isclose(dx1, 0), l_m)
    my1 = np.logical_and(np.isclose(dy1, 0), l_m)
    mz1 = np.logical_and(np.isclose(dz1, 0), l_m)
    mx2 = np.logical_and(np.isclose(dx2, 0), l_m)
    my2 = np.logical_and(np.isclose(dy2, 0), l_m)
    mz2 = np.logical_and(np.isclose(dz2, 0), l_m)

    m = np.int32(mx1)+np.int32(my1)+np.int32(mz1)+np.int32(mx2)+np.int32(my2)+np.int32(mz2)
    assert np.max(m) < 3
    not_m = m == 1
    mx1 = np.logical_and(mx1, not_m)
    my1 = np.logical_and(my1, not_m)
    mz1 = np.logical_and(mz1, not_m)
    mx2 = np.logical_and(mx2, not_m)
    my2 = np.logical_and(my2, not_m)
    mz2 = np.logical_and(mz2, not_m)

    
    m2 = np.logical_and(m == 0, l_m)
    d_min = np.minimum.reduce([dx1, dy1, dz1, dx2, dy2, dz2])
    
    mx1_ = np.logical_and(m2, dx1 == d_min)
    my1_ = np.logical_and(m2, dy1 == d_min)
    mz1_ = np.logical_and(m2, dz1 == d_min)
    mx2_ = np.logical_and(m2, dx2 == d_min)
    my2_ = np.logical_and(m2, dy2 == d_min)
    mz2_ = np.logical_and(m2, dz2 == d_min)
    m_ = np.int32(mx1_)+np.int32(my1_)+np.int32(mz1_)+np.int32(mx2_)+np.int32(my2_)+np.int32(mz2_)
    assert np.max(m_) < 2

    mx1 = np.logical_or(mx1, mx1_)
    my1 = np.logical_or(my1, my1_)
    mz1 = np.logical_or(mz1, mz1_)
    mx2 = np.logical_or(mx2, mx2_)
    my2 = np.logical_or(my2, my2_)
    mz2 = np.logical_or(mz2, mz2_)
        

    dp = l_half - l_h

    dpdx = dp*dx
    dpdy = dp*dy
    dpdz = dp*dz
    sub = np.logical_or.reduce((
        np.logical_and(mx1, dx > 0), np.logical_and(my1, dy > 0), np.logical_and(mz1, dz > 0),
        np.logical_and(mx2, dx < 0), np.logical_and(my2, dy < 0), np.logical_and(mz2, dz < 0)
    ))
    fibers[sub, 0] -= dpdx[sub]
    fibers[sub, 1] -= dpdy[sub]
    fibers[sub, 2] -= dpdz[sub]
    sub = np.logical_or.reduce((
        np.logical_and(mx1, dx < 0), np.logical_and(my1, dy < 0), np.logical_and(mz1, dz < 0),
        np.logical_and(mx2, dx > 0), np.logical_and(my2, dy > 0), np.logical_and(mz2, dz > 0)
    ))
    fibers[sub, 0] += dpdx[sub]
    fibers[sub, 1] += dpdy[sub]
    fibers[sub, 2] += dpdz[sub]
    fibers[..., 6] = l_half

    return fibers

import matplotlib.pyplot as plt

def set_axes_equal(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def draw_cylinder(ax, p0, p1, R):
    v = p1 - p0
    mag = np.linalg.norm(v)
    v = v / mag
    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    n1 = np.cross(v, not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)
    t = np.linspace(0, mag, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    t, theta = np.meshgrid(t, theta)
    X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    ax.plot_surface(X, Y, Z, alpha=0.6)
    ax.plot(*zip(p0, p1), color='red')

def plot_fibers(fibers):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if len(fibers[0]) > 8:
        for fiber in fibers:
            coord = np.array(fiber)
            x = coord[:,0]
            y = coord[:,1]
            z = coord[:,2]
            sc = ax.scatter(x, y, z)
            c = sc.get_facecolors()[0]
            n = len(x)//2
            for j in range(n):
                ax.plot([x[j], x[j+n]], [y[j], y[j+n]], [z[j], z[j+n]], c=c)
    else:
        for x, y, z, dx, dy, dz, l_half, d in fibers:
            p0 = np.array([x-dx*l_half, y-dy*l_half, z-dz*l_half])
            p1 = np.array([x+dx*l_half, y+dy*l_half, z+dz*l_half])
            draw_cylinder(ax, p0, p1, d*0.5)

    ax.plot([0, 0], [100, 100], [0, 100], c="gray", zorder=10000)
    ax.plot([0, 100], [100, 100], [0, 0], c="gray", zorder=10000)
    ax.plot([100, 100], [100, 100], [0, 100], c="gray", zorder=10000)
    ax.plot([0, 100], [100, 100], [100, 100], c="gray", zorder=10000)
    ax.plot([0, 0], [0, 0], [0, 100], c="gray", zorder=10000)
    ax.plot([0, 100], [0, 0], [0, 0], c="gray", zorder=10000)
    ax.plot([100, 100], [0, 0], [0, 100], c="gray", zorder=10000)
    ax.plot([0, 100], [0, 0], [100, 100], c="gray", zorder=10000)
    ax.plot([0, 0], [0, 100], [100, 100], c="gray", zorder=10000)
    ax.plot([0, 0], [0, 100], [0, 0], c="gray", zorder=10000)
    ax.plot([100, 100], [0, 100], [0, 0], c="gray", zorder=10000)
    ax.plot([100, 100], [0, 100], [100, 100], c="gray", zorder=10000)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax)
    plt.tight_layout()
    return fig, ax

