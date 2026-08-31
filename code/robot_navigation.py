"""
Reproduces the qualitative structure of Figure 4 (Section 4.2, Application
on Dynamic Robot Navigation) in the manuscript: a robot with planar position
p(t) tracks a moving target while avoiding seven stationary obstacles via a
repulsive potential field, subject to the map's box constraints (Eq. 21).

The objective here is NOT quadratic in p (it is a target-tracking quadratic
term plus a sum of Gaussian repulsive bumps), so this script drives the
*generic* LBFN engine in lbfn_model.py with the problem's actual analytic
gradient/Hessian/grad_h_t, rather than the closed-form M(t), g(t) machinery
used for the quadratic numerical example -- see lbfn_model.py's module
docstring for why the same engine supports both cases.

Run this file directly to regenerate a PNG figure into ./figures_out/.

Note: this is an independent, clean-room reproduction of the described
experiment; it is not guaranteed to match the originally published .eps
figure pixel-for-pixel (which was produced with a separate plotting
pipeline), but reproduces the same qualitative behavior (fast convergence
to the moving target, obstacle avoidance, and the reported potential-field
snapshots at t = 10, 20, 30 s).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: avoids blocking on a GUI event loop
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from lbfn_model import BarrierConstraints, lbfn_rhs, simulate

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures_out")

# ---------------------------------------------------------------------------
# Problem parameters (Section 4.2)
# ---------------------------------------------------------------------------
K_T = 0.3
ALPHA = 1000.0
T_END = 30.0

OBSTACLES = np.array([
    [20, 25], [60, 20], [70, 60], [50, 85], [30, 72], [38, 35], [90, 40],
], dtype=float)
K_OBS = np.array([0.8, 0.6, 1.0, 1.2, 1.2, 0.8, 0.9])

MAP_A = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
MAP_D = np.array([100.0, 100.0, 0.0, 0.0])


def p_tar(t):
    return np.array([50 + 40 * np.sin(0.25 * t), 50 + 25 * np.cos(0.25 * t)])


def p_tar_dot(t):
    return np.array([10.0 * np.cos(0.25 * t), -6.25 * np.sin(0.25 * t)])


def h_value(p, t):
    val = 0.5 * K_T * np.sum((p - p_tar(t)) ** 2)
    for pObs, k in zip(OBSTACLES, K_OBS):
        val += np.exp(-np.sum((p - pObs) ** 2) / k)
    return val


def grad_h(p, t):
    g = K_T * (p - p_tar(t))
    for pObs, k in zip(OBSTACLES, K_OBS):
        e = p - pObs
        expo = np.exp(-(e @ e) / k)
        g += -(2.0 / k) * expo * e
    return g


def hess_h(p, t):
    H = K_T * np.eye(2)
    for pObs, k in zip(OBSTACLES, K_OBS):
        e = p - pObs
        expo = np.exp(-(e @ e) / k)
        H += -(2.0 / k) * expo * np.eye(2) + (4.0 / k ** 2) * expo * np.outer(e, e)
    # Regularize: the repulsive terms can locally reduce curvature very
    # close to an obstacle; floor the eigenvalues so the Newton step in
    # lbfn_model.lbfn_rhs stays well-defined throughout the simulation.
    eigval, eigvec = np.linalg.eigh(H)
    eigval = np.maximum(eigval, 1e-3)
    return (eigvec * eigval) @ eigvec.T


def grad_h_t(p, t):
    return -K_T * p_tar_dot(t)  # obstacles are stationary


def A_of_t(t):
    return MAP_A


def Adot_of_t(t):
    return np.zeros_like(MAP_A)


def d_of_t(t):
    return MAP_D


def ddot_of_t(t):
    return np.zeros_like(MAP_D)


def run_simulation(p0, n_pts=6000):
    t_eval = np.linspace(0, T_END, n_pts)
    cons = BarrierConstraints(A_of_t, Adot_of_t, d_of_t, ddot_of_t,
                               r0=5.0, gamma_r=0.5, w0=1.0, gamma_w=2.0)
    traj = simulate(lbfn_rhs, p0, (0, T_END), t_eval, grad_h, hess_h, grad_h_t, cons, ALPHA)
    return t_eval, traj


def grad_norm_field(X, Y, t):
    G = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            G[i, j] = np.linalg.norm(grad_h(np.array([X[i, j], Y[i, j]]), t))
    return G


def save_snapshot(prefix, t_eval, traj, t_snap):
    """Save the three panels for one snapshot time as <prefix>_motion.eps,
    <prefix>_contour.eps, <prefix>_land.eps, matching the manuscript's
    original filenames (t10_*, t20_*, t30_*)."""
    idx = np.searchsorted(t_eval, t_snap)
    tgt = p_tar(t_snap)
    grid = np.linspace(0, 100, 140)
    X, Y = np.meshgrid(grid, grid)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = h_value(np.array([X[i, j], Y[i, j]]), t_snap)
    G = grad_norm_field(X, Y, t_snap)

    # --- motion trajectory panel ---
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    ax.plot(traj[:idx, 0], traj[:idx, 1], "m-", lw=1.5, label="Trajectory of robot")
    ax.plot(traj[0, 0], traj[0, 1], "k^", ms=8, label="Starting point")
    ax.scatter(OBSTACLES[:, 0], OBSTACLES[:, 1], c="orangered", s=200, alpha=0.6, label="Obstacle")
    ax.plot(tgt[0], tgt[1], "bo", ms=8, label="Dynamic target")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Trajectory at t={t_snap:.0f}s")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{prefix}_motion.eps", format="eps")
    plt.close(fig)

    # --- potential force field (gradient-norm) contour panel ---
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    cf = ax.contourf(X, Y, G, levels=30, cmap="YlOrRd")
    ax.plot(traj[:idx, 0], traj[:idx, 1], "m-", lw=1.2)
    ax.plot(tgt[0], tgt[1], "bo", ms=6)
    ax.scatter(OBSTACLES[:, 0], OBSTACLES[:, 1], c="cyan", s=20, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Potential force field at t={t_snap:.0f}s")
    fig.tight_layout()
    fig.savefig(f"{prefix}_contour.eps", format="eps")
    plt.close(fig)

    # --- potential landscape panel ---
    fig = plt.figure(figsize=(3.6, 3.3))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="copper", linewidth=0, antialiased=True)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel(r"$\hbar(p,t)$")
    ax.set_title(f"Potential landscape at t={t_snap:.0f}s")
    fig.tight_layout()
    fig.savefig(f"{prefix}_land.eps", format="eps")
    plt.close(fig)


def main():
    p0 = np.array([10.0, 10.0])
    t_eval, traj = run_simulation(p0)

    log_lines = []
    dist_to_target = np.linalg.norm(traj - np.array([p_tar(t) for t in t_eval]), axis=1)
    dist_to_obstacles = np.array([
        [np.linalg.norm(traj[k] - o) for o in OBSTACLES] for k in range(len(t_eval))
    ])
    min_obs_dist = dist_to_obstacles.min()
    idx_min = np.unravel_index(dist_to_obstacles.argmin(), dist_to_obstacles.shape)
    log_lines.append(f"Initial distance to target: {dist_to_target[0]:.3f} m")
    for tc in [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]:
        idx = np.searchsorted(t_eval, tc)
        log_lines.append(f"t={tc:>5}s: distance to moving target = {dist_to_target[idx]:.4f} m, "
                          f"min distance to any obstacle = {dist_to_obstacles[idx].min():.4f} m")
    log_lines.append(f"Closest robot-obstacle approach over the whole run: {min_obs_dist:.4f} m "
                      f"(obstacle #{idx_min[1]+1} at t={t_eval[idx_min[0]]:.2f}s)")
    within_bounds = np.all((traj >= 0) & (traj <= 100))
    log_lines.append(f"Robot stayed within the [0,100]x[0,100] map bounds for the entire run: {within_bounds}")
    for line in log_lines:
        print(line)
    with open("run_log_robot.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")

    for prefix, t_snap in zip(["t10", "t20", "t30"], [10.0, 20.0, 30.0]):
        save_snapshot(prefix, t_eval, traj, t_snap)
    print("Done. Panel .eps files and run_log_robot.txt written to the current directory.")


if __name__ == "__main__":
    main()
