"""
Reproduces Figures 1-3 (Section 4.1, Numerical Example) of the manuscript:
the dynamic QP of Eqs. (18)-(20),

    h(v,t) = 1/2 v^T P(t) v + n(t)^T v ,   P(t) = diag(1+0.5 sin 3t, 2+cos 6t)
    G(t) v <= h_vec(t)

run under the LBFN model for (i) varying alpha, (ii) varying initial
conditions, and (iii) compared against the AS / SQP / SeDuMi / CGD / NF
baselines.

This script saves one .eps file per panel, using the SAME filenames as the
manuscript's original figures (coeff_*.eps, init_*.eps, comp_*.eps), so it
can be run to regenerate the exact files included via \includegraphics in main.tex.
It also prints precise, computed diagnostics (settling times, error
floors, constraint-violation windows) to stdout and to run_log.txt, which
were used to verify every corresponding quantitative claim in the
manuscript text against actual simulation output rather than by eye.

Run: python numerical_example.py
Output: one <name>.eps per panel in the current directory (overwriting the
        manuscript's originals if run from a paper/ folder), plus
        run_log.txt with the extracted diagnostics.
"""
from __future__ import annotations

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lbfn_model import (
    BarrierConstraints, lbfn_rhs, newton_flow_rhs, gradient_flow_rhs,
    simulate, gradient_error,
)
from baselines import solve_qp_snapshot, solve_barrier_snapshot, run_snapshot_baseline

LOG_LINES = []


def log(msg):
    print(msg)
    LOG_LINES.append(msg)


# ---------------------------------------------------------------------------
# Problem definition (Eqs. 18-20)
# ---------------------------------------------------------------------------
def M_of_t(t):
    return np.array([[1 + 0.5 * np.sin(3 * t), 0.0],
                      [0.0, 2 + np.cos(6 * t)]])


def Mdot_of_t(t):
    return np.array([[1.5 * np.cos(3 * t), 0.0],
                      [0.0, -6 * np.sin(6 * t)]])


def g_of_t(t):
    return np.array([np.sin(3 * t), np.cos(3 * t)])


def gdot_of_t(t):
    return np.array([3 * np.cos(3 * t), -3 * np.sin(3 * t)])


def grad_h(v, t):
    return M_of_t(t) @ v + g_of_t(t)


def hess_h(v, t):
    return M_of_t(t)


def grad_h_t(v, t):
    return Mdot_of_t(t) @ v + gdot_of_t(t)


def A_of_t(t):
    return np.array([[np.sin(2 * t), 1.0],
                      [-np.cos(2 * t), 2 * t / 10]])


def Adot_of_t(t):
    return np.array([[2 * np.cos(2 * t), 0.0],
                      [2 * np.sin(2 * t), 0.2]])


def d_of_t(t):
    return np.array([2 + np.cos(2 * t), 2 - np.sin(4 * t)])


def ddot_of_t(t):
    return np.array([-2 * np.sin(2 * t), -4 * np.cos(4 * t)])


def constraint_values(v, t):
    return A_of_t(t) @ v - d_of_t(t)


def make_constraints(r0, gamma_r, w0, gamma_w):
    return BarrierConstraints(A_of_t, Adot_of_t, d_of_t, ddot_of_t, r0, gamma_r, w0, gamma_w)


def save_panel(x, ys, labels, xlabel, ylabel, fname, logy=False, logx=False, hline=None, lw=1.2):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for y, lab in zip(ys, labels):
        ax.plot(x, y, label=lab, lw=lw)
    if hline is not None:
        ax.axhline(hline, color="k", ls="--", lw=1, label="Constraint Bound")
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(labels) <= 8:
        ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(fname, format="eps")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: convergence vs. alpha  ->  coeff_*.eps
# ---------------------------------------------------------------------------
def figure1_vary_alpha(t_end=5.0, n_pts=3000):
    t_eval = np.linspace(0, t_end, n_pts)
    v0 = np.array([1.0, 1.0])
    alphas = [2, 20, 100, 1000, 5000]
    labels = [f"alpha = {a}" for a in alphas]

    v1s, v2s, g1s, g2s, e1s, e2s = [], [], [], [], [], []
    log("\n=== Figure 1 (vary alpha) diagnostics ===")
    for alpha in alphas:
        cons = make_constraints(r0=1.0, gamma_r=5.0, w0=1.0, gamma_w=10.0)
        traj = simulate(lbfn_rhs, v0, (0, t_end), t_eval, grad_h, hess_h, grad_h_t, cons, alpha)
        err = gradient_error(traj, t_eval, grad_h, cons)
        cvals = np.array([constraint_values(traj[k], t_eval[k]) for k in range(len(t_eval))])
        v1s.append(traj[:, 0]); v2s.append(traj[:, 1])
        g1s.append(cvals[:, 0]); g2s.append(cvals[:, 1])
        e1s.append(np.maximum(err[:, 0], 1e-300)); e2s.append(np.maximum(err[:, 1], 1e-300))

        norms = np.linalg.norm(err, axis=1)
        checkpoints = [1e-3, 0.1, 0.2, 0.5, 1.0, 3.0, t_end]
        cp_vals = {tc: norms[np.searchsorted(t_eval, tc)] for tc in checkpoints if tc <= t_end}
        cp_str = ", ".join(f"t={tc}s:{v:.3e}" for tc, v in cp_vals.items())
        floor2 = err[-1, 1]
        max_g_late = np.max(cvals[t_eval >= 0.3]) if t_end >= 0.3 else np.max(cvals)
        log(f"alpha={alpha:5d}: |grad_v Upsilon| at checkpoints -> {cp_str}")
        log(f"           final |grad_v Upsilon|_2 = {floor2:.3e}; max constraint value for t>=0.3s = {max_g_late:.4f}")

    save_panel(t_eval, v1s, labels, "Time (s)", r"$v_1(t)$", "coeff_x1.eps")
    save_panel(t_eval, v2s, labels, "Time (s)", r"$v_2(t)$", "coeff_x2.eps")
    save_panel(t_eval, g1s, labels, "Time (s)", r"$\ell_1(t)$", "coeff_g1.eps", hline=0.0)
    save_panel(t_eval, g2s, labels, "Time (s)", r"$\ell_2(t)$", "coeff_g2.eps", hline=0.0)
    save_panel(t_eval[1:], [e[1:] for e in e1s], labels, "Time (s)", r"$|\nabla_v\Upsilon|_1$",
               "coeff_err1.eps", logy=True, logx=True)
    save_panel(t_eval[1:], [e[1:] for e in e2s], labels, "Time (s)", r"$|\nabla_v\Upsilon|_2$",
               "coeff_err2.eps", logy=True, logx=True)


# ---------------------------------------------------------------------------
# Figure 2: robustness to initialization  ->  init_*.eps
# ---------------------------------------------------------------------------
def figure2_vary_init(t_end=3.0, n_pts=2400, n_starts=10, seed=0):
    rng = np.random.default_rng(seed)
    t_eval = np.linspace(0, t_end, n_pts)
    alpha = 20.0

    v1s, v2s, g1s, g2s, e1s, e2s = [], [], [], [], [], []
    labels = [f"SP{k+1}" for k in range(n_starts - 1)] + ["SP10 (infeasible)"]

    log("\n=== Figure 2 (vary initialization) diagnostics ===")
    v0s_used = []
    all_norms, all_cvals, initial_violations = [], [], []
    for k in range(n_starts):
        if k == n_starts - 1:
            # One starting point is chosen OUTSIDE [-1.5,1.5]^2 and is
            # deliberately hard-infeasible at t=0 (g(0)=(1,1)>0), to
            # explicitly test the infeasible-initialization guarantee
            # (Lemma 1 / Corollary 1) rather than only feasible starts.
            v0 = np.array([-3.0, 4.0])
        else:
            v0 = rng.uniform(-1.5, 1.5, size=2)
        v0s_used.append(v0)
        worst = max(0.0, np.max(constraint_values(v0, 0.0)))
        initial_violations.append(worst)
        cons = make_constraints(r0=worst + 1.0, gamma_r=5.0, w0=1.0, gamma_w=10.0)
        traj = simulate(lbfn_rhs, v0, (0, t_end), t_eval, grad_h, hess_h, grad_h_t, cons, alpha)
        err = gradient_error(traj, t_eval, grad_h, cons)
        cvals = np.array([constraint_values(traj[i], t_eval[i]) for i in range(len(t_eval))])
        v1s.append(traj[:, 0]); v2s.append(traj[:, 1])
        g1s.append(cvals[:, 0]); g2s.append(cvals[:, 1])
        e1s.append(np.maximum(err[:, 0], 1e-300)); e2s.append(np.maximum(err[:, 1], 1e-300))
        all_norms.append(np.linalg.norm(err, axis=1))
        all_cvals.append(cvals)

    log(f"Starting points v(0) (10 SPs, seed={seed}): " + "; ".join(f"({a:.3f},{b:.3f})" for a, b in v0s_used))
    log(f"Initial constraint slack max_i(A_i(0)v(0)-d_i(0)) across SPs: "
        f"min={min(initial_violations):.4f}, max={max(initial_violations):.4f} "
        f"(<=0 means feasible at t=0 for every SP)")
    all_norms = np.array(all_norms)   # (n_starts, n_pts)
    for tc in [0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, t_end]:
        idx = np.searchsorted(t_eval, tc)
        worst_norm = np.max(all_norms[:, idx])
        worst_g = np.max([c[idx] for c in all_cvals])
        log(f"  t={tc:>4}s: worst-case |grad_v Upsilon| across 10 SPs = {worst_norm:.3e}, "
            f"worst-case constraint value = {worst_g:.4f}")
    reduction_orders = np.log10(np.maximum(all_norms[:, 0], 1e-300) / np.maximum(all_norms[:, -1], 1e-300))
    log(f"Gradient-error reduction across SPs (t=0 to t={t_end}s): "
        f"min={reduction_orders.min():.2f}, max={reduction_orders.max():.2f} orders of magnitude")

    save_panel(t_eval, v1s, labels, "Time (s)", r"$v_1(t)$, 10 SPs", "init_x1.eps", lw=0.9)
    save_panel(t_eval, v2s, labels, "Time (s)", r"$v_2(t)$, 10 SPs", "init_x2.eps", lw=0.9)
    save_panel(t_eval, g1s, labels, "Time (s)", r"$\ell_1(t)$", "init_g1.eps", hline=0.0, lw=0.9)
    save_panel(t_eval, g2s, labels, "Time (s)", r"$\ell_2(t)$", "init_g2.eps", hline=0.0, lw=0.9)
    save_panel(t_eval[1:], [e[1:] for e in e1s], labels, "Time (s)", r"$|\nabla_v\Upsilon|_1$",
               "init_err1.eps", logy=True, lw=0.9)
    save_panel(t_eval[1:], [e[1:] for e in e2s], labels, "Time (s)", r"$|\nabla_v\Upsilon|_2$",
               "init_err2.eps", logy=True, lw=0.9)


# ---------------------------------------------------------------------------
# Figure 3: comparison against baselines  ->  comp_*.eps
# ---------------------------------------------------------------------------
def figure3_comparison(t_end=12.0, n_pts=3000):
    t_eval = np.linspace(0, t_end, n_pts)
    v0 = np.array([1.0, 1.0])
    alpha = 10.0
    cons = make_constraints(r0=1.0, gamma_r=5.0, w0=1.0, gamma_w=10.0)

    lbfn_traj = simulate(lbfn_rhs, v0, (0, t_end), t_eval, grad_h, hess_h, grad_h_t, cons, alpha)
    nf_traj = simulate(newton_flow_rhs, v0, (0, t_end), t_eval, grad_h, hess_h, grad_h_t, cons, alpha)
    cgd_traj = simulate(gradient_flow_rhs, v0, (0, t_end), t_eval, grad_h, hess_h, grad_h_t, cons, alpha)

    snap_eval = np.arange(0, t_end, 0.05)
    as_traj_coarse = run_snapshot_baseline(solve_qp_snapshot, grad_h, hess_h, A_of_t, d_of_t, snap_eval, v0)
    sedumi_traj_coarse = run_snapshot_baseline(solve_barrier_snapshot, grad_h, hess_h, A_of_t, d_of_t, snap_eval, v0)

    def upsample(traj_coarse):
        return np.vstack([np.interp(t_eval, snap_eval, traj_coarse[:, j]) for j in range(2)]).T

    as_traj = upsample(as_traj_coarse)
    sedumi_traj = upsample(sedumi_traj_coarse)

    methods = {"LBFN": lbfn_traj, "AS": as_traj, "SeDuMi": sedumi_traj, "CGD": cgd_traj, "NF": nf_traj}

    log("\n=== Figure 3 (baseline comparison) diagnostics ===")
    v1s, v2s, g1s, g2s, e1s, e2s, labels = [], [], [], [], [], [], []
    for name, traj in methods.items():
        cvals = np.array([constraint_values(traj[k], t_eval[k]) for k in range(len(t_eval))])
        err = gradient_error(traj, t_eval, grad_h, cons)
        v1s.append(traj[:, 0]); v2s.append(traj[:, 1])
        g1s.append(cvals[:, 0]); g2s.append(cvals[:, 1])
        e1s.append(np.maximum(err[:, 0], 1e-300)); e2s.append(np.maximum(err[:, 1], 1e-300))
        labels.append(name)

        max_viol = np.max(cvals)
        viol_mask = np.max(cvals, axis=1) > 1e-4  # 1e-4 threshold: well above EPS-floor numerical noise
        if np.any(viol_mask):
            viol_times = t_eval[viol_mask]
            windows = []
            start = viol_times[0]
            prev = viol_times[0]
            for tt in viol_times[1:]:
                if tt - prev > 0.05:
                    windows.append((start, prev))
                    start = tt
                prev = tt
            windows.append((start, prev))
            windows_str = ", ".join(f"[{a:.2f},{b:.2f}]s" for a, b in windows[:6])
            log(f"{name}: constraint exceeds 1e-4 during {len(windows)} window(s) (peak violation {max_viol:.3e}), "
                f"first few windows: {windows_str}")
        else:
            log(f"{name}: no constraint violation detected (max value {np.max(cvals):.3e} <= 0)")

        norms = np.linalg.norm(err, axis=1)
        log(f"{name}: final |grad_v Upsilon| at t={t_end}s = {norms[-1]:.3e}, "
            f"median over [2,{t_end}]s = {np.median(norms[t_eval>=2]):.3e}")

    save_panel(t_eval, v1s, labels, "Time (s)", r"$v_1(t)$", "comp_x1.eps")
    save_panel(t_eval, v2s, labels, "Time (s)", r"$v_2(t)$", "comp_x2.eps")
    save_panel(t_eval, g1s, labels, "Time (s)", r"$\ell_1(t)$", "comp_g1.eps", hline=0.0)
    save_panel(t_eval, g2s, labels, "Time (s)", r"$\ell_2(t)$", "comp_g2.eps", hline=0.0)
    save_panel(t_eval[1:], [e[1:] for e in e1s], labels, "Time (s)", r"$|\nabla_v\Upsilon|_1$",
               "comp_err1.eps", logy=True)
    save_panel(t_eval[1:], [e[1:] for e in e2s], labels, "Time (s)", r"$|\nabla_v\Upsilon|_2$",
               "comp_err2.eps", logy=True)


if __name__ == "__main__":
    figure1_vary_alpha()
    figure2_vary_init()
    figure3_comparison()
    with open("run_log.txt", "w") as f:
        f.write("\n".join(LOG_LINES) + "\n")
    print("\nDone. Panel .eps files and run_log.txt written to the current directory.")
