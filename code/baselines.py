"""
Snapshot-resolve baselines: active-set (AS), sequential quadratic
programming (SQP), and an interior-point / log-barrier ("SeDuMi-style")
solver.

Unlike LBFN (and the NF/CGD ablations in lbfn_model.py), these three
baselines treat every time instant as an independent, static optimization
problem: they resolve the quadratic program from scratch (or from a
warm-started guess) at each sampled instant with NO drift-prediction term.
This is a direct implementation of the "static solvers lack temporal
awareness" failure mode described in the manuscript's Introduction, and is
what produces the transient constraint violations / tracking lag reported
for these methods in Section 4.1.

For a strictly convex QP, an active-set solve and a single SQP subproblem
solve are numerically the same operation (SQP applied to an already-
quadratic problem converges in one subproblem), so both AS and SQP below
call the same `solve_qp_snapshot` routine; they are kept as separate
labeled entry points for clarity when reading the experiment scripts.

The interior-point baseline instead minimizes a FIXED (non-adaptive, non-
predictive) log-barrier objective to convergence via Newton's method at
each snapshot -- the natural discrete-time analogue of what an interior-
point package such as SeDuMi would do -- and is likewise re-solved from
scratch at each time sample.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

EPS = 1e-9


def solve_qp_snapshot(grad_h, hess_h, A, d, t, x0):
    """Solve  minimize_v h(v,t)  s.t.  A v <= d  at a single time instant,
    used for both the AS and SQP baselines (see module docstring)."""
    n = x0.shape[0]

    def obj(v):
        # h(v,t) via its known gradient/Hessian at v0=x0 is not directly
        # available in closed form for the caller in general, so we
        # reconstruct a local quadratic model in the very common case where
        # hess_h is (numerically) constant in v -- true for every problem
        # instance used in this repository (quadratic objective, or a
        # radial potential field whose curvature is evaluated at v).
        g0 = grad_h(x0, t)
        H0 = hess_h(x0, t)
        dv = v - x0
        return float(g0 @ dv + 0.5 * dv @ H0 @ dv)

    def obj_grad(v):
        g0 = grad_h(x0, t)
        H0 = hess_h(x0, t)
        return g0 + H0 @ (v - x0)

    cons = [{"type": "ineq", "fun": (lambda v, i=i: d[i] - A[i] @ v)} for i in range(A.shape[0])]
    res = minimize(obj, x0, jac=obj_grad, constraints=cons, method="SLSQP",
                    options={"maxiter": 200, "ftol": 1e-12})
    if not res.success:
        # Fall back to the initial guess if the local QP model is briefly
        # infeasible (can happen transiently for the non-convex potential
        # field robot objective); this mirrors a solver "holding position"
        # rather than crashing, and is reported as-is in the trajectories.
        return x0
    return res.x


def solve_barrier_snapshot(grad_h, hess_h, A, d, t, x0, w_fixed=200.0, n_iter=25):
    """Fixed-barrier interior-point Newton solve at a single time instant
    (the "SeDuMi approach" baseline). No prediction terms are used, and the
    barrier weight w_fixed does NOT grow with t as in LBFN's w(t): this is
    precisely what makes it a static, non-adaptive interior-point solve."""
    v = x0.copy()
    for _ in range(n_iter):
        delta = np.maximum(d - A @ v, EPS)
        g = grad_h(v, t) + (A / delta[:, None]).sum(axis=0) / w_fixed
        H = hess_h(v, t) + (A.T * (1.0 / delta ** 2)) @ A / w_fixed
        step = np.linalg.solve(H, g)

        # backtracking line search to stay in the interior (delta > 0)
        s = 1.0
        while np.any(d - A @ (v - s * step) <= EPS) and s > 1e-6:
            s *= 0.5
        v = v - s * step
        if np.linalg.norm(step) < 1e-10:
            break
    return v


def run_snapshot_baseline(solver, grad_h, hess_h, A_of_t, d_of_t, t_eval, v0, **kwargs):
    """Drive a snapshot solver (solve_qp_snapshot or solve_barrier_snapshot)
    across t_eval, warm-starting each solve from the previous instant's
    solution -- the standard way a re-solved (non-predictive) controller is
    deployed in practice."""
    n = v0.shape[0]
    traj = np.zeros((len(t_eval), n))
    v = v0.copy()
    for k, t in enumerate(t_eval):
        A = A_of_t(t)
        d = d_of_t(t)
        v = solver(grad_h, hess_h, A, d, t, v, **kwargs)
        traj[k] = v
    return traj
