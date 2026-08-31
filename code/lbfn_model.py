"""
Core implementation of the Logarithmic Barrier Function-based Newton (LBFN) model.

Implements the generic prediction-correction dynamics of Eq. (3) in the manuscript,

    v_dot(t) = -[Hess_v Upsilon]^-1 [ alpha * grad_v Upsilon
                                       + grad_vr Upsilon * r_dot
                                       + grad_vw Upsilon * w_dot
                                       + grad_vt Upsilon ]

for a dynamic program

    minimize_v   h(v, t)
    subject to   A_i(t) v <= d_i(t),  i = 1, ..., p

where h(v, t) need not be quadratic: the caller supplies grad_h, hess_h and
grad_h_t (the partial derivative of grad_h with respect to t at fixed v).
This lets the same engine drive both the quadratic numerical example
(Section 4.1 of the manuscript) and the non-quadratic potential-field robot
navigation case study (Section 4.2), which only specializes the (h, grad_h,
hess_h, grad_h_t) callbacks.

Also implements the "Newton flow" (NF) and "continuous gradient descent"
(CGD) baselines as literal ablations of the same barrier-augmented
objective: NF drops the prediction terms (grad_vr*r_dot + grad_vw*w_dot +
grad_vt) from the LBFN update, and CGD additionally drops the Hessian
preconditioning, i.e. both are obtained by deleting terms from the exact
same equations used for LBFN -- not separate, unrelated code paths.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

EPS = 1e-6  # numerical floor to keep the barrier well-defined under discrete integration.
# Kept moderate (not machine precision) on purpose: for a slow-converging
# case (small alpha) combined with a fast-shrinking r(t) (Sec. 2.1), the
# true trajectory can legitimately approach an active constraint boundary
# fast relative to alpha's correction rate. Flooring delta at machine
# precision turns 1/delta^2 into ~1e18 there, which is *correct* in an
# idealized continuous-time sense but makes the discretized ODE extremely
# stiff for an explicit/adaptive-step integrator. Flooring at 1e-6 instead
# saturates the barrier force at a large-but-finite value, trading a small
# amount of fidelity exactly in that boundary-hugging regime for a solve
# that completes in a fraction of a second instead of stalling.


class BarrierConstraints:
    """Time-varying linear inequality constraints A(t) v <= d(t), plus the
    dynamic slack r(t) and barrier weight w(t) schedules of Eq. (2)."""

    def __init__(self, A_of_t, Adot_of_t, d_of_t, ddot_of_t,
                 r0: float, gamma_r: float, w0: float, gamma_w: float):
        self.A_of_t = A_of_t        # t -> (p, n) array
        self.Adot_of_t = Adot_of_t  # t -> (p, n) array
        self.d_of_t = d_of_t        # t -> (p,) array
        self.ddot_of_t = ddot_of_t  # t -> (p,) array
        self.r0, self.gamma_r = r0, gamma_r
        self.w0, self.gamma_w = w0, gamma_w

    def r(self, t):
        return self.r0 * np.exp(-self.gamma_r * t)

    def rdot(self, t):
        return -self.gamma_r * self.r(t)

    def w(self, t):
        return self.w0 * np.exp(self.gamma_w * t)

    def wdot(self, t):
        return self.gamma_w * self.w(t)

    def delta(self, v, t):
        A = self.A_of_t(t)
        d = self.d_of_t(t)
        return self.r(t) - (A @ v - d)  # (p,)


def _barrier_terms(v, t, cons: BarrierConstraints):
    """Common barrier gradient/Hessian building blocks shared by every
    method below (Eqs. (4)-(8) with M v + g replaced by the generic grad_h)."""
    A = cons.A_of_t(t)
    Adot = cons.Adot_of_t(t)
    d = cons.d_of_t(t)
    ddot = cons.ddot_of_t(t)
    delta = np.maximum(cons.r(t) - (A @ v - d), EPS)  # (p,)
    w = cons.w(t)

    barrier_grad = (A / delta[:, None]).sum(axis=0) / w             # sum_i A_i^T / delta_i / w

    barrier_hess = (A.T * (1.0 / delta ** 2)) @ A / w               # sum_i A_i^T A_i / delta_i^2 / w

    grad_vr = (A / (delta ** 2)[:, None]).sum(axis=0) / w
    grad_vw = -(A / delta[:, None]).sum(axis=0) / (w ** 2)

    Adot_v_minus_ddot = Adot @ v - ddot
    grad_vt_barrier = (Adot / delta[:, None]).sum(axis=0) / w \
        + (A * (Adot_v_minus_ddot / (delta ** 2))[:, None]).sum(axis=0) / w

    return barrier_grad, barrier_hess, grad_vr, grad_vw, grad_vt_barrier


def lbfn_rhs(t, v, grad_h, hess_h, grad_h_t, cons: BarrierConstraints, alpha: float):
    """Full LBFN dynamics, Eq. (3)/(9) in the manuscript."""
    bgrad, bhess, grad_vr, grad_vw, grad_vt_barrier = _barrier_terms(v, t, cons)

    grad_upsilon = grad_h(v, t) + bgrad
    hess_upsilon = hess_h(v, t) + bhess
    grad_vt = grad_h_t(v, t) + grad_vt_barrier

    rhs = alpha * grad_upsilon + grad_vr * cons.rdot(t) + grad_vw * cons.wdot(t) + grad_vt
    vdot = -np.linalg.solve(hess_upsilon, rhs)
    return vdot


def newton_flow_rhs(t, v, grad_h, hess_h, grad_h_t, cons: BarrierConstraints, alpha: float):
    """Newton-flow (NF) baseline: same barrier-augmented objective and the
    same Newton (Hessian-preconditioned) correction as LBFN, but WITHOUT the
    prediction terms that compensate for the drift of A(t), d(t). This is
    exactly the mechanism the manuscript's Introduction identifies as
    missing from static/non-predictive solvers."""
    bgrad, bhess, _, _, _ = _barrier_terms(v, t, cons)
    grad_upsilon = grad_h(v, t) + bgrad
    hess_upsilon = hess_h(v, t) + bhess
    vdot = -alpha * np.linalg.solve(hess_upsilon, grad_upsilon)
    return vdot


def gradient_flow_rhs(t, v, grad_h, hess_h, grad_h_t, cons: BarrierConstraints, alpha: float):
    """Continuous gradient descent (CGD) baseline: first-order descent on
    the same barrier-augmented objective (no Hessian preconditioning, no
    prediction terms)."""
    bgrad, _, _, _, _ = _barrier_terms(v, t, cons)
    grad_upsilon = grad_h(v, t) + bgrad
    return -alpha * grad_upsilon


def simulate(rhs_fn, v0, t_span, t_eval, grad_h, hess_h, grad_h_t,
             cons: BarrierConstraints, alpha: float, method: str = "Radau",
             rtol: float = 1e-5, atol: float = 1e-7, max_step: float = 0.01):
    """Integrate one of the RHS functions above over t_span and return the
    trajectory sampled at t_eval, shape (len(t_eval), n).

    Defaults to the implicit Radau method with a bounded max_step: the
    barrier terms make this an increasingly stiff ODE as r(t) -> 0 and
    w(t) -> infinity (by design, Sec. 2.1), and an explicit/adaptive solver
    with tight tolerances (e.g. LSODA at rtol=1e-8) can stall taking
    vanishingly small steps once r(t) approaches machine precision. Radau
    with a capped step size integrates the same dynamics reliably and
    quickly; reduce rtol/atol or max_step if you need tighter accuracy for
    a specific study and are willing to trade off runtime."""
    sol = solve_ivp(
        lambda t, v: rhs_fn(t, v, grad_h, hess_h, grad_h_t, cons, alpha),
        t_span, v0, t_eval=t_eval, method=method, rtol=rtol, atol=atol, max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol.y.T  # (len(t_eval), n)


def gradient_error(v_traj, t_eval, grad_h, cons: BarrierConstraints):
    """||grad_v Upsilon(v(t), t)||: the Lyapunov quantity V(t) in Eq. (10)
    of the manuscript (returned per-entry, not squared/halved), used to
    reproduce the "absolute gradient error" panels in Figures 1-3."""
    out = np.zeros_like(v_traj)
    for k, t in enumerate(t_eval):
        v = v_traj[k]
        bgrad, _, _, _, _ = _barrier_terms(v, t, cons)
        out[k] = np.abs(grad_h(v, t) + bgrad)
    return out
