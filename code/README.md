# Code

Python reference implementation of the LBFN model and the five comparison
baselines used in the manuscript. **The `.eps` figures currently embedded in
both `paper/scientific_reports/main.tex` and `paper/discover_computing/main.tex`
(and in `../figures/`) are the literal, unedited output of running the two
scripts below** -- every quantitative claim in the manuscript's Experimental
Results section (settling times, error floors, constraint-violation windows,
robot-navigation distances) was read directly off these runs, not estimated
from the plots. See `verified_run_logs/` for the exact console output.

## Files

- `lbfn_model.py` - the core LBFN dynamics (Eqs. 3-9), plus the Newton-flow
  (NF) and continuous-gradient-descent (CGD) baselines, implemented as
  literal ablations of the same barrier-augmented objective (NF drops the
  prediction terms; CGD additionally drops the Hessian preconditioning).
  Generic in the objective: works for both a quadratic `h(v,t)` and the
  non-quadratic potential field used in the robot navigation case study.
  Uses the implicit `Radau` integrator with a bounded step size and an
  `EPS=1e-6` barrier floor by default -- see the comments at the top of the
  file for why (an explicit/tight-tolerance solver can stall for small
  `alpha` combined with the prescribed fast-shrinking `r(t)` schedule).
- `baselines.py` - the active-set (AS), sequential quadratic programming
  (SQP), and interior-point ("SeDuMi-style") baselines. These re-solve the
  problem from scratch (warm-started) at each sampled time instant, with no
  drift-prediction term -- the "static solver" failure mode described in
  the manuscript's Introduction.
- `numerical_example.py` - the 2D dynamic QP of Eqs. (18)-(20); regenerates
  `coeff_*.eps` (Figure 1, convergence vs. alpha), `init_*.eps` (Figure 2,
  robustness to initialization, including one deliberately infeasible
  starting point), and `comp_*.eps` (Figure 3, comparison against all five
  baselines), and prints/logs the diagnostics quoted in the manuscript text.
- `robot_navigation.py` - the dynamic robot navigation case study of Eq.
  (21); regenerates `t10_*.eps`, `t20_*.eps`, `t30_*.eps` (Figure 4:
  trajectory, potential force field, and potential landscape at t = 10, 20,
  30 s) and logs distance-to-target / distance-to-obstacle diagnostics.
- `verified_run_logs/` - the exact stdout captured from running the two
  scripts above (`numerical_example_run_log.txt`,
  `robot_navigation_run_log.txt`), kept so every number quoted in the
  manuscript can be traced back to a specific, reproducible run.

## Setup

```bash
pip install -r requirements.txt
```

## Running

Run from inside `paper/scientific_reports/` or `paper/discover_computing/`
(so the regenerated `.eps` files land next to `main.tex`, overwriting the
manuscript's current figures) with these scripts on your `PYTHONPATH`, or
copy them in alongside the `.tex` file:

```bash
python numerical_example.py   # writes coeff_*.eps, init_*.eps, comp_*.eps, run_log.txt
python robot_navigation.py    # writes t10_*.eps, t20_*.eps, t30_*.eps, run_log_robot.txt
pdflatex main.tex && pdflatex main.tex
```

## Notes on fidelity

This is an independent, clean-room implementation derived directly from the
manuscript's equations. Several experimental details are not fully pinned
down by the manuscript text alone (e.g. the exact initial condition for
Figures 1/3, the slack/barrier parameters `r(0)`, `w(0)`, and the random
seed for Figure 2's starting points) -- reasonable choices are made
explicitly in the scripts (and printed in the run logs) so the results are
fully reproducible from this code, even though they are not guaranteed to
be bit-identical to whatever produced an earlier draft's figures. The
manuscript text was written to precisely match this code's verified output,
so text, figures, and code are mutually consistent by construction.
