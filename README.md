# LBFN: A Logarithmic Barrier Function-Based Newton's Model for Dynamic Quadratic Programming

This repository accompanies the manuscript **"A Logarithmic Barrier Function-Based Newton's Model for Solving Dynamic Quadratic Programming Subject to Inequality Constraints"** by Yuqun Zhou and Jiufan Wang.

The paper proposes a logarithmic barrier function-based Newton (LBFN) model for solving dynamic quadratic programming problems subject to inequality constraints (DQP-IC) in real time, without relying on Karush-Kuhn-Tucker (KKT) reformulations or auxiliary dual variables. The model is shown to be globally convergent from arbitrary initial conditions, with an exponentially decaying, tunable-rate tracking error, and a per-instant computational complexity of O(n^3 + p n^2). The approach is validated numerically against five established solvers and through a real-time robot navigation case study (target tracking with dynamic obstacle avoidance).

The manuscript has been prepared in two versions for submission to two different journals; both versions share the same theoretical and experimental content but differ in formatting and framing to match each journal's scope.

## Repository structure

```
paper/
  scientific_reports/   Manuscript source formatted for Scientific Reports (Nature Portfolio wlscirep.cls)
  discover_computing/   Manuscript source formatted for Discover Computing (Springer Nature sn-jnl.cls)
figures/                Master copies of all figure source files (.eps)
code/                   Implementation code for the LBFN model and baseline solvers (add your scripts here)
```

Each `paper/<journal>/` folder is self-contained: it includes the `.tex` source, the compiled `.pdf`, the journal's LaTeX class file, and all figure files needed to recompile with `pdflatex`.

## Reproducing the manuscript PDF

From within either `paper/scientific_reports/` or `paper/discover_computing/`:

```bash
pdflatex main.tex
pdflatex main.tex   # run twice to resolve cross-references and citation numbers
```

## Code

`code/` contains a Python (NumPy/SciPy/Matplotlib) reference implementation of the LBFN model and all five baselines (AS, SQP, SeDuMi-style interior point, CGD, NF), plus scripts that reproduce the qualitative behavior of Figures 1-4. See `code/README.md` for details, setup, and a note on fidelity to the originally published figures. Both manuscripts' Data/Code Availability statements point to this repository (https://github.com/yzhou364/LBFN); make the repository public before journal submission so that editors and reviewers can access it.

## Citation

If you use this work, please cite (update once the DOI/volume/issue are assigned upon publication):

```bibtex
@article{zhou_wang_lbfn,
  title   = {A Logarithmic Barrier Function-Based Newton's Model for Solving Dynamic Quadratic Programming Subject to Inequality Constraints},
  author  = {Zhou, Yuqun and Wang, Jiufan},
  journal = {[Scientific Reports / Discover Computing -- update once accepted]},
  year    = {2026}
}
```

## License

Code in this repository is released under the MIT License (see `LICENSE`). The manuscript text and figures are the authors' copyrighted work; upon publication in an open-access journal (both Scientific Reports and Discover Computing publish under CC BY), the published version of record will carry the journal's open-access license.
