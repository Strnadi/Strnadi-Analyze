"""One experiment per file, each a self-contained script in notebook style (`# %%` cells).

Every script reads top to bottom:

    markdown header   what the experiment is, the question it answers, its status and result
    Imports
    Settings          a `TrainSettings(...)` listing what differs from the defaults
    Model             `make_model()`, built from `model/` and `layers/`
    Train             `cross_validate(RUN, make_model, settings)`
    Evaluate          `summarize_run`, `plot_run`, `plot_error_examples` from `evaluation/report.py`

Run one from the repository root:

    python -m experiments.r2_factorized_weak

or open it in an editor that understands `# %%` cells (VS Code, PyCharm, Jupytext) with the
repository root as working directory and step through it.

    python -m tools.list_experiments            every experiment with its origin and status
    STRNADI_DRY_RUN=1 python -m experiments.X   build the model and stop before training
    STRNADI_FOLDS=0,1 STRNADI_THREADS=5 ...     train only some folds / set the thread count
    STRNADI_SEED=1 ...                          another seed; the run is stored as <name>_s1
"""
