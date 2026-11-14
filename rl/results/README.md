# rl/results: committed result summaries

One CSV per run (M4's schema): every row carries environment version, episode-set hash, threshold A, configuration
hash and seed, so the invariant checks (I2, I3, I5, I10) can run on them. Raw runs stay in `rl/runs/` (git-ignored).
`checks/` holds the JSON summaries written by `python -m rl.scripts.check`.
