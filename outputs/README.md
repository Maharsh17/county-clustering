# Outputs

Model artifacts written by `scripts/train.py`. Regenerate them with:

```bash
./.venv/bin/python -m scripts.train
```

| File | What it is |
|---|---|
| `combined_clusters.csv` | Every county with its type, its cross-validated predicted mobility, and the residual between them |

`combined_clusters.csv` is what `app/app.py` reads to show the "vs predicted" figure and the two residual rankings, so run the pipeline before the app.

Figures are not here. They are written straight into `docs/figures/` because their
only purpose is to appear on the published site, which keeps one copy of each
rather than a published duplicate.
