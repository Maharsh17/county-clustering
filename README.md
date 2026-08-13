# What Kind of Place Is Your County?

Sorting 3,128 US counties into four types using 16 social vulnerability measures plus how much money kids raised poor in a place go on to earn as adults. Then spending most of the effort on a harder question: are those groups real, or an artifact of what somebody decided to measure?

Built with K-Means, Ward hierarchical validation, and a random forest, in Python with scikit-learn and Streamlit.

[![Four types of US county](docs/figures/4_cluster_map.png)](https://maharsh17.github.io/county-clustering/)

### Read Or Try It

| | |
|---|---|
| **[Full write up](https://maharsh17.github.io/county-clustering/)** | The whole project: results, methodology, the bias probe, and every limitation |
| **[Live explorer](https://county-clustering.streamlit.app)** | Uncheck any measure and the model refits in front of you |
| **[Data documentation](data/README.md)** | All 23 columns, the three sources, and how the row counts reconcile |

### The Short Version

Four types fell out of the data, numbered by mean upward mobility so Type 1 is always the lowest.

| Type | Counties | Mean mobility | What stands out |
|---|---|---|---|
| 1. Rural hardship | 987 | 0.390 | Mobile homes, disability, poverty |
| 2. Costly big metros | 630 | 0.412 | Multi-unit housing, housing cost burden |
| 3. Immigrant gateways | 196 | 0.422 | Limited English, crowded housing, minority |
| 4. Comfortable America | 1,315 | 0.470 | Above average mobility, below average minority share |

Refitting without minority share and limited English moved cluster separation by 0.001, and 67% of counties kept their type. Those two columns did nothing statistically. What they bought was the name written on the group. That experiment is the point of the project, and the [live explorer](https://county-clustering.streamlit.app) lets you rerun it yourself.

### Running It

```bash
python -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

./.venv/bin/python analysis.py       # 8 figures, the zoomable map, and combined_clusters.csv
./.venv/bin/python test_app.py       # smoke test for clustering, type ordering, neighbours
./.venv/bin/python -m streamlit run app.py
```

Run `analysis.py` before the app, because the app reads the predictions it writes out.

| Path | What it is |
|---|---|
| `analysis.py` | The whole pipeline, top to bottom |
| `app.py` | The interactive explorer |
| `test_app.py` | Checks type ordering, ablation wiring, and the neighbour search |
| `data/county_svi_mobility.csv` | The merged input, 3,132 counties and 23 columns |
| [`data/README.md`](data/README.md) | Data dictionary and source documentation |
| `docs/` | The published site. `index.md` is the write up, `figures/` holds the generated output |

## Authors

**Maharsh Jani** ([majani2@illinois.edu](mailto:majani2@illinois.edu)), University of Illinois Urbana-Champaign. Built for the AI4ALL Ignite accelerator.

## License

Copyright © 2026 Maharsh Jani. Released under the [GNU Affero General Public License v3.0](LICENSE).

AGPL is the strongest copyleft the OSI approves, and section 13 stretches it to cover network use. Host a modified version as a web service and you owe your users the source. Since the main artifact here is a hosted app, plain GPL would have left that loophole wide open.
