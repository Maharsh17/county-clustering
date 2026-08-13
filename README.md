# What Kind of Place Is Your County?

Sorting 3,128 US counties into four types of place using 16 social vulnerability measures plus how much money kids raised poor there go on to earn as adults. Then spending most of the effort on a harder question: are those groups real, or an artifact of what somebody decided to measure?

Built with K-Means, Ward hierarchical validation, and a random forest, in Python with scikit-learn and Streamlit, as a portfolio project in AI4ALL's Ignite accelerator.

To read the whole thing, with the results, the methodology, the bias probe, and every limitation, go to the **[full write up](https://maharsh17.github.io/county-clustering/)**. To poke at the model yourself, the **[live explorer](https://county-cluster.streamlit.app)** refits it in your browser every time you uncheck a measure.

[![Four types of US county](docs/figures/4_cluster_map.png)](https://maharsh17.github.io/county-clustering/)

## The Data

Three public datasets joined on 5 digit county FIPS, 3,132 counties and 23 columns:

- **CDC/ATSDR Social Vulnerability Index** gives 16 `EP_*` measures like poverty, unemployment, housing cost burden, and disability. These are the clustering inputs and the random forest's only predictors.
- **Opportunity Atlas** gives `MOBILITY`, the mean adult income rank of children whose parents sat at the 25th percentile. It is both a clustering input and the thing the forest tries to predict.
- **NCHS Urban-Rural 2023** gives `CODE2023`, deliberately held out of every model so it can serve as an outside check on whether the types mean anything.

For all 23 columns, the decisions behind each source, and how four different county counts all turn out to be correct, see **[`data/README.md`](data/README.md)**.

## The Short Version

Four types fell out of the data, numbered by mean upward mobility so Type 1 is always the lowest.

| Type | Counties | Mean mobility | What stands out |
|---|---|---|---|
| 1. Rural hardship | 987 | 0.390 | Mobile homes, disability, poverty |
| 2. Costly big metros | 630 | 0.412 | Multi-unit housing, housing cost burden |
| 3. Immigrant gateways | 196 | 0.422 | Limited English, crowded housing, minority |
| 4. Comfortable America | 1,315 | 0.470 | Above average mobility, below average minority share |

Refitting without minority share and limited English moved cluster separation by 0.001, and 67% of counties kept their type. Those two columns did nothing statistically. What they bought was the name written on the group, and that experiment is the point of the project.

## Running It

```bash
python -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

./.venv/bin/python -m scripts.train       # 8 figures, the zoomable map, and combined_clusters.csv
./.venv/bin/python -m tests.test_app       # smoke test for clustering, type ordering, neighbours
./.venv/bin/python -m streamlit run app/app.py
```

`scripts/train.py` takes about 20 seconds. Run it before the app, because the app reads the predictions it writes out.

```
county-clustering/
├── config/
│   └── config.yaml                 K, the seed, the population floor, the feature list
├── src/                            shared by the pipeline and the app
│   ├── data.py                     loading and cleaning the merged file
│   ├── model.py                    clustering, mobility-ordered relabelling, neighbours
│   ├── evaluate.py                 the four cluster-quality tests
│   └── utils.py                    config, palettes, paths
├── scripts/
│   └── train.py                    the whole pipeline, top to bottom
├── app/
│   └── app.py                      the interactive explorer
├── tests/
│   └── test_app.py                 type ordering, ablation wiring, neighbour search
├── data/
│   ├── county_svi_mobility.csv     the merged input, 3,132 counties x 23 columns
│   └── README.md                   data dictionary and source documentation
├── outputs/
│   └── combined_clusters.csv       every county with its type, prediction, residual
├── docs/                           the published site
│   ├── README.md                   the full write up
│   ├── _config.yml
│   ├── assets/css/style.scss
│   └── figures/                    8 PNGs plus the zoomable map
└── requirements.txt                everything the app and the pipeline need
```

## Authors

**Maharsh Jani** ([majani2@illinois.edu](mailto:majani2@illinois.edu)), University of Illinois Urbana-Champaign.

## License

Copyright © 2026 Maharsh Jani. Released under the [GNU Affero General Public License v3.0](LICENSE).

AGPL is the strongest copyleft the OSI approves, and section 13 stretches it to cover network use. Host a modified version as a web service and you owe your users the source. Since the main artifact here is a hosted app, plain GPL would have left that loophole wide open.
