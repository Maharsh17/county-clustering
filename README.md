# What Kind of Place Is Your County?

I grouped 3,128 US counties by 16 social-vulnerability measures and how much money kids raised poor there earn as adults, then spent most of my effort on a harder question: are the groups I found real, or are they an artifact of what I chose to measure? Built using K-Means clustering, hierarchical validation, and a supervised random forest, in Python with scikit-learn and Streamlit.

I go to school at UIUC and I've always seen a very stark difference between Downtown Champaign and Campus Town. Interestingly, when I looked at **Champaign County Illinois, it is most similar to Ingham County Michigan, Alachua County Florida, Tippecanoe County Indiana, Johnson County Iowa, and Charlottesville City Virginia.** Every one is a college town, and they sit an average of 376 miles apart. Nobody told the model that universities exist. It found them from poverty rates, housing, and age structure alone!

![Four types of US county](figures/4_cluster_map.png)

**[Try it yourself](https://county-clustering.streamlit.app)**: uncheck any measure in the sidebar and the model refits in front of you. Drop Minority and Limited English and watch how little changes.

---

## Problem Statement

Public-health and anti-poverty programs usually rank counties from worst to best and send help down the list. A ranking tells you *who* is struggling. It cannot tell you what kind of trouble they are in.

That distinction decides whether an intervention works. A rural county where a third of homes are mobile homes and a quarter of adults have a disability needs something different from a coastal metro where families are priced out of housing they can otherwise reach. Rank both as "high need" and you lose the only information that would tell you what to send.

The stakes are concrete. Where a child grows up measurably changes what they earn as an adult, and county-level allocation decisions are made every budget cycle with a ranking as the input. If grouping counties by kind is more useful than ordering them by severity, that is a change any agency could adopt tomorrow.

But there is a second problem underneath, and it is the one this project actually spends its time on. Any grouping is a function of what you decided to measure. The Social Vulnerability Index treats minority status as a component of vulnerability. Feed that to a clustering algorithm and it will hand you a group defined by race, which a human will then name. I wanted to know whether my groups were tracking material conditions or ethnicity, so I built an experiment that could have embarrassed me.

## Key Results

1. **Four types of American county emerged**, numbered by mean upward mobility so Type 1 is always the lowest.

   | Type | Counties | Mean mobility | What stands out (z-scores) |
   |---|---|---|---|
   | **1. Rural hardship** | 987 | 0.390 | Mobile homes +1.0, disability +0.9, poverty +0.9 |
   | **2. Costly big metros** | 630 | 0.412 | Multi-unit housing +1.2, housing cost burden +0.8 |
   | **3. Immigrant gateways** | 196 | 0.422 | Limited English +2.4, crowded housing +2.3, minority +2.0 |
   | **4. Comfortable America** | 1,315 | 0.470 | Mobility +0.6, minority -0.6 |

2. **The two demographic features improved cluster separation by 0.001.** I refit without minority share and limited English. Silhouette moved from 0.155 to 0.156, and 67% of counties kept their type (adjusted Rand index 0.735). Statistically those two columns contributed nothing. What they bought was the *name* I wrote on the group.

3. **Sixteen vulnerability measures predict 60% of the variance in upward mobility.** A random forest on a held-out 20% test set reached R² 0.600, MAE 0.0281, RMSE 0.0394, beating linear regression (R² 0.503) by ten points, so the relationships are not straight lines. Unemployment alone carries 26% of the predictive weight.

4. **The 40% the model misses is where the useful finding lives.** RMSE runs 1.40× MAE, meaning a minority of counties are missed badly. Ranking counties by how far they beat their own prediction surfaces Stearns County MN, Ector County TX, and Starr County TX as places doing markedly better for their poor children than their conditions predict. A severity ranking buries every one of them mid-list.

5. **Four statistical tests for the number of clusters gave four different answers** (silhouette 2, Calinski-Harabasz 2, Davies-Bouldin 9, elbow 3). I ran 4 and explain why below. Independent Ward hierarchical clustering, which never uses K, finds a genuine gap at the four-way cut.

6. **[A live interactive version](https://county-clustering.streamlit.app)** lets anyone refit the model themselves. Every checkbox removes a measure and redraws the map, so the claim that these groups depend on what I chose to measure is something you can test rather than take my word for.

## Methodologies

**Type of learning: unsupervised (clustering), with a supervised regression added as a check on it.**

| Method | Inputs | Output | Why |
|---|---|---|---|
| **K-Means** | 16 SVI measures + mobility, standardized | one type label per county | Centroids are directly readable as a profile |
| **PCA** | the same 17 standardized features | 2 components | Only to draw clusters in 2-D |
| **Ward hierarchical** | the same 17 features | dendrogram | Tests whether 4 is a real seam, without using K |
| **Random forest** | the 16 SVI measures only | predicted mobility + importances | Do the inputs predict the outcome at all? |

I standardized every feature to mean 0 and standard deviation 1 before clustering, because K-Means measures straight-line distance and raw percentages would let whichever column happened to have the widest range dominate. Clusters are renumbered by mean mobility after fitting, so Type 1 is the lowest-mobility group on every run regardless of how K-Means happens to order its labels.

**Why K-Means, and what it costs** The centroids are the answer, as each one is a readable profile of a kind of place. A method producing better-separated but uninterpretable groups would be useless here. Against that, K-Means assumes clusters are round and similar-sized, which county data is not. It forces every county into a group, so there is no "unlike anything else" category. K is chosen by me rather than learned. And it is entirely at the mercy of feature selection and scaling, which is precisely the weakness my bias probe exploits.

**Choosing K.** Silhouette and Calinski-Harabasz both prefer K=2, and that is a real finding rather than a bug. At the coarsest level American counties split into "doing fine" and "not doing fine." That split is statistically cleanest and analytically useless, because it is the ranking I was trying to escape. Davies-Bouldin's K=9 yields groups too small to act on. I chose K=4 for interpretability and I say so out loud rather than reporting the K that flatters the metrics.

![Choosing K](figures/2_optimal_k_tests.png)

I did not take that on faith. Ward linkage, built bottom-up and never told about K, shows a long vertical gap right at the four-way cut. It only partly reproduces the K-Means grouping (adjusted Rand index 0.43), which tells me the seam at four is real but exactly where the lines fall depends on the method. That is a limitation, not a footnote.

![Dendrogram](figures/5_dendrogram.png)

**Evaluating the clusters.** Silhouette, Calinski-Harabasz, and Davies-Bouldin across K=2 through 10, plus a visual sweep of the same counties clustered at every K from 2 to 7. The sweep is the honest version of the argument: you can see the groups stay coherent as K rises and then start splitting hairs.

![K sweep](figures/1_k_sweep_grid.png)

Each type is then a profile of z-scores, which is the entire model in one picture.

![The types](figures/3_combined_cluster_profiles.png)

**Evaluating the supervised model.** An 80/20 train-test split, scored on data the model never saw:

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear regression | 0.503 | 0.0323 | 0.0439 |
| **Random forest** | **0.600** | **0.0281** | **0.0394** |

![Model evaluation](figures/6_model_evaluation.png)

RMSE exceeds MAE by a factor of 1.40. Since RMSE squares errors before averaging, that gap means the model is not uniformly a little wrong. It is close on most counties and badly wrong on a few. Those few are exactly what the next figure is built from.

**Finding the counties that beat their odds.** I used cross-validated predictions so that no county is ever scored by a model that saw it during training, then ranked by residual. Counties under 50,000 people are excluded because their mobility estimates rest on too few children.

![Resilience](figures/7_resilience.png)

**Testing my own bias.** I refit the entire pipeline without minority share and limited English, then measured what changed.

![Bias probe](figures/8_bias_probe.png)

| | All 17 measures | Two demographic measures removed |
|---|---|---|
| Silhouette | 0.155 | **0.156** |
| Counties keeping their type | | **66.8%** |
| Adjusted Rand index vs original | | **0.735** |

Type 3 returns as a group defined by crowded housing (+1.8), being uninsured (+1.8), and single parenthood (+1.6). The same places, described by their material conditions instead of their ethnicity.

### How AI/ML Can Amplify or Mitigate Bias Here

My pipeline has three places where bias enters, and the algorithm is not one of them.

**What I chose to measure.** SVI treats minority status as a component of vulnerability. CDC has defensible reasons, since these communities do face compounded barriers in a disaster. But K-Means has no idea *why* a column is present. Standardizing gave all 17 features equal variance, which silently declared that a county's minority share matters exactly as much as its unemployment rate. Nobody decided that. It was the default.

**What I named the result.** The model outputs the integer 2. "Immigrant gateways" is mine. My probe shows the group is held together by crowded housing and lack of insurance, so a more accurate name is something like "high-deprivation young metros." The name I first reached for described who lives there rather than what they are up against, and a policymaker reading it would draw a different conclusion about cause.

**What I would do with it.** Allocating resources by type means every county in a group gets identical treatment. My proposal listed aggregation bias as a caveat. Acting on the clusters turns it into a mechanism.

**Mitigation is why the probe exists.** I did not want to write a disclaimer, so I built a test whose result could have gone against me. It partly did. A third of counties change type when the demographic features come out, so those features are not inert. But separation quality does not improve at all, which means I cannot justify keeping them on performance grounds. In the live app, the checkboxes that reproduce this experiment are the first thing a visitor sees.

**The positive case for this work.** Typing rather than ranking lets you notice that Type 3 counties carry high vulnerability *and* better-than-expected mobility, so copying an intervention from Type 4 into Type 1 may be exactly wrong. The residual ranking points at specific counties doing something right, a question a severity ranking cannot even ask.

**The negative case.** Four labels over 3,128 counties is a large amount of forgetting. "Comfortable America" contains 1,315 counties and some of them are not comfortable. A cluster label is a stereotype with a confidence interval nobody prints.

### Limitations

- **Mobility is historical.** These estimates follow children born around 1978-1983. They describe the county those children grew up in, not necessarily the county that exists now.
- **Small counties are noisy.** Estimates for low-population counties rest on few children. The resilience ranking excludes counties under 50,000, and the raw figures do not.
- **Correlation only.** R² of 0.60 means the features track the outcome. Nothing here shows that changing a county's unemployment rate would change its children's earnings.
- **County averages hide neighborhoods.** Cook County contains some of the highest and lowest mobility tracts in America and appears here as one dot.
- **Coverage.** I analyze 3,128 of roughly 3,143 US counties. Eleven are absent from my merged file and four more were dropped for missing values. I have not audited which, and a systematic absence would bias the result.
- **The four names are editorial.** See the bias probe.

### How the Project Changed

I proposed clustering **CDC PLACES 2024**, 40 measures of disease and access to care. I changed datasets partway through, and the reason was in my own proposal: I had flagged that PLACES values are *modeled estimates* built from a survey plus demographics, not counts. That is fatal for clustering. If two counties look demographically similar, the model that generated PLACES gives them similar health numbers by construction, so clustering PLACES risks rediscovering the imputation model rather than learning anything about health.

I moved to the **CDC/ATSDR Social Vulnerability Index**, built from American Community Survey counts, and joined it to county upward-mobility estimates. That bought me something PLACES could never provide: an actual outcome variable. Every PLACES column is another description of a county. Mobility is a result. Having one made the supervised model, the residual analysis, and the resilience ranking possible.

The hardest problem I hit was the four K-selection tests disagreeing completely. My solution was to stop treating it as a question with one right answer, report all four disagreeing, choose on interpretability grounds, and validate with a second method built on different principles.

## Data Sources <!--- do not change this line -->

- **CDC/ATSDR Social Vulnerability Index**, county level. 16 `EP_*` estimate fields plus the `RPL_THEMES` overall percentile. [atsdr.cdc.gov/place-health/php/svi](https://www.atsdr.cdc.gov/place-health/php/svi/index.html)
- **NCHS Urban-Rural Classification Scheme** (`CODE2023`), 1 = large central metro through 6 = non-core rural. Used as context, not as a clustering input.
- **County upward mobility** (`MOBILITY`): mean adult household income rank for children raised at the 25th percentile. Values fall in the 0.39-0.47 range characteristic of the Opportunity Atlas `kfr_pooled_pooled_p25` measure. *This attribution is provisional until I confirm the exact file and vintage.*
- **County boundaries** for mapping, from the Plotly sample GeoJSON dataset.

### References

1. Chetty, R., Friedman, J. N., Hendren, N., Jones, M. R., & Porter, S. R. (2026). The Opportunity Atlas: Mapping the childhood roots of social mobility. *American Economic Review, 116*(1), 1-51. [NBER w25147](https://www.nber.org/papers/w25147)
2. Flanagan, B. E., Gregory, E. W., Hallisey, E. J., Heitgerd, J. L., & Lewis, B. (2011). A social vulnerability index for disaster management. *Journal of Homeland Security and Emergency Management, 8*(1). [doi:10.2202/1547-7355.1792](https://doi.org/10.2202/1547-7355.1792)
3. Flanagan, B. E., Hallisey, E. J., Adams, E., & Lavery, A. (2018). Measuring community vulnerability to natural and anthropogenic hazards: The CDC's Social Vulnerability Index. *Journal of Environmental Health, 80*(10), 34-36.
4. Chetty, R., Jackson, M. O., Kuchler, T., Stroebel, J., et al. (2022). Social capital I: Measurement and associations with economic mobility. *Nature, 608*, 108-121. [doi:10.1038/s41586-022-04996-4](https://doi.org/10.1038/s41586-022-04996-4)
5. Chetty, R., Jackson, M. O., Kuchler, T., Stroebel, J., et al. (2022). Social capital II: Determinants of economic connectedness. *Nature, 608*, 122-134. [doi:10.1038/s41586-022-04997-3](https://doi.org/10.1038/s41586-022-04997-3)
6. Bowser, D. M., Mauricio, K., Ruscitti, B. A., & Crown, W. H. (2024). American clusters: Using machine learning to understand health and health care disparities in the United States. *Health Affairs Scholar, 2*(3), qxae017. [doi:10.1093/haschl/qxae017](https://doi.org/10.1093/haschl/qxae017)
7. Khan, S. S., Krefman, A. E., McCabe, M. E., Petito, L. C., Yang, X., Kershaw, K. N., Pool, L. R., & Allen, N. B. (2022). Association between county-level risk groups and COVID-19 outcomes in the United States: A socioecological study. *BMC Public Health, 22*, Article 81. [doi:10.1186/s12889-021-12469-y](https://doi.org/10.1186/s12889-021-12469-y)

## Technologies Used <!--- do not change this line -->

- **Python 3.14**
- **scikit-learn** for K-Means, PCA, the random forest, the train/test split, cross-validated prediction, and every clustering and regression metric
- **SciPy** for Ward hierarchical linkage and the dendrogram
- **pandas** and **NumPy** for data preparation and the z-score profiles
- **Matplotlib** for the eight static figures
- **Plotly** for the choropleth maps and interactive charts
- **Streamlit** for the live explorer

### Running It Yourself

```bash
python -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

./.venv/bin/python analysis.py       # writes all 8 figures + combined_clusters.csv
./.venv/bin/python test_app.py       # smoke test: clustering, type ordering, neighbours
./.venv/bin/python -m streamlit run app.py
```

`analysis.py` takes about a minute, most of it the cross-validated random forest. Run it before the app, since the app reads the predictions it writes.

| File | What it is |
|---|---|
| `analysis.py` | The whole pipeline, top to bottom, 8 figures |
| `app.py` | Interactive explorer |
| `test_app.py` | Checks type ordering, ablation wiring, and the neighbour search |
| `figures/combined_clusters.csv` | Every county with its type, prediction, and residual |

In the app, every sidebar checkbox removes a measure and refits everything. Unchecking Minority and Limited English reproduces the bias probe live, and the silhouette readout shows you the delta.

The hosted copy lives at **[county-clustering.streamlit.app](https://county-clustering.streamlit.app)**, running on Streamlit Community Cloud against this repo's `main`, so it redeploys on every push. GitHub Pages serves the write-up you are reading, but it only serves static files and cannot run the app itself.

## Authors <!--- do not change this line -->

**Maharsh Jani** ([majani2@illinois.edu](mailto:majani2@illinois.edu))

Harshitha Sheshala, Anelda Agyei, Cyril Joseph

### License

Copyright © 2026 Maharsh Jani. Released under the [GNU Affero General Public License v3.0](LICENSE).

AGPL is the strongest copyleft license the OSI approves. Anything built on this code has to stay open under the same terms, and section 13 extends that to network use: host a modified version as a web service and you owe your users the source. That clause is why I picked it over plain GPL, since this project's main artifact is a hosted app and a license without it would leave the obvious loophole open.

The underlying data is public and carries its own terms. See the Data Sources section.

### Next Steps

| What | By when |
|---|---|
| Audit which 15 counties are missing and whether their absence is systematic | |
| Weight counties by population in the resilience ranking so it stops favoring small places | |
| Record the demo video | |

Longer term, the residual analysis is the piece worth continuing. Identifying counties that beat their predicted mobility is a real research question, and pairing it with the Chetty social-capital measures would let me test whether cross-class friendship explains the gap my model leaves open.
