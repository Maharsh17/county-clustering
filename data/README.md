# The Data Folder

One file: `county_svi_mobility.csv`. Three public datasets joined on county FIPS, 3,132 rows, 23 columns.
It is the only input `analysis.py` and `app.py` ever open. Raw sources are not in the repo.

The original proposal clustered CDC PLACES health outcomes. PLACES says how sick a county is, not what
happens to the children raised in it. Swapping to an upward mobility outcome turned this from a description
into a question.

One section per source, each with the decisions that are not obvious from the column names. The combined
file comes last, because the interesting problems only appear once all three sit in one table.

## 1. CDC/ATSDR Social Vulnerability Index

Supplies 16 of the 17 clustering inputs, every random forest predictor, and all the identifiers. The
modelled columns are `EP_` fields: estimated percentage of population, 0 to 100.

| Column | Meaning | Range here |
|---|---|---|
| `EP_POV150` | Below 150% of the poverty line | 2.6 to 66.6 |
| `EP_UNEMP` | Unemployed | 0.0 to 32.0 |
| `EP_HBURD` | Housing cost burdened | 0.0 to 48.5 |
| `EP_NOHSDP` | No high school diploma | 0.3 to 66.7 |
| `EP_UNINSUR` | No health insurance | 0.0 to 45.1 |
| `EP_AGE65` | Aged 65 and over | 2.9 to 57.9 |
| `EP_AGE17` | Aged 17 and under | 7.1 to 41.2 |
| `EP_DISABL` | With a disability | 4.5 to 41.2 |
| `EP_SNGPNT` | Single parent households | 0.0 to 20.3 |
| `EP_LIMENG` | Limited English speaking | 0.0 to 36.4 |
| `EP_MINRTY` | Minority | 0.0 to 98.3 |
| `EP_MUNIT` | In multi-unit housing | 0.0 to 89.7 |
| `EP_MOBILE` | In mobile homes | 0.0 to 56.3 |
| `EP_CROWD` | In crowded housing | 0.0 to 40.7 |
| `EP_NOVEH` | No vehicle | 0.0 to 85.9 |
| `EP_GROUPQ` | In group quarters | 0.0 to 43.0 |

Also `FIPS`, `COUNTY`, `ST_ABBR`, `E_TOTPOP` (used only for the 50,000 person floor on the resilience
ranking), and `RPL_THEMES`.

### Check The Ranges Before Modelling

`EP_LIMENG` averages 1.6. `EP_POV150` averages 24. K-Means measures straight line distance, so raw
percentages let poverty run the geometry while limited English contributes nothing. Everything gets
standardized to mean 0, standard deviation 1.

That fixes the scale problem and creates a subtler one. Every feature now counts the same, so minority
share weighs as much as unemployment. Nobody chose that. It is the default, and it is what the bias probe
attacks.

### RPL_THEMES Stays Out Of The Models

It is SVI's own vulnerability percentile, built from the same 16 components already going in. Including it
double counts them and imports a CDC weighting decision into a model meant to find its own structure.
Context only.

### The -999 Trap

SVI writes missing values as `-999`, not blank. Load it raw and a county has a poverty rate of negative
nine hundred: no error, no warning, wrecked centroids. The pipeline nulls them first.

This extract is clean and has zero, so the guard is currently a no-op. It stays. The next re-pull might not
be so tidy.

## 2. The Opportunity Atlas

The outcome, and the column most often misread. From Opportunity Insights (Chetty, Friedman, Hendren,
Jones, and Porter). Variable `kfr_pooled_pooled_p25` in `county_outcomes_simple.csv`, landing here as
`MOBILITY`.

| Piece | Meaning |
|---|---|
| `kfr` | Kid family rank. Where the child lands as an adult, as a national income percentile |
| `pooled_pooled` | Pooled across race and gender, so one number covers every child in the county |
| `p25` | Only children whose **parents** sat at the 25th percentile nationally |

In one sentence: take every child raised in a county by parents at the 25th percentile, follow them to ages
31 to 37, and average where their own household income lands nationally.

### Three Ways To Misread It

**A rank, not dollars.** The scale is 0 to 1 and never leaves it. A county at 0.50 raised poor kids who
ended up dead average. Converting to income needs the national distribution alongside.

**Tighter than it looks.** Range is 0.154 (Yakutat City and Borough, AK) to 0.688 (Harding County, SD),
mean 0.430. But the middle 90% sits between 0.345 and 0.550 and the standard deviation is 0.063. A 0.08 gap
between two types is 1.3 standard deviations, not a rounding error. Judge against the band, not the axis.

**It describes the past.** The cohort was observed at ages 31 to 37 in 2014 and 2015, so they grew up in
the 1980s and 1990s. Every other column is current. No modelling choice repairs that.

### The Four Blanks

The Atlas suppresses estimates built on too few children. That is the only source of nulls in the file, and
the same logic behind the 50,000 floor on the resilience ranking.

## 3. NCHS Urban-Rural Classification, 2023

One column, `CODE2023`, and it earns its place by staying out of the models.

| Code | Name | Definition |
|---|---|---|
| 1 | Large central metro | Metro area of 1M+, and contains the largest principal city's whole population, sits entirely inside it, or holds 250,000+ residents of any principal city |
| 2 | Large fringe metro | Metro area of 1M+ but not large central. Suburbs, in practice |
| 3 | Medium metro | Metro area of 250,000 to 999,999 |
| 4 | Small metro | Metro area of 50,000 to 249,999 |
| 5 | Micropolitan | In a micropolitan statistical area, nonmetropolitan |
| 6 | Noncore | Did not qualify as micropolitan. The most rural category |

Codes 1 through 4 are metropolitan, 5 and 6 are not. Built on OMB's July 2023 delineation plus 2022
postcensal population estimates. Fourth version after 2013, 2006, and 1990, using the same rules as 2013
and 2006, so those three compare and 1990 does not. The 2023 revision also dropped the discriminant
analysis that 2006 and 2013 ran to confirm the large central versus large fringe split.

### Holding It Out Paid Off

Every other substantive column goes into the clustering. This one does not, so it cannot have shaped the
group boundaries and can be used afterwards to test whether they mean anything.

Costly big metros came out 80% metropolitan against 23% for Rural hardship. That is the only external
validation in the project, and clustering on it would have made it worthless. It fails honestly too:
Immigrant gateways and Comfortable America land at 33% and 29%, too close to say anything.

### The Change That Broke Connecticut

The NCHS release note lists what is new in 2023, and that list is 9 Connecticut planning regions plus 2
Alaska census areas (Chugach, Copper River).

Connecticut retired its 8 counties for 9 planning regions on new FIPS codes. NCHS followed. The Opportunity
Atlas still publishes against the old codes. The join had nothing to match, raised no error, and dropped a
state. An inner join losing every row of a state looks exactly like a state that was never in the file.

Second consequence: `CODE2023` sits in the row filter, so a county with no NCHS code drops out even when
its SVI and mobility data are complete.

## The Fourth Source, Never Written To Disk

County polygons come from the Plotly sample GeoJSON at runtime. They draw the choropleth, and the app
averages each outline for a rough centroid. Unweighted means, fine for a marker on a national map and wrong
for anything needing real geometry.

## The Combined Dataset

| Group | Columns | Count |
|---|---|---|
| Clustering inputs | 16 `EP_*` plus `MOBILITY` | 17 |
| Held out for validation | `CODE2023` | 1 |
| Context and identifiers | `FIPS`, `COUNTY`, `ST_ABBR`, `E_TOTPOP`, `RPL_THEMES` | 5 |

### Four Row Counts, All Correct

| Count | What it is |
|---|---|
| 3,160 | Records in the NCHS source file, invalid entries included |
| 3,144 | NCHS valid counties in the 2023 scheme |
| **3,132** | **Rows here, after the join** |
| 3,128 | Rows the models see, after dropping nulls |

The join costs 12 counties against NCHS. Nine are Connecticut and two are the new Alaska areas, both
confirmed absent by name. The per-code shortfall matches that story, since Connecticut is mostly
metropolitan and the Alaska pair is rural.

| Code | Here | NCHS 2023 | Short |
|---|---|---|---|
| 1 Large central metro | 66 | 67 | 1 |
| 2 Large fringe metro | 367 | 368 | 1 |
| 3 Medium metro | 390 | 395 | 5 |
| 4 Small metro | 355 | 356 | 1 |
| 5 Micropolitan | 656 | 658 | 2 |
| 6 Noncore | 1,298 | 1,300 | 2 |
| **Total** | **3,132** | **3,144** | **12** |

That accounts for 11. The twelfth needs the raw NCHS file, which is not in this repo, so it stays a known
unknown rather than a guess.

One check worth running: `E_TOTPOP` sums to 327,476,612. Add Connecticut's roughly 3.6 million and you land
at a national total, so nothing else large went missing quietly.

The last 4 rows go when `analysis.py` drops nulls. Petroleum MT, Arthur NE, King TX, and Loving TX are all
missing `MOBILITY` and nothing else.

### Gotchas

- Read `FIPS` as a string. Use `dtype={"FIPS": str}`, then `.str.zfill(5)` before joining to GeoJSON, or pandas eats the leading zero and the failure looks like missing data.
- `COUNTY` is not a key. 1,870 distinct names across 3,132 rows, because every state has a Washington County.
- `CODE2023` arrives as a float despite holding only whole numbers. Cast before comparing.
- Any row count you quote needs to say which of the four it came from.

### Still Open

The SVI extract's release year is not recorded anywhere in this repo. `EP_POV150` and `EP_HBURD` both
arrived in the 2020 SVI release, so it is 2020 or later. Pinning it matters, because the ACS 5 year window
behind those percentages decides how badly the timing mismatch with a 1980s childhood cohort bites.

## Sources

- CDC/ATSDR Social Vulnerability Index, County Level. [atsdr.cdc.gov/place-health/php/svi](https://www.atsdr.cdc.gov/place-health/php/svi/index.html)
- Opportunity Atlas, Opportunity Insights. [opportunityinsights.org/data](https://opportunityinsights.org/data/)
- NCHS Urban-Rural Classification Scheme for Counties, 2023. [cdc.gov/nchs/data-analysis-tools/urban-rural.html](https://www.cdc.gov/nchs/data-analysis-tools/urban-rural.html)
