# Canva Presentation Prompt

Working file for building the AI4ALL final presentation deck. Paste the block
below into Canva's AI presentation generator, then upload the figures listed
underneath. Numbers here match what `analysis.py` produces, so if the pipeline
output changes, this needs updating too.

---

## The Prompt

```
Create a 13-slide presentation for a student data science showcase.

STYLE
Clean and editorial. White or off-white background, one strong accent color
(blue #2a78d6), dark charcoal text, generous whitespace. Sans-serif. Big
readable numbers. No stock photos of people pointing at laptops. No clipart.
Every slide gets a headline that states a finding, not a topic label. Plain
conversational language, no em-dashes, no semicolons, no corporate filler.

TITLE
What Kind of Place Is Your County?
Subtitle: Grouping 3,128 US counties by social vulnerability and upward
mobility, then testing whether the groups are real
Presenter: Maharsh Jani, AI4ALL Ignite

SLIDE 1 - Title slide as above.

SLIDE 2 - "Ranking tells you who is struggling. It cannot tell you what kind
of trouble they are in." Anti-poverty programs rank counties worst to best
and send help down the list. A rural county where a third of homes are mobile
homes needs something different from a metro where families are priced out.
Rank both as high need and you lose the only information that tells you what
to send.

SLIDE 3 - "Three public datasets, joined on county FIPS." Table:
CDC Social Vulnerability Index gives 16 measures like poverty, unemployment,
housing burden. Opportunity Atlas gives upward mobility, which is the adult
income rank of kids raised poor. NCHS gives an urban-rural code. 3,128
counties across 49 states and DC.

SLIDE 4 - "Unsupervised learning. K-Means clustering." Inputs are 17
standardized features, which is the 16 vulnerability measures plus mobility.
Output is one type label per county. Chosen because the cluster centers read
directly as a profile of a kind of place. Costs are that it assumes round
equal-sized groups, forces every county into one, and K is picked by a human.

SLIDE 5 - "Four tests for K gave four different answers." Silhouette says 2.
Calinski-Harabasz says 2. Elbow says 3. Davies-Bouldin says 9. Follow the
rules to the letter and nothing resolves. Read all four together and K=4 is
the only value that holds up on every one at once. Silhouette and
Calinski-Harabasz both third highest, Davies-Bouldin at a local minimum,
inertia still inside its bend.

SLIDE 6 - "Four types of American county." Table with four rows:
Rural hardship, 987 counties, mobility 0.390, high mobile homes and disability.
Costly big metros, 630 counties, mobility 0.412, high multi-unit housing.
Immigrant gateways, 196 counties, mobility 0.422, high limited English and
crowded housing. Comfortable America, 1,315 counties, mobility 0.470.

SLIDE 7 - Full-bleed map slide. Headline "Nobody gave the model a latitude."
Leave the center empty for a screenshot. Caption: Type 1 traces Appalachia and
the rural South. Type 3 lands on the border and the coastal port metros.

SLIDE 8 - "Champaign County's five closest matches are all college towns."
Ingham Michigan, Alachua Florida, Tippecanoe Indiana, Johnson Iowa,
Charlottesville City Virginia. They sit an average of 377 miles apart. Nobody told
the model that universities exist. It found them from poverty rates, housing
and age structure alone.

SLIDE 9 - "Do those 16 measures actually predict anything?" A random forest
trained on 80 percent and scored on the held-out 20 percent. R squared 0.600,
MAE 0.0281, RMSE 0.0394. Linear regression only reached 0.503, so the
relationships bend. Unemployment alone carries 26 percent of the predictive
weight.

SLIDE 10 - "The 40 percent the model misses is the interesting part."
Immigrant gateways carries the heaviest vulnerability load of any type and its
kids still beat the prediction by 0.012, with 61 percent of its counties above
the line. Rural hardship misses low. Checked against regression to the mean,
correlation between prediction and residual is 0.02, so it is real.

SLIDE 11 - "Feed the model race and it hands back a race-shaped group."
SVI counts minority status as vulnerability. Out came a cluster at plus 2.0
minority and plus 2.4 limited English, and a human typed Immigrant gateways
next to it. So the whole pipeline got refit without those two features.

SLIDE 12 - "Those two features improved separation by 0.001." Silhouette went
from 0.155 to 0.156. Two thirds of counties kept their type. The group came
back defined by crowded housing, being uninsured and single parenthood. The
same places described by their material conditions instead of their ethnicity.
What those columns bought was the name, not the grouping.

SLIDE 13 - "Positive and negative, then what is next." Positive is that typing
instead of ranking reveals counties doing better than their conditions predict,
a question ranking cannot ask. Negative is that four labels over 3,128 counties
is a lot of forgetting, and a cluster label is a stereotype with a confidence
interval nobody prints. Next steps are fixing the Connecticut join, weighting
the resilience ranking by population, and recording a demo.
Links: github.com/Maharsh17/county-clustering and county-clustering.streamlit.app
```

---

## Figures To Upload

Canva will not generate these. Pull them from `figures/` once the deck exists.

| Slide | File |
|---|---|
| 5 | `2_optimal_k_tests.png` |
| 6 | `3_combined_cluster_profiles.png` |
| 7 | `4_cluster_map.png` |
| 9 | `6_model_evaluation.png` |
| 10 | `7_resilience.png` |
| 12 | `8_bias_probe.png` |

Six visualizations. The rubric's top band asks for three.

## Rubric Coverage

| Criterion | Slides |
|---|---|
| Project Description | 2, 3 |
| Data Visualization | 5, 6, 7, 9, 10, 12 |
| Algorithm with type, inputs, outputs, pros and cons | 4 |
| Essential Question | 11, 12 |
| Positive and negative impact | 13 |
| Next Steps | 13 |
| GitHub link | 13 |

## Two Things To Add By Hand

**A references slide.** All seven citations from the README. Missing citations
scores a zero rather than a deduction, so this is not optional.

**How the project changed.** The rubric wants a comment on how the project
evolved plus one challenge that got solved. Neither lives in the README
anymore, so both need to be in the talk track. The four disagreeing K tests on
slide 5 are the challenge. The switch away from CDC PLACES is the evolution,
and it earns a sentence on slide 3. PLACES publishes modeled estimates rather
than counts, so clustering it risks rediscovering the imputation model instead
of learning anything about health. Swapping to SVI also brought in a real
outcome variable, which is what made the supervised model and the residual
analysis possible at all.

## Talk Track Timing

Ten minutes across 13 slides is about 45 seconds each. Slides 8, 10 and 12
carry the findings people will remember, so give those closer to 75 seconds
and move faster through 3 and 4.

The live demo is worth 60 seconds on its own. Open the app, search a county
someone in the room knows, then uncheck Minority and Limited English and let
the silhouette delta do the talking.
