"""County social-mobility clustering: 16 SVI vulnerability measures plus upward mobility,
urban-rural as context. Run: ./.venv/bin/python analysis.py

Copyright (C) 2026 Maharsh Jani. Licensed under the GNU Affero General
Public License v3.0 or later. See LICENSE, or <https://www.gnu.org/licenses/>.
"""
import json
import textwrap
from urllib.request import urlopen
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd, numpy as np, os
import plotly.express as px
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score, r2_score, mean_absolute_error,
                             root_mean_squared_error)

# figures are written straight into the published site, so there is one copy of
# each and index.md can reference them relatively. See docs/_config.yml.
OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)
INK, MUT = "#0b0b0b", "#8a8a86"
DIVERGE = LinearSegmentedColormap.from_list("bwr", ["#256abf", "#f4f4f2", "#e34948"])
PAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4"]

FEATS = {
    "EP_POV150": "Poverty (<150%)", "EP_UNEMP": "Unemployment", "EP_HBURD": "Housing cost burden",
    "EP_NOHSDP": "No HS diploma", "EP_UNINSUR": "Uninsured", "EP_AGE65": "Aged 65+",
    "EP_AGE17": "Aged 17 & under", "EP_DISABL": "Disability", "EP_SNGPNT": "Single-parent",
    "EP_LIMENG": "Limited English", "EP_MINRTY": "Minority", "EP_MUNIT": "Multi-unit housing",
    "EP_MOBILE": "Mobile homes", "EP_CROWD": "Crowded housing", "EP_NOVEH": "No vehicle",
    "EP_GROUPQ": "Group quarters", "MOBILITY": "Upward mobility",
}
COLS = list(FEATS)
K = 4

# Clusters are renumbered by mean upward mobility (see the `remap` step below), so
# type 0 is always the lowest-mobility group. These names are mine, not the model's,
# and they only hold for K=4 on the full feature set. Every figure uses this labelling.
NAMES = ["Rural hardship", "Costly big metros", "Immigrant gateways", "Comfortable America"]
LABELS = [f"Type {i+1}\n{NAMES[i]}" for i in range(K)]

plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#cccccc",
                     "axes.grid": True, "grid.color": "#eeeeee", "figure.dpi": 130})

df = pd.read_csv("data/county_svi_mobility.csv", dtype={"FIPS": str}).replace(-999, np.nan)
df = df.dropna(subset=COLS + ["CODE2023", "E_TOTPOP"]).reset_index(drop=True)
print("combined dataset:", len(df), "counties x", len(COLS), "features\n")

X = StandardScaler().fit_transform(df[COLS])

pca = PCA(n_components=2)
xy = pca.fit_transform(X)
ev = pca.explained_variance_ratio_
fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
sil_by_k = {}
for ax, k in zip(axes.ravel(), range(2, 8)):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sil_by_k[k] = silhouette_score(X, km.labels_)
    for c in range(k):
        m = km.labels_ == c
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.45, color=PAL[c], edgecolors="none")
    ax.set_title(f"K = {k}    silhouette = {sil_by_k[k]:.3f}", fontsize=11, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#cccccc")
fig.suptitle("Same counties, clustered at K = 2 to 7  (PCA projection)", fontsize=14, weight="bold")
fig.text(0.5, 0.005, f"Axes are the first two principal components "
         f"({ev[0]*100:.0f}% + {ev[1]*100:.0f}% of variance). "
         "Higher silhouette = cleaner separation.", ha="center", fontsize=9, color=MUT)
fig.tight_layout(rect=[0, 0.02, 1, 0.97])
fig.savefig(f"{OUT}/1_k_sweep_grid.png", bbox_inches="tight")
print("saved docs/figures/1_k_sweep_grid.png")
print("silhouette by K:", {k: round(v, 3) for k, v in sil_by_k.items()})

Ks = list(range(2, 11))
inertia, sil, ch, db = [], [], [], []
for k in Ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    lab = km.labels_
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, lab))
    ch.append(calinski_harabasz_score(X, lab))
    db.append(davies_bouldin_score(X, lab))

print("\n--- each test's pick ---")
print("silhouette (higher better)       -> K =", Ks[int(np.argmax(sil))])
print("calinski-harabasz (higher better)-> K =", Ks[int(np.argmax(ch))])
print("davies-bouldin (lower better)    -> K =", Ks[int(np.argmin(db))])
d2 = np.diff(inertia, 2)
print("elbow (inertia, sharpest bend)   -> K =", Ks[int(np.argmax(d2)) + 1])

fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
specs = [("Elbow  (inertia, look for the bend)", inertia, "#2a78d6", "lower, bend"),
         ("Silhouette  (higher = better)", sil, "#1baf7a", "max"),
         ("Calinski-Harabasz  (higher = better)", ch, "#eda100", "max"),
         ("Davies-Bouldin  (lower = better)", db, "#e34948", "min")]
for a, (title, vals, color, goal) in zip(ax.ravel(), specs):
    a.plot(Ks, vals, "-o", color=color, lw=2, markersize=6)
    best = Ks[int(np.argmax(vals))] if goal == "max" else (
           Ks[int(np.argmin(vals))] if goal == "min" else None)
    if best is not None:
        a.axvline(best, color="#888", ls="--", lw=1)
        a.text(best, a.get_ylim()[1], f" K={best}", color="#555", fontsize=9, va="top")
    a.set_title(title, fontsize=11, weight="bold")
    a.set_xlabel("number of clusters (K)")
    a.grid(color="#eee")
    a.set_axisbelow(True)
fig.suptitle("How many clusters? Four tests on the combined dataset", fontsize=14, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/2_optimal_k_tests.png", bbox_inches="tight")
print("saved docs/figures/2_optimal_k_tests.png")

km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
df["cluster"] = km.labels_
cent = pd.DataFrame(km.cluster_centers_, columns=[FEATS[c] for c in COLS])
order = cent["Upward mobility"].sort_values().index.tolist()
remap = {old: i for i, old in enumerate(order)}
df["cluster"] = df["cluster"].map(remap)
cent = cent.iloc[order].reset_index(drop=True)

prof = df.groupby("cluster").agg(mobility=("MOBILITY", "mean"), vuln=("RPL_THEMES", "mean"),
                                  urban_rural=("CODE2023", "mean"), n=("FIPS", "size")).round(3)
print("\ncluster profiles (sorted low->high mobility):")
print(prof.to_string())

print("\n=== what DEFINES each cluster (|z| >= 0.6) ===")
for c in range(K):
    row = cent.iloc[c]
    hi = row[row >= 0.6].sort_values(ascending=False)
    lo = row[row <= -0.6].sort_values()
    print(f"\nType {c+1} ({NAMES[c]}):  HIGH -> " + ", ".join(f"{n} (+{v:.1f})" for n, v in hi.items()))
    print(f"           LOW  -> " + ", ".join(f"{n} ({v:.1f})" for n, v in lo.items()))

print("\n=== 4 biggest counties in each type ===")
for c in range(K):
    d = df[df.cluster == c].nlargest(4, "E_TOTPOP")
    names = ", ".join(f"{r.COUNTY} {r.ST_ABBR}" for _, r in d.iterrows())
    print(f"Type {c+1} ({NAMES[c]}): {names}")

fig, ax = plt.subplots(figsize=(9.5, 7))
M = cent.T.values
im = ax.imshow(M, cmap=DIVERGE, vmin=-1.6, vmax=1.6, aspect="auto")
ax.grid(False)
ax.set_xticks(range(K))
ax.set_xticklabels(LABELS, fontsize=9.5)
ax.set_yticks(range(len(COLS)))
ax.set_yticklabels([FEATS[c] for c in COLS])
ax.axhline(15.5, color="#555", lw=1.2, ls="--")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 1.0 else INK)
ax.set_title("What defines each type (z-score vs national average)", fontsize=13, weight="bold", pad=10)
cb = fig.colorbar(im, ax=ax, shrink=0.6)
cb.set_label("above / below US average", fontsize=9)
fig.text(0.5, -0.01, "Dashed line separates the mobility outcome from the 16 vulnerability inputs",
         ha="center", fontsize=8.5, color=MUT)
fig.tight_layout()
fig.savefig(f"{OUT}/3_combined_cluster_profiles.png", bbox_inches="tight")
print("\nsaved docs/figures/3_combined_cluster_profiles.png")

ORDER = [f"Type {i+1}: {NAMES[i]}" for i in range(K)]
MAP_COLORS = dict(zip(ORDER, ["#2a78d6", "#1baf7a", "#e34948", "#eda100"]))

map_df = df[["FIPS", "cluster"]].copy()
map_df["FIPS"] = map_df["FIPS"].str.zfill(5)
map_df["Type"] = map_df["cluster"].map(dict(enumerate(ORDER)))
print("\ncounties to map:", len(map_df), "| types:", map_df["Type"].value_counts().to_dict())

URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
with urlopen(URL) as resp:
    counties = json.load(resp)
print("loaded county geometries:", len(counties["features"]))

fig = px.choropleth(
    map_df, geojson=counties, locations="FIPS", color="Type",
    color_discrete_map=MAP_COLORS, category_orders={"Type": ORDER},
    scope="usa", title="Four social-mobility types of US counties",
)
fig.update_traces(marker_line_width=0)
fig.update_layout(legend_title_text="County type", title_x=0.5, margin=dict(l=0, r=0, t=50, b=0))
# ponytail: fixed div_id, otherwise plotly stamps a fresh uuid every run and git
# sees the whole 16 MB file as changed when nothing about the map moved
fig.write_html(f"{OUT}/4_cluster_map.html", div_id="county-cluster-map")
fig.write_image(f"{OUT}/4_cluster_map.png", width=1100, height=700, scale=2)
print("saved docs/figures/4_cluster_map.html and .png")

# --- 5. hierarchical clustering: is K=4 natural or forced? -------------------
# Ward linkage on Euclidean distance. metric="euclidean" is scipy's default and is
# also the only metric Ward accepts, since its criterion is the increase in
# within-cluster sum of squares. X is already standardized, so this is the same
# space K-Means works in: same objective, minimized bottom-up instead of top-down.
Z = linkage(X, method="ward", metric="euclidean")
hier = fcluster(Z, t=K, criterion="maxclust")
ari_hier = adjusted_rand_score(km.labels_, hier)
print(f"\n=== hierarchical check ===")
print(f"Ward linkage cut at {K} groups vs K-Means: adjusted Rand index = {ari_hier:.3f}")
print("hierarchical group sizes:", np.bincount(hier)[1:])

fig, ax = plt.subplots(figsize=(11, 5.5))
dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, color_threshold=Z[-(K - 1), 2],
           above_threshold_color=MUT, no_labels=True)
ax.axhline(Z[-(K - 1), 2], color="#e34948", ls="--", lw=1.4)
ax.text(0.01, Z[-(K - 1), 2], f"  cut here = {K} groups", color="#e34948",
        fontsize=9.5, va="bottom", transform=ax.get_yaxis_transform())
gap4 = Z[-3, 2] - Z[-4, 2]
gap2 = Z[-1, 2] - Z[-2, 2]
ax.set_title("A second opinion, and it does not simply agree",
             fontsize=14, weight="bold", pad=10)
ax.set_ylabel("Ward merge cost\n(not a distance, it scales with cluster size)")
ax.set_xlabel("counties (bottom 30 branches merged)")
ax.grid(False)
# wrapped, or bbox_inches="tight" widens the canvas to fit one long line and the
# dendrogram ends up a narrow strip floating in whitespace
fig.text(0.5, -0.02, textwrap.fill(
         f"Ward merges counties bottom-up and never uses K. Its widest gap is at TWO "
         f"groups ({gap2:.0f} units of empty tree), not four. The four-way cut sits in a crowded "
         f"stretch with only {gap4:.1f} units under it, so the dendrogram does not independently "
         f"argue for four. What it does show is moderate agreement with K-Means once you cut there "
         f"(adjusted Rand index = {ari_hier:.2f}).", 130),
         ha="center", va="top", fontsize=8.5, color=MUT)
fig.tight_layout()
fig.savefig(f"{OUT}/5_dendrogram.png", bbox_inches="tight")
print("saved docs/figures/5_dendrogram.png")

# --- 6. supervised check: can the 16 inputs predict the outcome? -------------
SVI_ONLY = [c for c in COLS if c != "MOBILITY"]
Xs, y = df[SVI_ONLY].values, df["MOBILITY"].values
Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)

lin = LinearRegression().fit(Xtr, ytr)
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr, ytr)
print("\n=== supervised evaluation (80/20 split, held-out test set) ===")
print(f"{'model':20s} {'R2':>7} {'MAE':>8} {'RMSE':>8}")
for name, m in [("linear regression", lin), ("random forest", rf)]:
    p = m.predict(Xte)
    print(f"{name:20s} {r2_score(yte, p):7.3f} {mean_absolute_error(yte, p):8.4f} "
          f"{root_mean_squared_error(yte, p):8.4f}")
pred_te = rf.predict(Xte)
r2_rf = r2_score(yte, pred_te)
mae_rf = mean_absolute_error(yte, pred_te)
rmse_rf = root_mean_squared_error(yte, pred_te)
# RMSE > MAE means a minority of counties are missed badly. Those are the
# residual outliers figure 7 is built from, so the gap is the point, not noise
print(f"\nRMSE/MAE ratio = {rmse_rf/mae_rf:.2f}  "
      f"(1.0 would mean every county is missed by the same amount)")

imp = pd.Series(rf.feature_importances_, index=[FEATS[c] for c in SVI_ONLY]).sort_values()
print("\ntop 5 predictors of upward mobility:")
print(imp.tail(5)[::-1].round(3).to_string())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
a1.scatter(yte, pred_te, s=10, alpha=0.4, color="#2a78d6", edgecolors="none")
lims = [min(yte.min(), pred_te.min()), max(yte.max(), pred_te.max())]
a1.plot(lims, lims, ls="--", color="#e34948", lw=1.3)
a1.set_xlabel("actual upward mobility")
a1.set_ylabel("predicted upward mobility")
a1.set_title(f"Held-out predictions\nR² = {r2_rf:.2f}   MAE = {mae_rf:.4f}   RMSE = {rmse_rf:.4f}",
             fontsize=11.5, weight="bold")
a1.text(0.03, 0.95, "points on the line = perfect prediction", transform=a1.transAxes,
        fontsize=8.5, color=MUT, va="top")
top = imp.tail(10)
a2.barh(range(len(top)), top.values, color=["#e34948" if n in ("Minority", "Limited English")
        else "#2a78d6" for n in top.index], edgecolor="white")
a2.set_yticks(range(len(top)))
a2.set_yticklabels(top.index, fontsize=9)
a2.set_xlabel("share of predictive power")
a2.grid(axis="y", visible=False)
a2.set_title("What the model leans on", fontsize=11.5, weight="bold")
a2.text(0.97, 0.05, "red = demographic feature", transform=a2.transAxes,
        fontsize=8.5, color="#e34948", ha="right")
fig.suptitle("Random forest: predicting a county's upward mobility from its 16 vulnerability measures",
             fontsize=13, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/6_model_evaluation.png", bbox_inches="tight")
print("saved docs/figures/6_model_evaluation.png")

# --- 7. resilience: counties that beat their prediction ----------------------
# cross_val_predict so every county is scored by a model that never saw it
df["pred"] = cross_val_predict(RandomForestRegressor(n_estimators=300, random_state=42,
                                                     n_jobs=-1), Xs, y, cv=5)
df["residual"] = df["MOBILITY"] - df["pred"]
big = df[df.E_TOTPOP >= 50_000].copy()   # small-county mobility estimates are noisy
over = big.nlargest(12, "residual")
under = big.nsmallest(12, "residual")
print(f"\n=== resilience (counties >= 50k people, n={len(big)}) ===")
print("beating their profile:", ", ".join(f"{r.COUNTY} {r.ST_ABBR}" for _, r in over.head(5).iterrows()))
print("falling short:       ", ", ".join(f"{r.COUNTY} {r.ST_ABBR}" for _, r in under.head(5).iterrows()))

fig, ax = plt.subplots(figsize=(9.5, 7))
show = pd.concat([under[::-1], over])
labels = [f"{r.COUNTY}, {r.ST_ABBR}" for _, r in show.iterrows()]
ax.barh(range(len(show)), show.residual.values,
        color=["#e34948" if v < 0 else "#1baf7a" for v in show.residual],
        edgecolor="white")
ax.set_yticks(range(len(show)))
ax.set_yticklabels(labels, fontsize=8.5)
ax.axvline(0, color=INK, lw=1)
ax.set_xlabel("actual mobility minus predicted mobility")
ax.set_title("Which counties outperform what their conditions predict?",
             fontsize=13, weight="bold", pad=10)
ax.grid(axis="y", visible=False)
ax.set_axisbelow(True)
fig.text(0.5, -0.01, "Green = poor kids do better here than the model expects given local conditions. "
         "Counties under 50,000 people excluded (noisy estimates).",
         ha="center", fontsize=8.5, color=MUT)
fig.tight_layout()
fig.savefig(f"{OUT}/7_resilience.png", bbox_inches="tight")
print("saved docs/figures/7_resilience.png")

# --- 8. bias probe: what happens if the demographic features come out? -------
DROP = ["EP_MINRTY", "EP_LIMENG"]
KEPT = [c for c in COLS if c not in DROP]
Xa = StandardScaler().fit_transform(df[KEPT])
kma = KMeans(n_clusters=K, n_init=10, random_state=42).fit(Xa)
cent_a = pd.DataFrame(kma.cluster_centers_, columns=[FEATS[c] for c in KEPT])
order_a = cent_a["Upward mobility"].sort_values().index.tolist()
lab_a = pd.Series(kma.labels_).map({o: i for i, o in enumerate(order_a)}).values
cent_a = cent_a.iloc[order_a].reset_index(drop=True)
ari_abl = adjusted_rand_score(df["cluster"], lab_a)
sil_full, sil_abl = silhouette_score(X, km.labels_), silhouette_score(Xa, kma.labels_)

print("\n=== bias probe: refit without Minority + Limited English ===")
print(f"silhouette  full {sil_full:.3f}  ->  ablated {sil_abl:.3f}   (change {sil_abl-sil_full:+.3f})")
print(f"adjusted Rand index vs original labels: {ari_abl:.3f}")
print(f"counties keeping the same type: {(df['cluster'].values == lab_a).mean()*100:.1f}%")

shared = [FEATS[c] for c in KEPT]
# two panels share this width, so each column is about half as wide as it is in
# figure 3 and the full-length names run into each other
TIGHT = [f"Type {i+1}\n" + textwrap.fill(NAMES[i], 11) for i in range(K)]
fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
panels = [(cent[shared], f"All 17 inputs\nsilhouette {sil_full:.3f}", TIGHT),
          (cent_a[shared], f"Minority + Limited English removed\nsilhouette {sil_abl:.3f}",
           [f"Type {i+1}\n(regrouped)" for i in range(K)])]
for a, (cent_x, ttl, xlab) in zip(axes, panels):
    M = cent_x.T.values
    im = a.imshow(M, cmap=DIVERGE, vmin=-1.6, vmax=1.6, aspect="auto")
    a.set_xticks(range(K))
    a.set_xticklabels(xlab, fontsize=9)
    a.set_title(ttl, fontsize=11.5, weight="bold")
    a.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=7.5,
                   color="white" if abs(M[i, j]) > 1.0 else INK)
axes[0].set_yticks(range(len(shared)))
axes[0].set_yticklabels(shared, fontsize=9)
fig.suptitle("Does the structure survive without the demographic features?",
             fontsize=14, weight="bold")
fig.colorbar(im, ax=axes, shrink=0.55, label="above / below US average")
fig.text(0.5, 0.015, textwrap.fill(
         f"Dropping the two demographic inputs changes separation quality by "
         f"{sil_abl-sil_full:+.3f} and {(df['cluster'].values == lab_a).mean()*100:.0f}% of counties keep "
         f"their type (adjusted Rand index {ari_abl:.2f}). The four-way structure survives, so it was "
         "mostly tracking material conditions. What the two features bought was not separation. "
         "It was the name that got written on the group. Note the panels are numbered "
         "independently, so the gateways group moves to Type 2 on the right.", 150),
         ha="center", va="top", fontsize=9, color=MUT)
fig.savefig(f"{OUT}/8_bias_probe.png", bbox_inches="tight")
print("saved docs/figures/8_bias_probe.png")

# rounded so the file is byte-stable between runs. n_jobs=-1 sums trees in
# whatever order they finish, which wobbles pred at ~1e-16 and churns the diff.
out = df[["FIPS", "COUNTY", "ST_ABBR", "CODE2023", "E_TOTPOP", "RPL_THEMES", "MOBILITY",
          "pred", "residual", "cluster"]].round({"pred": 6, "residual": 6})
out.to_csv(f"{OUT}/combined_clusters.csv", index=False)
print("saved docs/figures/combined_clusters.csv (now with predictions + residuals)")
