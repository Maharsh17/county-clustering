"""County social-mobility clustering: 16 SVI vulnerability measures plus upward mobility,
urban-rural as context. Run: ./.venv/bin/python analysis.py
"""
import json
from urllib.request import urlopen
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd, numpy as np, os
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
INK, MUT = "#0b0b0b", "#8a8a86"
DIVERGE = LinearSegmentedColormap.from_list("bwr", ["#256abf", "#f4f4f2", "#e34948"])
CLUSTER_COLORS = ["#2a78d6", "#1baf7a", "#e34948", "#eda100", "#4a3aa7"]
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
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#cccccc")
fig.suptitle("Same counties, clustered at K = 2 to 7  (PCA projection)", fontsize=14, weight="bold")
fig.text(0.5, 0.005, f"Axes are the first two principal components "
         f"({ev[0]*100:.0f}% + {ev[1]*100:.0f}% of variance). "
         "Higher silhouette = cleaner separation.", ha="center", fontsize=9, color=MUT)
fig.tight_layout(rect=[0, 0.02, 1, 0.97])
fig.savefig(f"{OUT}/1_k_sweep_grid.png", bbox_inches="tight")
print("saved figures/1_k_sweep_grid.png")
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
    a.set_title(title, fontsize=11, weight="bold"); a.set_xlabel("number of clusters (K)")
    a.grid(color="#eee"); a.set_axisbelow(True)
fig.suptitle("How many clusters? Four tests on the combined dataset", fontsize=14, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/2_optimal_k_tests.png", bbox_inches="tight")
print("saved figures/2_optimal_k_tests.png")

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
    print(f"\nCluster {c}:  HIGH -> " + ", ".join(f"{n} (+{v:.1f})" for n, v in hi.items()))
    print(f"           LOW  -> " + ", ".join(f"{n} ({v:.1f})" for n, v in lo.items()))

print("\n=== 4 biggest counties in each cluster ===")
for c in range(K):
    d = df[df.cluster == c].nlargest(4, "E_TOTPOP")
    names = ", ".join(f"{r.COUNTY} {r.ST_ABBR}" for _, r in d.iterrows())
    print(f"Cluster {c}: {names}")

fig, ax = plt.subplots(figsize=(8.6, 6))
for c in range(K):
    d = df[df.cluster == c]
    ax.scatter(d.RPL_THEMES, d.MOBILITY, s=12, alpha=0.5,
               color=CLUSTER_COLORS[c], label=f"Type {c+1}", edgecolors="none")
ax.set_xlabel("Social vulnerability  (SVI percentile, higher = more vulnerable)")
ax.set_ylabel("Upward mobility  (adult income rank of poor kids)")
ax.set_title("Social-mobility types of US counties", fontsize=14, weight="bold", pad=10)
ax.legend(title="County type", frameon=False, markerscale=1.6, fontsize=9)
ax.text(0.02, 0.03, "Bottom-right = high vulnerability, low mobility (traps)\n"
        "Top-right = vulnerable but still lifts kids (resilient)",
        transform=ax.transAxes, fontsize=8.5, color=MUT, va="bottom")
fig.tight_layout(); fig.savefig(f"{OUT}/3_mobility_vs_vulnerability.png", bbox_inches="tight")
print("\nsaved figures/3_mobility_vs_vulnerability.png")

fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.barh(range(K), prof.mobility.values, color=[CLUSTER_COLORS[c] for c in range(K)], edgecolor="white")
for i, (mob, v, ur) in enumerate(zip(prof.mobility, prof.vuln, prof.urban_rural)):
    ax.text(mob + 0.005, i, f"mobility {mob:.2f} | vuln {v:.2f} | urban-rural {ur:.1f}",
            va="center", fontsize=9, color=INK)
ax.set_yticks(range(K)); ax.set_yticklabels([f"Type {c+1}" for c in range(K)])
ax.set_xlabel("Mean upward mobility"); ax.set_xlim(0, 0.6)
ax.set_title("Four county types, low to high mobility", fontsize=13, weight="bold", pad=10)
ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/4_cluster_profiles.png", bbox_inches="tight")
print("saved figures/4_cluster_profiles.png")

fig, ax = plt.subplots(figsize=(9.5, 7))
M = cent.T.values
im = ax.imshow(M, cmap=DIVERGE, vmin=-1.6, vmax=1.6, aspect="auto")
ax.grid(False)
ax.set_xticks(range(K)); ax.set_xticklabels([f"Cluster {c}" for c in range(K)])
ax.set_yticks(range(len(COLS))); ax.set_yticklabels([FEATS[c] for c in COLS])
ax.axhline(15.5, color="#555", lw=1.2, ls="--")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 1.0 else INK)
ax.set_title("What defines each cluster (z-score vs national average)", fontsize=13, weight="bold", pad=10)
cb = fig.colorbar(im, ax=ax, shrink=0.6); cb.set_label("above / below US average", fontsize=9)
fig.text(0.5, -0.01, "Dashed line separates the mobility outcome from the 16 vulnerability inputs",
         ha="center", fontsize=8.5, color=MUT)
fig.tight_layout(); fig.savefig(f"{OUT}/5_combined_cluster_profiles.png", bbox_inches="tight")
print("saved figures/5_combined_cluster_profiles.png")

df[["FIPS", "COUNTY", "ST_ABBR", "CODE2023", "RPL_THEMES", "MOBILITY", "cluster"]].to_csv(
    f"{OUT}/combined_clusters.csv", index=False)
print("saved figures/combined_clusters.csv")

NAMES = {0: "Rural hardship", 1: "Costly big metros", 2: "Immigrant gateways", 3: "Comfortable America"}
MAP_COLORS = {"Rural hardship": "#2a78d6", "Costly big metros": "#1baf7a",
              "Immigrant gateways": "#e34948", "Comfortable America": "#eda100"}
ORDER = ["Rural hardship", "Costly big metros", "Immigrant gateways", "Comfortable America"]

map_df = df[["FIPS", "cluster"]].copy()
map_df["FIPS"] = map_df["FIPS"].str.zfill(5)
map_df["Type"] = map_df["cluster"].map(NAMES)
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
fig.write_html(f"{OUT}/6_cluster_map.html")
fig.write_image(f"{OUT}/6_cluster_map.png", width=1100, height=700, scale=2)
print("saved figures/6_cluster_map.html and .png")
