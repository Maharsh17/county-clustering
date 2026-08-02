"""County social-mobility explorer. Run: ./.venv/bin/streamlit run app.py

One map, controls down the left, the selected county down the right. Every
checkbox refits the model, so the map redraws to show what the model looks like
when it is allowed to see less. Click any county and its five nearest matches
light up, usually a few hundred miles away.
"""
import json
from urllib.request import urlopen
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="What kind of place is your county?",
                   page_icon="🗺️", layout="wide")

FEATS = {
    "EP_POV150": "Poverty (<150%)", "EP_UNEMP": "Unemployment", "EP_HBURD": "Housing cost burden",
    "EP_NOHSDP": "No HS diploma", "EP_UNINSUR": "Uninsured", "EP_AGE65": "Aged 65+",
    "EP_AGE17": "Aged 17 & under", "EP_DISABL": "Disability", "EP_SNGPNT": "Single-parent",
    "EP_LIMENG": "Limited English", "EP_MINRTY": "Minority", "EP_MUNIT": "Multi-unit housing",
    "EP_MOBILE": "Mobile homes", "EP_CROWD": "Crowded housing", "EP_NOVEH": "No vehicle",
    "EP_GROUPQ": "Group quarters", "MOBILITY": "Upward mobility",
}
COLS = list(FEATS)
PAL = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#4a3aa7", "#008300", "#e87ba4"]
GEOJSON = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
BASELINE_SIL = 0.155   # all 17 measures at K=4, from analysis.py


@st.cache_data
def load():
    df = pd.read_csv("data/county_svi_mobility.csv", dtype={"FIPS": str}).replace(-999, np.nan)
    df = df.dropna(subset=COLS + ["CODE2023", "E_TOTPOP"]).reset_index(drop=True)
    df["FIPS"] = df["FIPS"].str.zfill(5)
    df["name"] = df["COUNTY"] + ", " + df["ST_ABBR"]
    # residuals come from analysis.py (cross-validated random forest) if it has been run
    try:
        r = pd.read_csv("figures/combined_clusters.csv", dtype={"FIPS": str})
        r["FIPS"] = r["FIPS"].str.zfill(5)
        df = df.merge(r[["FIPS", "pred", "residual"]], on="FIPS", how="left")
    except (FileNotFoundError, KeyError):
        df["pred"] = df["residual"] = np.nan
    return df


@st.cache_data
def load_geojson():
    with urlopen(GEOJSON) as resp:
        return json.load(resp)


@st.cache_data
def centroids():
    """Rough lon/lat per FIPS, averaged over the county outline.

    ponytail: good enough to drop a marker on. A real centroid needs shapely and
    an area weighting, which buys nothing at this zoom level.
    """
    out = {}
    for f in load_geojson()["features"]:
        g = f["geometry"]
        rings = ([g["coordinates"][0]] if g["type"] == "Polygon"
                 else [p[0] for p in g["coordinates"]])
        pts = np.concatenate([np.asarray(r, dtype=float)[:, :2] for r in rings])
        out[f["id"]] = pts.mean(axis=0)
    return out


# ponytail: same 10 lines as analysis.py. A shared module would mean importing a
# script that runs a full analysis on import, so the duplication is the cheap side.
@st.cache_data
def fit(cols: tuple, k: int):
    """Cluster on `cols`, relabelled so type 0 always has the lowest mean mobility."""
    cols = list(cols)
    X = StandardScaler().fit_transform(load()[cols])
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    cent = pd.DataFrame(km.cluster_centers_, columns=[FEATS[c] for c in cols])
    key = "Upward mobility" if "MOBILITY" in cols else cent.columns[0]
    order = cent[key].sort_values().index.tolist()
    labels = pd.Series(km.labels_).map({o: i for i, o in enumerate(order)}).values
    return labels, cent.iloc[order].reset_index(drop=True), silhouette_score(X, km.labels_), X


def nearest(X: np.ndarray, i: int, n: int = 5) -> np.ndarray:
    """Row indices of the n counties closest to row i.

    Every feature counts equally, which is what StandardScaler already implies:
    a county's minority share weighs the same as its unemployment rate here.
    That is a choice, not a law. See the bias probe in the README.
    """
    d = np.linalg.norm(X - X[i], axis=1)
    return np.argsort(d)[1:n + 1]


def miles(a, b) -> float:
    """Great-circle distance between two (lon, lat) pairs."""
    lo1, la1, lo2, la2 = np.radians([a[0], a[1], b[0], b[1]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(3958.8 * 2 * np.arcsin(np.sqrt(h)))


def describe(row: pd.Series) -> str:
    """Plain-English read on a type, built from whichever z-scores are extreme."""
    hi = row[row >= 0.6].sort_values(ascending=False)
    lo = row[row <= -0.6].sort_values()
    parts = []
    if len(hi):
        parts.append("Unusually high " + ", ".join(hi.index[:3]))       # keep proper nouns
    if len(lo):
        parts.append("unusually low " + ", ".join(lo.index[:2]))
    if not parts:
        return "Close to the national average on everything. This is the group that "\
               "exists because the others do."
    return " and ".join(parts) + "."


def build_map(df, K, sel_idx, nb_idx):
    m = df[["FIPS", "name"]].copy()
    m["Type"] = "Type " + (df["type"] + 1).astype(str)
    fig = px.choropleth(
        m, geojson=load_geojson(), locations="FIPS", color="Type",
        color_discrete_sequence=PAL[:K], hover_name="name",
        category_orders={"Type": [f"Type {i+1}" for i in range(K)]}, scope="usa")
    fig.update_traces(marker_line_width=0, hovertemplate="%{hovertext}<extra></extra>")

    if sel_idx is not None:
        cen = centroids()
        pts = [(df.at[j, "FIPS"], df.at[j, "name"]) for j in nb_idx]
        xy = [cen.get(f) for f, _ in pts]
        keep = [(p, c) for p, c in zip(pts, xy) if c is not None]
        if keep:
            fig.add_trace(go.Scattergeo(
                lon=[c[0] for _, c in keep], lat=[c[1] for _, c in keep],
                text=[p[1] for p, _ in keep], mode="markers", name="most similar",
                marker=dict(size=13, color="white", line=dict(width=2.5, color="#0b0b0b")),
                hovertemplate="most similar: %{text}<extra></extra>"))
        s = cen.get(df.at[sel_idx, "FIPS"])
        if s is not None:
            fig.add_trace(go.Scattergeo(
                lon=[s[0]], lat=[s[1]], text=[df.at[sel_idx, "name"]], mode="markers",
                name="selected",
                marker=dict(size=19, color="#0b0b0b", symbol="star",
                            line=dict(width=2, color="white")),
                hovertemplate="%{text}<extra></extra>"))
            # true overlay: annotations are drawn inside the figure, on top of the map
            far = [miles(s, c) for _, c in keep]
            avg = int(np.mean(far)) if far else 0
            fig.add_annotation(
                x=0.01, y=0.02, xref="paper", yref="paper", showarrow=False,
                align="left", xanchor="left", bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#cccccc", borderwidth=1, borderpad=9, font=dict(size=12),
                text=(f"<b>{df.at[sel_idx, 'name']}</b><br>"
                      f"★ selected &nbsp;&nbsp; ○ its {len(keep)} closest matches<br>"
                      f"<span style='color:#8a8a86'>closest in <i>profile</i>, and an average of "
                      f"<b>{avg:,} miles</b> away</span>"))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=610,
                      legend=dict(orientation="h", y=1.03, x=0, title=None),
                      geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="white"))
    return fig


def main():
    df = load()

    # ---- left rail: what the model is allowed to see -----------------------
    st.sidebar.title("What the model can see")
    st.sidebar.caption("Uncheck a measure and the map refits without it.")
    chosen = [c for c in COLS if c != "MOBILITY"
              and st.sidebar.checkbox(FEATS[c], value=True, key=f"f_{c}")]
    include_mob = st.sidebar.checkbox("**Upward mobility** (the outcome)", value=True)
    st.sidebar.divider()
    K = st.sidebar.slider("How many types?", 2, 7, 4)
    if st.sidebar.button("Reset to all 17 measures", use_container_width=True):
        for c in COLS:
            st.session_state[f"f_{c}"] = True
        st.rerun()

    active = chosen + (["MOBILITY"] if include_mob else [])
    if len(active) < 2:
        st.error("Keep at least two measures checked.")
        st.stop()

    labels, cent, sil, X = fit(tuple(active), K)
    df["type"] = labels
    dropped = [FEATS[c] for c in COLS if c not in active]

    # ---- selection, from a map click or the picker -------------------------
    names = sorted(df.name)
    if "sel" not in st.session_state:
        st.session_state.sel = ("Champaign County, IL" if "Champaign County, IL" in names
                                else names[0])
    # a click is applied once; after that the dropdown wins, otherwise the last
    # click would keep overriding every manual selection on each rerun
    ev = st.session_state.get("usmap")
    if ev and ev.get("selection", {}).get("points"):
        loc = ev["selection"]["points"][0].get("location")
        if loc and loc != st.session_state.get("_clicked"):
            st.session_state._clicked = loc
            hit = df.index[df.FIPS == loc]
            if len(hit):
                st.session_state.sel = df.at[hit[0], "name"]

    # ---- top strip: the model's current state ------------------------------
    st.title("What kind of place is your county?")
    a, b, c, d = st.columns(4)
    a.metric("Counties", f"{len(df):,}")
    b.metric("Types", K)
    c.metric("Measures in use", f"{len(active)} of {len(COLS)}")
    c_sil = f"{sil - BASELINE_SIL:+.3f} vs full model" if dropped else None
    d.metric("Silhouette", f"{sil:.3f}", c_sil,
             help="How cleanly separated the groups are. Higher is better.")
    if dropped:
        st.warning(f"**Refitted without {', '.join(dropped)}.** Separation changed by "
                   f"{sil - BASELINE_SIL:+.3f}. If that number is near zero, the measures "
                   f"you removed were not doing the work of separating these groups.")

    # ---- hero map + right rail ---------------------------------------------
    left, right = st.columns([3.4, 1.45], gap="medium")

    pick = st.session_state.sel
    i = int(df.index[df.name == pick][0])
    nb = nearest(X, i)

    with left:
        st.plotly_chart(build_map(df, K, i, nb), use_container_width=True,
                        on_select="rerun", selection_mode="points", key="usmap")
        st.caption("Click any county to select it. Colours are types, numbered low to high "
                   "mean mobility, so Type 1 is always the lowest no matter what you uncheck.")

    with right:
        row = df.loc[i]
        st.selectbox("Selected county", names, key="sel")   # value lives in session_state
        st.markdown(f"### Type {int(row.type) + 1}")
        st.caption(describe(cent.iloc[int(row.type)]))

        m1, m2 = st.columns(2)
        m1.metric("Mobility", f"{row.MOBILITY:.3f}",
                  help="Mean adult income rank of kids raised in the bottom 25%.")
        m2.metric("Vulnerability", f"{row.RPL_THEMES:.2f}")
        m1.metric("Population", f"{int(row.E_TOTPOP):,}")
        if pd.notna(row.residual):
            m2.metric("vs predicted", f"{row.residual:+.3f}",
                      help="Actual minus what the random forest predicted. Positive is better "
                           "than its conditions suggest.")

        st.markdown("**Against the national average**")
        z = pd.Series(X[i], index=[FEATS[c] for c in active]).sort_values()
        bar = go.Figure(go.Bar(x=z.values, y=z.index, orientation="h",
                               marker_color=["#e34948" if v > 0 else "#2a78d6" for v in z.values]))
        bar.update_layout(height=19 * len(z) + 60, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis_title=None, yaxis=dict(tickfont=dict(size=10)),
                          showlegend=False)
        st.plotly_chart(bar, use_container_width=True)

        st.markdown("**Most similar counties**")
        for j in nb:
            st.markdown(f"○ **{df.at[j, 'name']}** · Type {int(df.at[j, 'type']) + 1} "
                        f"· mobility {df.at[j, 'MOBILITY']:.3f}")
        st.caption(f"Similarity uses all {len(active)} active measures at once. Geography is "
                   "not one of them.")

    # ---- bottom: the types themselves --------------------------------------
    st.divider()
    st.subheader("What defines each type")
    st.caption("Every measure, every type, as standard deviations from the national average. "
               "Red is above, blue is below. This grid is the entire model in one picture.")

    hm, key = st.columns([3.1, 1.75], gap="medium")
    with hm:
        order = [FEATS[c] for c in active if c != "MOBILITY"]
        if "MOBILITY" in active:
            order += [FEATS["MOBILITY"]]        # outcome pinned to the bottom row
        grid = cent[order].T
        heat = px.imshow(grid, color_continuous_scale=["#256abf", "#f4f4f2", "#e34948"],
                         zmin=-1.6, zmax=1.6, aspect="auto", text_auto=".1f",
                         labels=dict(color="z-score"))
        heat.update_xaxes(side="top", tickvals=list(range(K)),
                          ticktext=[f"<b>Type {i+1}</b>" for i in range(K)])
        heat.update_traces(textfont_size=11, xgap=2, ygap=2,
                           hovertemplate="%{y}<br>%{x}: %{z:+.2f}<extra></extra>")
        if "MOBILITY" in active:                # separate the outcome from the inputs
            heat.add_hline(y=len(order) - 1.5, line=dict(color="#555", width=1.4, dash="dash"))
        heat.update_layout(height=30 * len(order) + 150, margin=dict(l=0, r=0, t=40, b=0),
                           coloraxis_colorbar=dict(thickness=12, len=0.6))
        st.plotly_chart(heat, use_container_width=True)
        if "MOBILITY" in active:
            st.caption("The dashed line separates the outcome I am trying to understand "
                       "from the 16 vulnerability measures fed in as inputs.")

    with key:
        for t in range(K):
            sub = df[df.type == t]
            st.markdown(
                f"<div style='border-left:7px solid {PAL[t]};padding:2px 0 2px 11px;"
                f"margin-bottom:6px'><b>Type {t+1}</b> &nbsp;"
                f"<span style='color:#8a8a86'>{len(sub):,} counties · mobility "
                f"{sub.MOBILITY.mean():.3f}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.87rem;margin:-2px 0 4px 18px'>"
                        f"{describe(cent.iloc[t])}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.8rem;color:#8a8a86;margin:0 0 14px 18px'>"
                        f"e.g. {', '.join(sub.nlargest(3, 'E_TOTPOP').name)}</div>",
                        unsafe_allow_html=True)
        st.caption("These groups have no names in the data. Any name is something a human "
                   "adds, and that is where bias gets in. See the README.")

    # ---- below the fold ------------------------------------------------------
    with st.expander("Which counties beat what their conditions predict?"):
        if df.residual.isna().all():
            st.warning("Run `python analysis.py` first to generate model predictions.")
        else:
            floor = st.slider("Minimum population", 0, 500_000, 50_000, step=10_000,
                              help="Mobility estimates for very small counties are noisy.")
            big = df[df.E_TOTPOP >= floor].dropna(subset=["residual"])
            cols_ = {"name": "County", "MOBILITY": "Actual", "pred": "Predicted",
                     "residual": "Gap"}
            l, r = st.columns(2)
            l.markdown("**Beating their profile**")
            l.dataframe(big.nlargest(12, "residual")[list(cols_)].rename(columns=cols_),
                        hide_index=True, use_container_width=True)
            r.markdown("**Falling short**")
            r.dataframe(big.nsmallest(12, "residual")[list(cols_)].rename(columns=cols_),
                        hide_index=True, use_container_width=True)

    with st.expander("Compare two counties"):
        p, q = st.columns(2)
        n1 = p.selectbox("County A", names, index=0, key="cA")
        n2 = q.selectbox("County B", names, index=1, key="cB")
        i1, i2 = int(df.index[df.name == n1][0]), int(df.index[df.name == n2][0])
        diff = pd.Series(X[i1] - X[i2], index=[FEATS[c] for c in active]).sort_values()
        f = go.Figure(go.Bar(x=diff.values, y=diff.index, orientation="h",
                             marker_color=["#e34948" if v > 0 else "#2a78d6" for v in diff.values]))
        f.update_layout(height=24 * len(diff) + 120, margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title=f"← more in {n2}     |     more in {n1} →")
        st.plotly_chart(f, use_container_width=True)

    st.caption("Data: CDC/ATSDR Social Vulnerability Index and county upward-mobility "
               "estimates. Types describe counties, not the people who live in them. "
               "See the README for sources, limitations, and how the bias probe works.")


if __name__ == "__main__":
    main()
