"""Smoke test for the pieces of app.py that can actually be wrong.

Run: ./.venv/bin/python -m tests.test_app
Streamlit widget wiring fails loudly in the browser. The clustering, the
mobility-ordered relabelling and the nearest-neighbour search fail quietly.
Those are what this checks.
"""
import numpy as np
from app.app import COLS, build_map, centroids, fit, load, nearest

df = load()
assert len(df) == 3128, f"expected 3128 counties, got {len(df)}"
assert df.FIPS.str.len().eq(5).all(), "FIPS codes must be zero-padded to 5 chars"

labels, cent, sil, X = fit(tuple(COLS), 4)
assert set(labels) == {0, 1, 2, 3}, f"expected 4 types, got {set(labels)}"
assert X.shape == (3128, 17), f"unexpected feature matrix {X.shape}"

# the whole app depends on type 0 being the lowest-mobility group at every K,
# otherwise colours and names shuffle every time a checkbox moves
means = [df.MOBILITY[labels == t].mean() for t in range(4)]
assert means == sorted(means), f"types not ordered by mobility: {means}"
for k in (2, 5, 7):
    lab, _, _, _ = fit(tuple(COLS), k)
    m = [df.MOBILITY[lab == t].mean() for t in range(k)]
    assert m == sorted(m), f"K={k} types not ordered by mobility: {m}"

# dropping features must change the grouping, or the bias probe proves nothing
kept = tuple(c for c in COLS if c not in ("EP_MINRTY", "EP_LIMENG"))
lab_a, _, _, Xa = fit(kept, 4)
assert Xa.shape[1] == 15, "ablated fit kept the wrong number of columns"
assert (lab_a != labels).sum() > 0, "ablation changed nothing, features are not wired through"

# nearest returns distinct other counties, never the query itself
i = int(df.index[df.name == "Champaign County, IL"][0])
nb = nearest(X, i)
assert len(set(nb)) == 5 and i not in nb, f"bad neighbour set {nb}"

# the map's highlight layer needs a coordinate per county. A few missing is fine,
# a lot missing means the geojson ids stopped matching my FIPS
cen = centroids()
covered = sum(1 for f in df.FIPS if f in cen)
assert covered > 0.99 * len(df), f"only {covered}/{len(df)} counties have a centroid"

fig = build_map(df.assign(type=labels), 4, i, nb)
assert len(fig.data) == 4 + 2, f"expected 4 type traces + selected + neighbours, got {len(fig.data)}"
assert len(fig.layout.annotations) == 1, "the on-map overlay card is missing"

print(f"ok, {len(df)} counties, silhouette {sil:.3f}, {covered} centroids, "
      f"Champaign's nearest: {', '.join(df.name.iloc[nb])}")
