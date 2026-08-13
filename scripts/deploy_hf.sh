#!/usr/bin/env bash
# Push the explorer to a Hugging Face Space.
#
# Streamlit Community Cloud rebuilds its container from requirements.txt every
# time a sleeping app wakes, which for this project means reinstalling roughly
# 570 MB of scientific Python. Hugging Face caches a built image instead, so a
# wake starts a container rather than a pip install.
#
# Usage:
#   export HF_TOKEN=hf_...                       # from huggingface.co/settings/tokens, write scope
#   ./scripts/deploy_hf.sh <user>/<space-name>
#
# The Space must already exist. Create it at huggingface.co/new-space with SDK
# "Streamlit". Everything else, including the app_file path, is set from here.
set -euo pipefail

SPACE="${1:-}"
[ -n "$SPACE" ] || { echo "usage: $0 <user>/<space-name>" >&2; exit 1; }
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is not set" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> cloning space $SPACE"
git clone -q "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE}" "$WORK/space"
cd "$WORK/space"

echo "==> syncing the runtime files"
# only what the app needs at runtime. The pipeline, the tests and the published
# site stay on GitHub, which keeps the Space image small and the build fast.
for p in app src config .streamlit LICENSE requirements.txt; do
  rm -rf "./${p}"
  cp -R "${ROOT}/${p}" "./${p}"
done
mkdir -p data outputs
cp "${ROOT}/data/county_svi_mobility.csv" data/
cp "${ROOT}/outputs/combined_clusters.csv" outputs/
find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# The Space config lives in this front matter. app_file has to point at the
# real entry point, which is not the default app.py.
cat > README.md <<'MD'
---
title: What Kind of Place Is Your County?
emoji: 🗺️
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app/app.py
pinned: false
license: agpl-3.0
short_description: Clustering 3,128 US counties by vulnerability and upward mobility
---

# What Kind of Place Is Your County?

An interactive explorer for four types of US county, built from 16 CDC social
vulnerability measures plus Opportunity Atlas upward mobility.

Uncheck any measure in the sidebar and the model refits in front of you. Turning
off Minority and Limited English reproduces the project's bias probe live: the
groups barely move, which is the whole argument.

- **Full write up:** https://maharsh17.github.io/county-clustering/
- **Source and data documentation:** https://github.com/Maharsh17/county-clustering

Licensed under AGPL-3.0. Section 13 covers network use, so if you host a modified
version you owe your users the source.
MD

if git diff --quiet && git diff --cached --quiet && [ -z "$(git status --porcelain)" ]; then
  echo "==> nothing changed, space is already current"
  exit 0
fi

git add -A
git -c user.email="deploy@localhost" -c user.name="deploy" \
    commit -q -m "Sync from github.com/Maharsh17/county-clustering"
echo "==> pushing"
git push -q origin main
echo "==> done: https://huggingface.co/spaces/${SPACE}"
