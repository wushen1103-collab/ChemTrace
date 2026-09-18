#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p external

if [ ! -d external/tautobase/.git ]; then
  git clone https://github.com/WahlOya/Tautobase.git external/tautobase
else
  git -C external/tautobase pull --ff-only
fi

npm install openchemlib \
  --prefix external/tautobase_tools \
  --registry="${NPM_REGISTRY:-https://registry.npmmirror.com}"

cat <<'MSG'
Reviewer external sources are ready.

Expected files:
- external/tautobase/Tautobase.dwar
- external/tautobase_tools/node_modules/openchemlib

ChEMBL36 SQLite is intentionally not downloaded here. Supply its location
with --chembl-db when rebuilding the independent relation-gold tables.
MSG
