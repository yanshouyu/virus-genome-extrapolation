#!/usr/bin/env bash
# data/raw/baculovirus/run_download.sh
# Recreate baculovirus GenBank dataset from the manifest CSV.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." &>/dev/null && pwd)"

CSV="$SCRIPT_DIR/Baculoviridae.csv"
OUT_DIR="$SCRIPT_DIR/Baculoviridae"
BATCH="$REPO_ROOT/scripts/download_genbank_batch.sh"
RETTYPE="${RETTYPE:-gb}"

command -v curl >/dev/null 2>&1 || { echo "Error: curl is required."; exit 1; }
[[ -f "$CSV" ]] || { echo "Error: CSV not found: $CSV"; exit 1; }
[[ -f "$BATCH" ]] || { echo "Error: batch script not found: $BATCH"; exit 1; }

mkdir -p "$OUT_DIR"

echo "== Baculoviridae GenBank download =="
echo "CSV:        $CSV"
echo "Output dir: $OUT_DIR"
echo "Batch:      $BATCH"
echo "rettype:    $RETTYPE"
echo "-------------------------------------"

# No email/API key passed; batch script will run at default rate limits.
bash "$BATCH" "$CSV" "$OUT_DIR" "$RETTYPE"

echo "Done. Files in: $OUT_DIR"