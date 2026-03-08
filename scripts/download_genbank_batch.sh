#!/usr/bin/env bash
# Download GenBank records for accessions listed in the first column of a CSV.
#
# Assumptions:
# - CSV always has a header (first row will be skipped).
# - No empty rows.
# - First column contains valid accession IDs (e.g., NC_001623.1).
#
# Usage:
#   ./download_genbank_batch.sh input.csv output_dir [rettype]
#
# Examples:
#   ./download_genbank_batch.sh accessions.csv genbank_files
#   ./download_genbank_batch.sh accessions.csv genbank_files gbwithparts
#
# Optional environment variables:
#   export NCBI_EMAIL="your.name@domain.com"   # Recommended: identify your requests
#   export NCBI_API_KEY="..."                  # Optional: increases E-utilities rate limit
#   export SLEEP="0.10"                        # Optional: delay between requests
#
# Notes:
# - Default rettype: gb  (use "gbwithparts" for richer content)
# - Skips already-downloaded files (<ACCESSION>.gb is non-empty)
# - Basic network/content sanity check: requires 'LOCUS' in the first line

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input.csv> <output_dir> [rettype]" >&2
  exit 1
fi

CSV="$1"
OUT_DIR="$2"
RETTYPE="${3:-gb}"

EMAIL="${NCBI_EMAIL:-your_email@example.com}"
API_KEY="${NCBI_API_KEY:-}"
TOOL="genbank-batch-fetch"

[[ -f "$CSV" ]] || { echo "Error: CSV not found: $CSV" >&2; exit 1; }
mkdir -p "$OUT_DIR"

# Polite rate limiting defaults
if [[ -n "$API_KEY" ]]; then
  SLEEP="${SLEEP:-0.12}"   # ~8–10 req/s with key
else
  SLEEP="${SLEEP:-0.34}"   # ~3 req/s without key
fi

fetch_one() {
  local acc="$1"
  local out="$OUT_DIR/${acc}.gb"

  # Skip if already downloaded (non-empty)
  if [[ -s "$out" ]]; then
    echo "Skip (exists): $acc"
    return
  fi

  local tmp
  tmp="$(mktemp)"

  local url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
  local -a params
  params=( --get "$url"
           --data-urlencode "db=nuccore"
           --data-urlencode "id=${acc}"
           --data-urlencode "rettype=${RETTYPE}"
           --data-urlencode "retmode=text" )
  if [[ -n "$API_KEY" ]]; then
    params+=( --data-urlencode "api_key=${API_KEY}" )
  fi

  if ! curl -sSLo "$tmp" -H "User-Agent: ${TOOL} (mailto:${EMAIL})" --fail "${params[@]}"; then
    echo "Error: curl failed for $acc" >&2
    rm -f "$tmp"
    return
  fi

  # GenBank files start with a LOCUS line; this catches rare HTML/text error pages.
  if ! head -n 1 "$tmp" | grep -q '^LOCUS'; then
    echo "Error: Response for $acc doesn't look like GenBank (no LOCUS); kept at: $tmp" >&2
    return
  fi

  mv "$tmp" "$out"
  echo "Saved: $out"
}

echo "Input:    $CSV"
echo "Output:   $OUT_DIR"
echo "rettype:  $RETTYPE"
echo "Email:    $EMAIL"
[[ -n "$API_KEY" ]] && echo "API Key:  (set)" || echo "API Key:  (not set)"
echo "Sleep(s): $SLEEP"
echo "----------------------------------------"

# Read CSV, skip header, take first column only.
# Assumes no empty lines and valid accessions in col 1.
# Trims quotes/whitespace and CR (in case of CRLF files).
tail -n +2 "$CSV" | while IFS=, read -r acc _; do
  acc="${acc//$'\r'/}"         # strip CR if present
  acc="${acc%\"}"; acc="${acc#\"}"  # strip surrounding quotes
  acc="${acc//[[:space:]]/}"   # remove whitespace

  fetch_one "$acc"
  sleep "$SLEEP"
done

echo "Done."