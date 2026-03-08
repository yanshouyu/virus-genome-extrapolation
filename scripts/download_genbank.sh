#!/usr/bin/env bash
# Download a GenBank record by accession from NCBI E-utilities.
# Usage:
#   ./download_genbank.sh [ACCESSION] [OUTPUT_FILE]
# Examples:
#   ./download_genbank.sh                  # downloads NC_001623.1 to NC_001623.1.gb
#   ./download_genbank.sh NC_001623.1      # same as above
#   ./download_genbank.sh NC_001623.1 myfile.gb
#
# Tip: Set your email so NCBI can contact you if needed:
#   export NCBI_EMAIL="your.name@domain.com"

set -euo pipefail

ACCESSION="${1:-NC_001623.1}"
OUT="${2:-${ACCESSION}.gb}"
EMAIL="${NCBI_EMAIL:-your_email@example.com}"    # set NCBI_EMAIL env var to override
TOOL="genbank-fetch-bash"

BASE_URL="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Download to a temp file first; move into place only on success
TMP_OUT="$(mktemp "${OUT}.XXXX.tmp")"
trap 'rm -f "$TMP_OUT"' EXIT

curl -sSLo "$TMP_OUT" \
  -H "User-Agent: ${TOOL} (mailto:${EMAIL})" \
  --fail \
  --get "$BASE_URL" \
  --data-urlencode "db=nuccore" \
  --data-urlencode "id=${ACCESSION}" \
  --data-urlencode "rettype=gb" \
  --data-urlencode "retmode=text"

# Basic sanity check: GenBank files start with a LOCUS line
if ! head -n 1 "$TMP_OUT" | grep -q '^LOCUS'; then
  echo "Error: The downloaded file doesn't look like a GenBank record (no LOCUS line)."
  echo "File saved at: $TMP_OUT for inspection."
  exit 1
fi

mv "$TMP_OUT" "$OUT"
trap - EXIT
echo "✅ Downloaded ${ACCESSION} → ${OUT}"