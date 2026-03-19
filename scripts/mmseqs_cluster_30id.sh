#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF' >&2
Usage:   $0 [OPTIONS] <input.fasta>

Run MMseqs2 Linclust at 30% sequence identity to generate representative/member
clusters for downstream train/test splitting.

Arguments:
  <input.fasta>          Input FASTA (e.g., data/interim/tss_seq_range.fasta)

Options:
  -t THREADS             Number of threads to use (default: number of CPUs)
  -p PREFIX              Output prefix (default: mmseqs30_linclust)
  -d TMPDIR              Temporary directory (default: $TMPDIR or /tmp)
  -h                     Show this help message and exit

Outputs (prefix = <PREFIX>):
  <PREFIX>_clusters.tsv       Representative/member pairs (tab-delimited)
  <PREFIX>_cluster_sizes.tsv  Cluster size summary (rep\tcluster_size)
  <PREFIX>_cluster_report.tsv Summary report (metrics)

Notes:
  - Uses MMseqs2 linclust with: --min-seq-id 0.30 -c 1.0 --cov-mode 0 --seq-id-mode 1
  - Identifies clusters based on 30%% identity (shorter-sequence relative), full-length coverage.
EOF
}

THREADS=$(nproc 2>/dev/null || echo 1)
PREFIX="mmseqs30_linclust"
TMPROOT="${TMPDIR:-/tmp}"

while getopts ":t:p:d:h" opt; do
    case "$opt" in
        t) THREADS="$OPTARG" ;; 
        p) PREFIX="$OPTARG" ;; 
        d) TMPROOT="$OPTARG" ;; 
        h) usage; exit 0 ;; 
        :) echo "ERROR: -$OPTARG requires an argument" >&2; usage; exit 1 ;; 
        \?) echo "ERROR: invalid option: -$OPTARG" >&2; usage; exit 1 ;; 
    esac
done
shift $((OPTIND - 1))

if [[ $# -ne 1 ]]; then
    usage
    exit 1
fi

FASTA="$1"

if [[ ! -f "$FASTA" ]]; then
    echo "ERROR: FASTA file not found: $FASTA" >&2
    exit 1
fi

MMTMP=$(mktemp -d -p "$TMPROOT" "mmseqs_${PREFIX}_tmp.XXXXXX")

cleanup() {
    rm -rf "$MMTMP"
}
trap cleanup EXIT

DB="${PREFIX}_db"
CLU="${PREFIX}_clu"
MEMBERS_TSV="${PREFIX}_clusters.tsv"
SIZES_TSV="${PREFIX}_cluster_sizes.tsv"
REPORT_TSV="${PREFIX}_cluster_report.tsv"

cat <<EOF
[$(date)] Starting MMseqs2 Linclust
Input FASTA : $FASTA
Prefix      : $PREFIX
Threads     : $THREADS
Temp dir    : $MMTMP
EOF

# Create MMseqs database
mmseqs createdb "$FASTA" "$DB"

# Lean clustering with Linclust:
# - 30% minimum sequence identity
# - full-length / bidirectional coverage
# - identity relative to the shorter sequence
#
# This is intended for grouping similar sequences for downstream train/test splitting.
mmseqs linclust "$DB" "$CLU" "$MMTMP" \
    --min-seq-id 0.30 \
    -c 1.0 \
    --cov-mode 0 \
    --seq-id-mode 1 \
    --threads "$THREADS"

# Export representative-member pairs
mmseqs createtsv "$DB" "$DB" "$CLU" "$MEMBERS_TSV"

# Count input sequences
TOTAL_INPUT_SEQ=$(grep -c '^>' "$FASTA")

# Cluster sizes: representative \t cluster_size
awk 'BEGIN{OFS="\t"} {count[$1]++} END{for (rep in count) print rep, count[rep]}' "$MEMBERS_TSV" \
    | sort -k2,2nr -k1,1 \
    > "$SIZES_TSV"

TOTAL_CLUSTERS=$(wc -l < "$SIZES_TSV")
TOTAL_ASSIGNED_SEQ=$(awk '{s+=$2} END{print s+0}' "$SIZES_TSV")
SINGLETON_CLUSTERS=$(awk '$2==1{n++} END{print n+0}' "$SIZES_TSV")
NON_SINGLETON_CLUSTERS=$(awk '$2>1{n++} END{print n+0}' "$SIZES_TSV")
SEQ_IN_SINGLETONS=$(awk '$2==1{s+=$2} END{print s+0}' "$SIZES_TSV")
SEQ_IN_NON_SINGLETONS=$(awk '$2>1{s+=$2} END{print s+0}' "$SIZES_TSV")
LARGEST_CLUSTER=$(awk 'BEGIN{m=0} $2>m{m=$2} END{print m+0}' "$SIZES_TSV")
MEAN_CLUSTER_SIZE=$(awk -v n="$TOTAL_CLUSTERS" '{s+=$2} END{if(n>0) printf "%.6f\n", s/n; else print "0"}' "$SIZES_TSV")

# Summary report TSV
{
    echo -e "metric\tvalue"
    echo -e "input_fasta\t$FASTA"
    echo -e "total_input_sequences\t$TOTAL_INPUT_SEQ"
    echo -e "total_clusters\t$TOTAL_CLUSTERS"
    echo -e "total_sequences_assigned_to_clusters\t$TOTAL_ASSIGNED_SEQ"
    echo -e "singleton_clusters\t$SINGLETON_CLUSTERS"
    echo -e "non_singleton_clusters\t$NON_SINGLETON_CLUSTERS"
    echo -e "sequences_in_singletons\t$SEQ_IN_SINGLETONS"
    echo -e "sequences_in_non_singletons\t$SEQ_IN_NON_SINGLETONS"
    echo -e "largest_cluster_size\t$LARGEST_CLUSTER"
    echo -e "mean_cluster_size\t$MEAN_CLUSTER_SIZE"
} > "$REPORT_TSV"

# Sanity check
if [[ "$TOTAL_ASSIGNED_SEQ" -ne "$TOTAL_INPUT_SEQ" ]]; then
    echo "WARNING: total assigned sequences ($TOTAL_ASSIGNED_SEQ) != input sequences ($TOTAL_INPUT_SEQ)" >&2
    echo "Please inspect: $MEMBERS_TSV" >&2
fi

cat <<EOF
[$(date)] Done.
Outputs:
  Representative/member pairs : $MEMBERS_TSV
  Cluster sizes               : $SIZES_TSV
  Summary report              : $REPORT_TSV
EOF
