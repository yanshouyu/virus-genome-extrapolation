#!/usr/bin/env python3
"""
Script to extract negative set sequences for TSS identification.

Negative regions are 300bp long, from intergenic regions (not in CDS or TSS).
Sequences are extracted separately per strand - only excluding regions on the same strand.
Same number of sequences on + and - strands (balanced).
Genome treated as linear.
Output: FASTA file.
"""

import random
from Bio import SeqIO
import pandas as pd
import sys
import argparse

def merge_intervals(intervals):
    """Merge overlapping/adjacent intervals.
    
    Args:
        intervals: list of (start, end) tuples (1-based, inclusive).
    Returns:
        Sorted list of merged, non-overlapping intervals.
    """
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for st, ed in intervals[1:]:
        if st <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], ed))
        else:
            merged.append((st, ed))
    return merged

def get_safe_gaps(exclusion_intervals, genome_len):
    """Get safe (non-excluded) gaps.
    
    Args:
        exclusion_intervals: list of (start, end) tuples (1-based, inclusive).
        genome_len: length of genome.
    Returns:
        List of safe gaps as (start, end) tuples (1-based, inclusive).
    """
    merged_excl = merge_intervals(exclusion_intervals)
    gaps = []
    prev_end = 0
    for st, ed in merged_excl:
        if st > prev_end + 1:
            gaps.append((prev_end + 1, st - 1))
        prev_end = max(prev_end, ed)
    if prev_end < genome_len:
        gaps.append((prev_end + 1, genome_len))
    return gaps

def sample_windows_from_gap(gap_st, gap_ed, window_size=300, max_windows=5):
    """Sample non-overlapping windows randomly from a gap until no more can fit.
    
    Repeatedly: randomly pick an available interval, randomly sample a window
    within it, then remove that window. Continue until no interval >= window_size
    or max_windows is reached.
    
    Args:
        gap_st, gap_ed: gap boundaries (1-based, inclusive).
        window_size: size of each window.
        max_windows: maximum number of windows to sample from this gap.
    Returns:
        List of (start, end) tuples for windows (1-based, inclusive).
    """
    gap_len = gap_ed - gap_st + 1
    if gap_len < window_size:
        return []
    
    windows = []
    available = [(gap_st, gap_ed)]  # Track available intervals
    
    while available and len(windows) < max_windows:
        # Find intervals large enough for a window
        candidates = [iv for iv in available if iv[1] - iv[0] + 1 >= window_size]
        if not candidates:
            break
        
        # Randomly pick one candidate interval
        iv_st, iv_ed = random.choice(candidates)
        iv_len = iv_ed - iv_st + 1
        
        # Randomly pick a start position within this interval
        max_start = iv_ed - window_size + 1
        w_st = random.randint(iv_st, max_start)
        w_ed = w_st + window_size - 1
        
        windows.append((w_st, w_ed))
        
        # Update available intervals by removing the sampled window
        new_available = []
        for st, ed in available:
            if ed < w_st or st > w_ed:
                # No overlap, keep this interval
                new_available.append((st, ed))
            else:
                # Overlap or partial overlap, split
                if st < w_st:
                    new_available.append((st, w_st - 1))
                if ed > w_ed:
                    new_available.append((w_ed + 1, ed))
        available = new_available
    
    return windows

def main():
    random.seed(42)  # Freeze seed for reproducibility
    parser = argparse.ArgumentParser(description="Extract negative TSS sequences from baculovirus genome.")
    parser.add_argument('--output', '-o', default='data/interim/tss_neg_seq_range.fasta',
                        help='Output FASTA file path (default: data/interim/tss_neg_seq_range.fasta)')
    args = parser.parse_args()

    # Load the genome sequence
    gb_file = "data/raw/baculovirus/Baculoviridae/NC_001623.1.gb"
    try:
        with open(gb_file, 'r') as f:
            seq_record = next(SeqIO.parse(f, 'genbank'))
        seq = seq_record.seq
        genome_len = len(seq)
        print(f"Loaded genome: {seq_record.id}, length: {genome_len} bp", file=sys.stderr)
    except Exception as e:
        print(f"Error loading genome file: {e}", file=sys.stderr)
        sys.exit(1)

    # Load positive ranges from CSV with strand information
    csv_file = "data/interim/tss_seq_range.csv"
    try:
        df = pd.read_csv(csv_file)
        # Separate by strand
        pos_plus = [(int(row['range_st']), int(row['range_ed'])) for _, row in df[df['strand'] == '+'].iterrows()]
        pos_minus = [(int(row['range_st']), int(row['range_ed'])) for _, row in df[df['strand'] == '-'].iterrows()]
        print(f"Loaded {len(pos_plus)} positive ranges on '+' strand", file=sys.stderr)
        print(f"Loaded {len(pos_minus)} positive ranges on '-' strand", file=sys.stderr)
    except Exception as e:
        print(f"Error loading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Load CDS intervals from GenBank features with strand information
    cds_plus = []
    cds_minus = []
    for feature in seq_record.features:
        if feature.type == 'CDS':
            st = feature.location.start + 1  # Convert to 1-based
            ed = feature.location.end
            # feature.location.strand: 1 for +, -1 for -, None for unspecified
            if feature.location.strand == 1 or feature.location.strand is None:
                cds_plus.append((st, ed))
            elif feature.location.strand == -1:
                cds_minus.append((st, ed))
    print(f"Loaded {len(cds_plus)} CDS intervals on '+' strand", file=sys.stderr)
    print(f"Loaded {len(cds_minus)} CDS intervals on '-' strand", file=sys.stderr)

    # Create exclusion intervals per strand
    excl_plus = cds_plus + pos_plus
    excl_minus = cds_minus + pos_minus
    print(f"Total exclusion intervals on '+': {len(excl_plus)}", file=sys.stderr)
    print(f"Total exclusion intervals on '-': {len(excl_minus)}", file=sys.stderr)

    # Get safe gaps per strand
    gaps_plus = get_safe_gaps(excl_plus, genome_len)
    gaps_minus = get_safe_gaps(excl_minus, genome_len)
    print(f"Found {len(gaps_plus)} safe gaps on '+' strand", file=sys.stderr)
    print(f"Found {len(gaps_minus)} safe gaps on '-' strand", file=sys.stderr)

    # Sample windows per strand
    windows_plus = []
    for gap_st, gap_ed in gaps_plus:
        windows_in_gap = sample_windows_from_gap(gap_st, gap_ed, window_size=300)
        windows_plus.extend(windows_in_gap)
    
    windows_minus = []
    for gap_st, gap_ed in gaps_minus:
        windows_in_gap = sample_windows_from_gap(gap_st, gap_ed, window_size=300)
        windows_minus.extend(windows_in_gap)
    
    print(f"Found {len(windows_plus)} valid 300bp windows on '+' strand", file=sys.stderr)
    print(f"Found {len(windows_minus)} valid 300bp windows on '-' strand", file=sys.stderr)

    # Balance strands: use min count for both
    min_count = min(len(windows_plus), len(windows_minus))
    if min_count == 0:
        print("No valid windows found on one or both strands. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Randomly select min_count from each strand
    windows_plus = random.sample(windows_plus, min_count)
    windows_minus = random.sample(windows_minus, min_count)
    
    all_windows = windows_plus + windows_minus
    all_strands = ['+'] * len(windows_plus) + ['-'] * len(windows_minus)
    
    # Shuffle together
    combined = list(zip(all_windows, all_strands))
    random.shuffle(combined)
    all_windows, all_strands = zip(*combined)
    
    print(f"Generating {min_count} sequences on '+' and {min_count} on '-' strand", file=sys.stderr)

    # Output FASTA
    try:
        with open(args.output, 'w') as outfile:
            for i, ((w_st, w_ed), strand) in enumerate(zip(all_windows, all_strands)):
                # Extract sequence (convert 1-based to 0-based python slicing)
                subseq = seq[w_st - 1 : w_ed]
                if strand == '-':
                    subseq = subseq.reverse_complement()
                # FASTA header: neg_{idx}|{0-based start}:{0-based end}|{strand}|0
                name = f"neg_{i}|{w_st - 1}:{w_ed}|{strand}|0"
                outfile.write(f">{name}\n")
                outfile.write(f"{subseq}\n")
        print(f"Sequences written to {args.output}", file=sys.stderr)
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()