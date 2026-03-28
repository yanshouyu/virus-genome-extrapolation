#!/usr/bin/env python3
"""
Script to extract 300bp sequences around TSS sites from baculovirus genome.

Each sequence is extracted from range_st to range_ed (1-based, inclusive),
adjusted for strand orientation. Sequences are named with TSS position,
0-based range, strand, and label "1".
"""

from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract TSS sequences from baculovirus genome.")
    parser.add_argument('--output', '-o', default='data/interim/tss_seq_range.fasta',
                        help='Output FASTA file path (default: data/interim/tss_seq_range.fasta)')
    args = parser.parse_args()
    # Load the genome sequence
    gb_file = "data/raw/baculovirus/Baculoviridae/NC_001623.1.gb"
    try:
        with open(gb_file, 'r') as f:
            seq_record = next(SeqIO.parse(f, 'genbank'))
        seq = seq_record.seq
        genome_length = len(seq)
        print(f"Loaded genome: {seq_record.id}, length: {genome_length} bp", file=sys.stderr)
    except Exception as e:
        print(f"Error loading genome file: {e}", file=sys.stderr)
        sys.exit(1)

    # Read the CSV file
    csv_file = "data/interim/tss_seq_range.csv"
    try:
        df = pd.read_csv(csv_file)
        with open(args.output, 'w') as outfile:
            for _, row in df.iterrows():
                tss = int(row['TSS']) - 1  # Convert to 0-based
                strand = row['strand']
                range_st = int(row['range_st'])
                range_ed = int(row['range_ed'])

                # Convert to 0-based Python slicing
                start = range_st - 1
                end = range_ed  # seq[start:end] gives positions range_st to range_ed inclusive

                # Handle circular genome if needed (though ranges appear to be within bounds)
                if end > genome_length:
                    # If range wraps around, concatenate
                    subseq = seq[start:] + seq[:end - genome_length]
                else:
                    subseq = seq[start:end]

                # Reverse complement if negative strand
                if strand == '-':
                    subseq = subseq.reverse_complement()

                # Create sequence name: TSS|start:end|strand|1 (0-based positions)
                name = f"{tss}|{start}:{end}|{strand}|1"

                # Output in FASTA format
                outfile.write(f">{name}\n")
                outfile.write(f"{subseq}\n")

        print(f"Sequences written to {args.output}", file=sys.stderr)

    except Exception as e:
        print(f"Error processing CSV file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()