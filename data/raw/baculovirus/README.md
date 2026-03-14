# Baculoviridae raw data (regenerated on demand)

This folder holds a small **manifest** and a **runner script** to (re)download GenBank records for Baculoviridae accessions. Heavy data files are **not** tracked in Git; they are reproduced on any machine.

## Contents
- `Baculoviridae.csv` — accession manifest. Downloaded from https://www.ncbi.nlm.nih.gov/labs/virus/vssi filtered by taxid: 10442 (Baculoviridae), nucleotide completeness, and assembly completeness.
- `run_download.sh` — reproducible runner that downloads all records listed in the CSV.
- `Baculoviridae/` — **downloaded** `.gb` files (ignored by Git).
- (Repo root) `scripts/download_genbank_batch.sh` — batch downloader used by the runner.
- `Baculovirus_TSS.csv`: Experimentally validated TSS sites. Selected columns from table S2 of Chen *et al.* 2013. The table is neither tidy nor normal. [`notebooks/0_process_baculovirus_tss.ipynb`](notebooks/0_process_baculovirus_tss.ipynb) records the processing of this table.

## Reproduce the genomes data
From the repo root (or from this folder):
```bash
# optional: make sure scripts are executable once
chmod +x data/raw/baculovirus/run_download.sh
chmod +x scripts/download_genbank_batch.sh

# run (creates data/raw/baculovirus/Baculoviridae/*.gb)
./data/raw/baculovirus/run_download.sh
``