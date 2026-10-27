# Historical adaptation CUDA failure: recovered evidence

The Nibi logs for jobs 11341550, 11373064 and 11434381 were recovered read-only on 2026-10-19. They concern the **NTv1 2.5B multispecies model**, not the NTv2 500M probe used in the new baseline comparison. The third job was run on April 2, 2026, on an H100. Its saved configuration used 6000-base windows with stride 3000, batch size 2, BF16 and gradient checkpointing. None of those historical jobs was rerun.

## Confirmed positional mismatch

The third traceback identifies the embedding operation directly: `position_embeddings(position_ids)` calls a table lookup and raises the CUDA device-side assertion. Earlier traces ended at later CUDA operations, which could reflect asynchronous error reporting; those traces alone were less diagnostic.

The cached model revision `b746b125aacd1b0970c05a32fd71ba726754542e` has absolute position embeddings, 1002 position-table rows and padding index 1. ESM's non-padding position IDs begin at 2, so the maximum safe number of non-padding tokens is **1000**. Its tokenizer also declares `model_max_length=1000`.

The available adaptation source on `origin/feat/domain-adaptation` hard-codes `max_length=1026`, overriding that tokenizer limit. The first real 6000-base AcMNPV window produces **1001 non-padding tokens**, including CLS, and its maximum position ID is **1002**, outside table indices 0..1001. Padding to 1026 is not by itself the problem: the number of non-padding tokens is already too large.

A new CPU-only diagnostic loaded the cached tokenizer/configuration and a tiny eight-dimensional ESM embedding table with the same positional capacity. It reproduced `IndexError: index out of range in self` on the real window. Explicit truncation to the safe 1000-token budget produced finite embeddings. No 2.5B weights, GPU training or adaptation pipeline were required. CPU job 24607770 completed successfully.

This confirms a configuration/input mismatch that reproduces the historical failing operation. The exact source commit used by the historical jobs was not recorded in the recovered run folders; the available branch is corroborating source, not an established exact run-to-commit mapping. Fixing this mismatch does not establish that all other adaptation issues are solved.

## Recommended future repair

Derive the token budget from the actual loaded model and tokenizer, including special tokens and the ESM position offset. Check actual non-padding position IDs before GPU execution; do not assume every 6000-base input always compresses to 1000 or fewer tokens. Either truncate explicitly with documented loss of sequence context or reduce/split the nucleotide windows and verify every resulting tokenized sample. A simple six-mer calculation must include CLS and any remainder/ambiguous-base tokens.

The new 300-base probing experiments do not use the adaptation corpus code. The adaptation branch was not merged or restarted. Its overlapping-window random validation split and loss of accession metadata still need repair before any scientific adaptation experiment.

The recovered [corpus report](failed-corpus-report.json) also explicitly includes **44 windows from NC_001623.1**, the genome used for the downstream TSS benchmark. Future adaptation must decide whether target-genome exposure is permitted. For a claim about unseen-genome transfer, exclude downstream target genomes and define the treatment of close relatives before adaptation; a randomly split unlabeled target-genome corpus does not provide that independence.

## Evidence and reproduction

- [Diagnostic result](position-budget.json).
- [Saved failed-run configuration](failed-run-config.json).
- [Traceback excerpt with original line numbers](position-embedding-traceback.txt).
- [Hashes of all six original log files](source-hashes.json); full logs were copied locally under ignored `runs/2026-10-19/historical-logs/` and remain unchanged on Nibi.
- Diagnostic source: `scripts/check_position_budget.py`. Run it in the prepared environment with `--model-dir` pointing to the cached revision, `--genbank data/raw/baculovirus/Baculoviridae/NC_001623.1.gb` and an explicit `--output` path. It does not load pretrained model weights.
