# October 24 verification amendment

Declared after scientific submission and before downstream outcomes from this study were read. This does not alter the original training/checkpoint-selection protocol or its saved hashes.

1. Add a secondary MLM audit during independent checkpoint replay: select256 held-out viral windows using Python random seed2718 after excluding every genomic locus represented in the128 selection windows, including its opposite strand. Keep masks fixed with torch seed2718. Evaluate original and all four selected-budget states. This tests whether loss improvements extend beyond the repeatedly consulted selection panel. It is an audit, not a new checkpoint selector, and uses the same nine held-out species groups rather than new biological replication. Report exact panel IDs/hash/group composition.
2. After the100M full array AND its replay have finished, the500M array may use two simultaneous tasks if needed. No more than two GPU jobs may run across the entire study. This changes scheduling only; scientific configurations remain identical. The LoRA pilot still waits for both primary replay jobs.
3. All per-budget intervals are unadjusted descriptive development intervals. The4096-step comparison and the4096-minus512 changes are the planned endpoint comparisons; do not select a best downstream budget.
