# Progress checkpoint — October 23 adaptation window

This is an active-work checkpoint, not a completion report. Continue until the declared development comparison and verification are done or the window ends. Deadline: October 23 at 23:43:37 Toronto / October 24 at 03:43:37 UTC. Do not run real downstream test inference in this window.

Read documents/adaptation-2026-10-23/window-record.md for fixes and exclusions. Source base was 65c888d. The training runner was repaired to use torch.no_grad instead of inference_mode in a trainable NTv2 process; regression and smoke checks pass. Its executed hash is 2c6a2ad9907ea0f85cc7513b170026c135ed31ca7e2b168312f0ab063096c2dc.

## Current scientific panel

All six baculovirus runs completed: 100M array 24896408 tasks 0/2/4 and 500M array 24896596 tasks 0/2/4, for seeds 42/43/44. Original general controls sampled individual strands and were excluded before TSS outcome analysis. They are preserved in runs/2026-10-21/excluded-general-unpaired/ on Nibi and runs/2026-10-23/excluded-general-unpaired/ locally. Old general tasks 3/5 were cancelled; aggregate CANCELLED states of the original arrays are therefore expected.

The corrected general control pairs both strands at each of 7,082 loci, matching baculovirus 14,164 windows and unique-locus count. Its hash is 6455320fd95ae86d27d32c80c101b18c9529cc064f252e7972181a947d852e04. Valid corrected general arrays: 24897074 (100M) and 24897075 (500M), tasks 1/3/5 for seeds 42/43/44, one active per size. These are underway. Both corpora and all file/source/strand checks pass; see corpus-report-v2.json and corpus-verification-v2.json in the October 23 report folder.

Every scientific run keeps 512 steps, lr 1e-5 and effective batch 32. 100M uses 20GB MIG, microbatch 4/accumulation 8; 500M uses full H100, microbatch 16/accumulation 2. Full H100 was used for queue availability; observed reserved memory about 20.2GB would fit a 40GB slice. Each job has a measured conservative 30-minute bound; at most two GPU jobs run concurrently.

## Verification and analysis jobs

CPU job 24896592 verified every source MLM parameter including the prediction head exactly against the pinned model files. CPU job 24896746 froze pooling/readout choices on unadapted training rows before adapted outcomes: mean pooling and C=.01 for both sizes, k-mer C=.01. Frozen artifact: runs/2026-10-21/frozen-readout-selection.json. Production analysis checks it reproduces these choices.

Fresh-process replay jobs: 24897149 (100M), 24897151 (500M). Development jobs: 24897152 (100M partial panel), 24897153 (full panel). All depend on the corrected arrays. Older replay/development jobs were cancelled and superseded. No secondary-test job was submitted.

Nibi project root: /home/shouyu/projects/def-aucoin/shouyu/virus-genome-extrapolation. Full scientific artifacts: runs/2026-10-21/scientific/. Compact local copies: runs/2026-10-23/collected/. Collect using rsync includes *.json, *.npz and COMPLETE; leave large selected_state.pt files on Nibi unless needed. The reused scientific names now contain only valid controls; excluded earlier outputs were moved aside before reruns.

Development outputs: runs/2026-10-21/development/ and development-100m/. Partial analysis uses SIZE_PANEL_COMPLETE and cannot gate final test. Full fresh-replay files: fresh-replay-100m.json / fresh-replay-500m.json. Analysis environment and corpus/source hashes are recorded separately.

## Remaining work

Monitor jobs and diagnose failures without retuning. Collect all twelve verified outputs and full development results. Run scripts/summarize_controlled_adaptation.py with --collected LOCAL_COLLECTION --development LOCAL_DEVELOPMENT --output documents/adaptation-2026-10-23. It requires corpus-report-v2.json, rejects old control hashes, checks checkpoint selection and matched input/masked token counts, and produces a table, figures, predictions and provenance. Inspect plots, independently replay readouts/recompute metrics, write final report/reproduction/handoff, commit owned changes and synchronize Nibi. Keep real test untouched.

Local tests: 24 passed, 2 large opt-in checks skipped; non-palindromic pairing assertion also checked. Historical 500M embedding differences are tiny floating-point differences across old/new paths; see baseline-continuity.json. The historical readout changes validation probabilities by at most 1.84e-6 and no decisions. Within-study paired unadapted baselines matched exactly at the first check.

Preserve the user's untracked scripts/run_training.slurm and all pre-existing Nibi artifacts. Shared SSH socket /home/s34yan/.ssh/nibi-codex.sock works. No other LLM, recurring automation, new chat or provider credentials were used.
