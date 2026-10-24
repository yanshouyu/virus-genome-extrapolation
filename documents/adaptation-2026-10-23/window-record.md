# October 23 execution record

Authorized window: 2026-10-23 21:43:37–23:43:37 Toronto (2026-10-24 01:43:37–03:43:37 UTC). No real downstream test inference or selection is authorized in this window.

## Cache/autograd repair

Original smoke jobs 24757107 (100M) and 24757108 (500M) failed on the first backward pass. Their initial evaluation under torch.inference_mode created cached rotary-position tensors that cannot be saved for backward. They did not fail from insufficient GPU memory and produced no completed adaptation result.

The trainable-model process now uses torch.no_grad for evaluation and extraction. A real tiny-NTv2 regression evaluates before training, then verifies finite backward/update and exact reload. Corpus, optimizer, learning rate, effective batch, scientific step budget and selection rules were unchanged by this repair. Failed outputs/logs remain preserved.

Corrected 100M smoke 24896237 completed in fresh directory smoke-100m-v2: initial encoder tensors exactly match pinned source; weights update; selected state reloads exactly. Peak reserved memory: 3.261GB on a 20GB MIG slice. Five-step training section: 4.85s.

A queued 100M microbatch-16 benchmark (24896356) was cancelled before execution to use the already-verified microbatch-4 configuration. The 40GB 500M retry (24896238) and its dependent benchmark (24896357) remained pending and were cancelled before execution. Full-H100 500M smoke 24896545, microbatch 16, completed in smoke-500m-full: 20.208GB reserved; five-step training section 6.63s; source fidelity, parameter updates and exact selected-state reload passed. Full H100 was chosen for queue availability, not because 40GB was insufficient.

## Fixed scientific runs

Baculovirus arrays: 100M 24896408 (20GB MIG, microbatch 4/accumulation 8); 500M 24896596 (full H100, microbatch 16/accumulation 2). Both use effective batch 32 and all 512 optimizer steps. Walltime was set to 30 minutes from measured progress, with substantial headroom. At most one task per size is active. Baculovirus seeds 42/43/44 correspond to task indices 0/2/4.

CPU job 24896592 independently verified every unique MLM parameter against pinned safetensors for both sizes, including the prediction head. Job 24896746 froze readouts using only unadapted training embeddings before adapted TSS outcomes were inspected: masked mean and C=0.01 for both model sizes; k-mer C=0.01. Production analysis checks that it reproduces these independently frozen choices.

Historical 500M embeddings are numerically close rather than bit-identical across old MIG/inference and new full-H100/no_grad extraction paths. Maximum differences: 4.39e-5 (CLS), 1.05e-5 (mean). Historical mean-readout validation probabilities differ by at most 1.84e-6; no decisions change. Source parameters match exactly. This is documented numerical continuity, not exact historical embedding equality. Within the new study, first paired baseline embeddings match exactly for both sizes.

## General-control pairing correction

An exposure audit before TSS outcome analysis found that the first general-control sampler selected individual oriented windows, whereas baculovirus data paired both orientations at each selected locus. Both had 14,164 windows, but independent-locus counts differed (7,082 baculovirus versus 13,333 general). The implementation did not fully satisfy the intended strand-paired control.

All completed/partial original general-control runs were moved to excluded-general-unpaired/ locally/on Nibi and excluded from the scientific panel. Remaining original general-control task indices 3/5 and their downstream analysis/replay jobs were cancelled. Baculovirus runs were retained unchanged. Original array aggregate states can therefore be CANCELLED while their individual baculovirus tasks remain valid.

The repaired sampler selects loci first, then includes their reverse complements. Both corpora now contain 14,164 windows, 7,082 independent genomic loci, and exactly 7,082 windows per strand. General allocation is 4,722/4,722/4,720 windows across yeast/worm/fly. Both corpora have the same total and unique-locus nucleotide counts. Generic GC composition and species diversity are intentionally not matched. Old control hash: 6ceb8e08eee7fa0e6e8d061edf50ba308401a0ed1d8e4943427a57606dc57bcd. Repaired control hash: 6455320fd95ae86d27d32c80c101b18c9529cc064f252e7972181a947d852e04.

All source-coordinate, orientation, target-exclusion, group separation and file-hash checks pass. A new regression prevents individual-strand sampling from replacing locus-first pairing. Corrected general arrays are 24897074 (100M) and 24897075 (500M), task indices 1/3/5 for seeds 42/43/44. They wait for their respective original arrays to finish, preserving the two-active-GPU bound. No downstream dataset, adaptation budget, seed or readout grid was changed. No adapted TSS performance was used to decide this correction.

Updated replay jobs: 24897149 (100M), 24897151 (500M). Updated development jobs: 24897152 (100M partial panel), 24897153 (full panel), dependent on the corrected arrays. Previous downstream jobs 24896760/24896761/24896828/24896829 were cancelled and superseded. An earlier dependency submission referencing the already-completed freezing job was rejected before any job was created; successful submissions instead check the frozen-selection file and depend on active arrays.
