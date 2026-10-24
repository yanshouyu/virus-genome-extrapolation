# Duration study and bounded LoRA pilot — fixed October 24, 2026

Authorized window: 2026-10-24 12:57:11–14:57:11 UTC (08:57:11–10:57:11 Toronto). Queued jobs may finish later; no automatic monitoring is installed. No downstream test inference is authorized in this window. Original AcMNPV dataset and target-excluded, species-grouped paired-strand corpora are unchanged.

## Primary question and design

Did the previous 512-step budget leave useful baculovirus adaptation unfinished? Both NTv2 100M (multispecies) and NTv2 500M (multispecies), pinned October 23 revisions, are restarted from their original pretrained weights. Old runs lack optimizer/RNG state and used cosine decay to zero at step 512; loading their weights would not be an exact continuation.

For each size, both baculovirus and general control, seeds 42/43/44, run exactly 4096 steps, effective batch32, AdamW lr1e-5, weight decay .01, warmup26 then cosine decay over4096 steps. Identical sequence/masking procedure, fixed128-window viral MLM validation with masksseed1729. GPU fullH100, microbatch16/accumulation2 after smoke verification. Maximum two simultaneous GPU jobs. No budget cuts based on TSS outcomes. Save all MLM evaluation points every256 steps plus128; at budgets512/1024/2048/4096 extract the best eligible trained state selected only by viral MLM loss. Initial model remains separately reported. Store selected states and identifiers for each budget, with exact reload checks.

The duration comparison uses prefixes of this one4096-step schedule. A 512-step prefix is not an independently optimized512-step schedule; comparison against October23 is descriptive because schedule and100M microbatch differ. More training within the new matched trajectories is the primary duration comparison. Both conditions share schedule/seed/input and masked-token exposure; report different selected checkpoint exposure.

Run to the fixed cap rather than adaptive stopping. Describe approximate held-out loss saturation only if cumulative best MLM loss improves <0.5% over the final1024 steps (3072→4096); otherwise explicitly report that adequacy is unresolved at the cap. This is a practical diagnostic, not proof of complete convergence. Retain endpoint losses to expose deterioration hidden by cumulative best.

## Readout and outcomes

Use the existing independently frozen masked-mean/C=.01 readout for both NT sizes and k-mers. Fit scaler/logistic only on233train windows; evaluate77validation windows/64groups, probability>.5. No pooling/C/threshold retuning. Report log loss/MCC/AUC at each budget and all seeds, with paired group-bootstrap95% intervals using2000 resamples. Compare long vs512prefix, baculovirus vs general, and fixed baselines. Curves are descriptive development analysis and cannot certify unseen-species performance. Existing78-window test remains unopened for prediction; it was inspected in historical project work and should not later be called pristine.

## Optional LoRA pilot

After engineering smoke, 100M(multispecies) only, query/value projections in every transformer layer, rank8, alpha16, zero-initializedB/KaimingA, no adapter dropout, freeze every original parameter including MLM head/embeddings. Same MLM loss, data, seed42, budget4096 and checkpoints, baculovirus plus general conditions. Fixed exploratory lr1e-4; all other optimizer/schedule settings unchanged. Small explicit PyTorch wrapper implementing W x+(alpha/r)BAx, following https://github.com/microsoft/LoRA/blob/main/loralib/layers.py; no package/environment upgrade. Verify zero-update baseline identity, nonzero adapter updates, frozen-weight identity and checkpoint reload. Save adapter-only states with pinned source identity.

This single-seed, fixed-hyperparameter pilot evaluates feasibility and actual memory/time. It is NOT an equally tuned three-seed superiority/noninferiority comparison. Do not rank LoRA as a primary winner. A formal follow-up would give full and LoRA equal small learning-rate selection allowances using held-out MLM only and repeat the selected settings across three seeds.

## Execution and reporting

Separate smoke outputs from scientific directories; failures stay preserved. Full100M/500M use separate six-task arrays with concurrency1 each; pilot must wait until both finish, preserving total≤2. Scientific time limits60minutes per run, refined only from smoke timing. Save source hashes, source reference fidelity, state hashes, loss curves, token counts, memory/cost and SLURM job accounting. GPU calculations only through SLURM. If resources remain queued near the deadline, deliver exact job IDs, pending reasons and commands for collection rather than claiming completion. The two-hour window is a wall-clock limit for agent work, not a requirement to occupy the entire period.
