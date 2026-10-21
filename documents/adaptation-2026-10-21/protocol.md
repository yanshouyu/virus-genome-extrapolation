# Multispecies NT baculovirus adaptation — fixed protocol, October 21, 2026

Authorized by the user after discussion of multispecies 100M/500M before/after adaptation and a feasible general-genome control. Existing clean_v2 is unchanged. Other-gene CDS overlap is accepted; the October 21 coordinate audit documents eight one-base boundaries and one compound-CDS exception. No additional negative dataset is generated.

## Question and model panel

Does masked-language-model continued pretraining on baculovirus improve TSS discrimination beyond the same unadapted multispecies checkpoint, equal-budget continued pretraining on a general-genome subset, and k-mers? Primary models: InstaDeepAI/nucleotide-transformer-v2-100m-multi-species and v2-500m-multi-species, with revisions resolved/pinned before execution. No human-only model is a primary arm. 250M and NTv1 2.5B remain potential follow-ons, not required primary jobs; 2.5B adaptation is not launched without a successful resource assessment and a scientific need.

## Corpus

Use the 97 existing GenBank records; exclude accession stem NC_001623, any record naming Autographa californica, the target organism/taxon, and exact/reverse-complement target-genome duplicates. Keep source hashes, accession/version, organism, taxonomy, coordinates and strand. Join same-species and exact-genome duplicate records before assigning groups. Hold out approximately 10% of species groups using seed 42 before windowing. Closely related different species may remain; this is target-excluded adaptation, not a strict unseen-clade claim.

Tile non-overlapping full 1536-base windows without circular wrapping; include forward and reverse complement in the same split. Keep only ACGT-only windows. This supplies local context longer than the downstream 300-base windows while bounding memory. Actual tokenizer lengths, including special tokens, must fit the loaded model; no silent truncation. Record dropped tails/ambiguous windows.

General control: use Saccharomyces cerevisiae, Caenorhabditis elegans and Drosophila melanogaster FASTA sources listed in the public original NT multispecies manifest. Pin downloaded file hashes and source URLs. Sample equal counts per organism as closely as possible, with the same 1536-base windows, orientations and number of training windows as the baculovirus training corpus. This is a small general-genome subset, not a reconstruction of the original 850-genome mixture. If sources are unavailable, report the control as unfinished rather than replacing it after performance inspection.

## Optimization and checkpoint selection

For each model and each corpus condition, use seeds 42,43,44. Full-parameter MLM continued pretraining: 512 optimizer steps; effective batch 32 (initial microbatch 4, accumulation 8); AdamW learning rate 1e-5, weight decay .01, linear warmup over first 26 steps then cosine decay; standard 15% token masking with 80% mask/10% random ordinary token/10% unchanged; bf16 autocast with fp32 parameters/optimizer. Exclude special/padding tokens from labels. Clip gradient norm 1.0. Change microbatch/accumulation only to preserve effective batch if memory demands it, recording the engineering change.

Use the same fixed 128 held-out baculovirus windows, selected without downstream labels, for both corpus conditions. Fixed validation masking seed 1729 makes checkpoint comparisons reproducible. Evaluate initially and at optimizer steps 128,256,384,512. Select the lowest MLM-loss trained checkpoint among those four (ties earlier step), with initial loss reported separately. Selection never consults TSS validation or test results. Equal optimizer steps, batch size, input lengths and masking probability control training exposure; record actual input/masked token counts and GPU seconds. Save selected encoder weights and configuration plus optimizer-step metrics. The same held-out baculovirus loss selects both corpus arms, giving both the same target-domain selection opportunity.

First use bounded engineering smoke jobs, five optimizer steps and shortened evaluation, before scientific arrays. No downstream test in smoke. Check finite losses/gradients, actual parameter changes, memory/time and save/reload identity. Treat failures as engineering evidence, preserve outputs, and do not alter scientific settings based on downstream performance.

## Downstream representation and readout

Read training/validation only during development. Extract final-layer CLS and masked mean. Select pooling for each size from its unadapted representation using nested grouped CV log loss on training only (five outer / three inner folds, seeds 42/43); ties CLS. Hold that pooling fixed for every before/after comparison at that size. StandardScaler and L2 logistic C=[.01,.1,1,10,100], lbfgs max_iter=5000, fitting scalers only within fitting folds. For final train-only readouts, choose C by grouped inner out-of-fold log loss on training only and hold C fixed across adaptation arms at that size. This isolates representation changes from readout selection; report nested unadapted screening separately from fixed-C comparison. Fit all readouts on 233 training windows, evaluate once on 77 validation windows per arm. Fixed probability>0.5 threshold. K-mer receives its own training-only C selection by the same rule.

Primary development outcomes: validation log loss and MCC; accuracy, AUC, AP, precision/recall secondary. Show every adaptation seed and paired group-bootstrap intervals (2000 paired validation-group resamples, averaging seed metrics). Intervals condition on this cohort and trained models; they do not establish new-genome replication. Do not choose a winning adaptation seed.

After all development outputs, selection rules and engineering checks are frozen, final comparison may predict the existing 78-window test once per final selected run and baseline, with settings unchanged. This cohort was inspected in previous work; label this as a fixed secondary evaluation, not a pristine independent external test. Do not expand the grid based on test outcomes. If the development results support no useful improvement, retain/report all arms unchanged rather than tune until positive.

## Resource and completion policy

Nibi saved documentation supports h100_1g.10gb (1/8 compute), h100_2g.20gb (2/8), h100_3g.40gb (3/8), and full h100 (80GB). Start 100M training smoke on 20GB and 500M on 40GB, 4 CPUs, 32GB host RAM. Use SLURM only for GPU training/inference, at most two concurrent training tasks. Scientific tasks initially bounded to two hours each, revised only from measured runtime. 2.5B may need a larger slice/full GPU; do not assume the 10GB probe allocation suffices.

Minimum deliverable: both multispecies baselines and both corpus adaptation conditions, three seeds, source/corpus audit, repeatable evaluation and cost report. Follow-ons: saved-model replay/parameter verification, uncertainty and failure analysis, fixed secondary test evaluation if all prerequisites pass, final report and handoff. Continue through the full declared backlog; no arbitrary sweeps to occupy time. Report actual elapsed time, queued/unfinished jobs and any remaining limitations. No automatic recurring tasks are needed.
