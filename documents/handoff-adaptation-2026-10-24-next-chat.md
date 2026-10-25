# New-chat instructions — interpret the completed October 24 study

Saved October 24, 2026 (America/Toronto). This supersedes handoff-adaptation-2026-10-24-progress.md for current status. The user requested a saved next-step plan so work can continue in a new chat. Do not restart training or redo completed GPU work merely because the old handoff says jobs were running.

## Objective and authorized scope for the next chat

Determine what the completed duration study and LoRA pilot imply about the usefulness of adapting a broad-genome DNA language model to baculovirus genomes for this AcMNPV regulatory-window probing task. Collect and independently check compact results, prepare scientific figures and an interpreted report, and recommend the minimum remaining work to wrap up the project.

Prioritize analysis and project closure. Do not automatically launch new model sweeps, new datasets, additional training, or real downstream test inference. The last study explicitly excluded test prediction. If a final test evaluation is recommended, first state the exact frozen comparison and its historical exposure limits; it is a separate next decision, not part of this analysis handoff.

## Verified current state

On October 24 at 20:55 Toronto, the agent checked SLURM accounting: both smoke jobs, all 12 primary tasks, both LoRA pilot tasks, all replays, both development analyses and the final report job COMPLETED with exit code0:0. Queue was empty. Server report verification completed October24 at15:43:13UTC (11:43Toronto), after the prior chat had handed work to the server early. No agent was monitoring in the background.

All 14 scientific runs and59 readouts passed the report gates. Fresh-process embedding replay maximum difference0.0; logistic probability replay maximum difference1.1102230246251565e-16. All scientific/analysis source hashes, selected-budget checkpoint selections, unchanged train/validation data hashes, and independent audit panel/masks match. No real downstream test predictions were generated. All six full baculovirus runs meet the predeclared practical loss-plateau rule; inspect independent audit losses before interpreting this as adequate adaptation. This criterion is conditional on the schedule and validation panel, not proof of universal convergence.

Local verified evidence now exists:

- documents/adaptation-2026-10-24/results/README.md — automatically generated full score/effect report.
- results/verification.json — final provenance/replay checks.
- results/development-metrics.json — all models, paired comparisons, readout settings and analysis versions.
- results/training-summary.csv, run-metadata.json, histories.json — resource use, selected steps, token counts, learning curves and provenance.
- results/independent-mlm-audit.csv and fresh-replay-*.json — audit losses and exact audit panel/mask identities.
- results/*_validation.csv — all59 readouts' predictions, with identifiers/labels.
- results/performance-table.md and lora-pilot-table.md — preserve separately labelled primary and exploratory results.
- documents/adaptation-2026-10-24/jobs-completed.psv — completed SLURM accounting, including exit codes and UTC times.

The entire results/ folder has been copied from Nibi. Large model weights have not been downloaded. Most local runs/2026-10-24/collected/ entries remain an early partial snapshot; do not mistake that snapshot for the completed server corpus of results.

## Read first

1. This file, then documents/adaptation-2026-10-24/results/README.md and verification.json.
2. documents/adaptation-2026-10-24/protocol.md and verification-amendment.md.
3. documents/adaptation-2026-10-24/execution.md for paths, job graph and collection.
4. documents/adaptation-2026-10-23/README.md for the original512-step comparison.
5. documents/agent_doc.tex and documents/project-assessment.md if the project scope or final deliverable needs clarification. Separate the current probing study from the broader aspirational program.

## Scientific design and interpretation constraints

- Primary checkpoints: NTv2 100M (multispecies) and NTv2 500M (multispecies), original pinned revisions retained in metadata. Every leaderboard must identify original pretraining scope: multispecies, human or1000G as applicable. No human-only or1000G model is part of the current primary panel. The earlier NTv1 2.5B (multispecies) result is a frozen exploratory reference; it was not adapted and is not a causal size comparison.
- Primary study: full-parameter MLM, both baculovirus and general-control corpora, seeds42/43/44,4096steps, effective batch32, microbatch16/accumulation2, fullH100, lr1e-5, warmup26 then cosine decay over4096steps. Save assessments at budgets512/1024/2048/4096, selecting trained checkpoints only by fixed held-out viral MLM loss.
- The new trajectories restart from pretrained weights. The old512-step runs lacked optimizer/RNG state and decayed learning rate to zero at512. A new512-step prefix of a4096-step schedule is not the old experiment or an independently optimized short schedule. Use within-trajectory4096-minus512 as the planned duration comparison, with this limitation stated.
- Practical plateau: cumulative-best selection loss improves <0.5% over final1024steps. Selection loss can look flat because learning rate decays; audit and downstream evidence are needed. Check endpoint loss as well as selected loss, and report each seed rather than only a mean.
- General control matches training budget and strand-paired locus exposure, but is a yeast/worm/fly subset with different composition and diversity. It tests domain-specific gains beyond ordinary continued training, not a perfectly matched reconstruction of the original pretraining mixture. Selected checkpoint exposures can differ even with matched maximum budgets.
- Adaptation excludes AcMNPV (NC_001623.1), including the original target-name/duplicate exclusions. The remaining baculovirus data is divided into training/held-out species groups. The new independent256-window audit excludes BOTH orientations of every selection-panel genomic locus. These audit windows share the nine held-out species groups; they are not independent biological replication.
- Downstream data remains233train/77validation/78test from ONE AcMNPV genome. Mean pooling and logistic C=.01 are frozen from earlier training-only selection. Scaling/logistic fitting uses training only. Threshold probability>.5. Do not retune pooling, C, threshold or choose a winning downstream budget/seed after seeing results.
- Main outcomes: log loss, MCC and ROC AUC; present uncertainty. The2000-resample paired group-bootstrap intervals use64validationgroups, condition on trained models/cohort, and do not represent independent genomes. Duration comparisons are multiple unadjusted development comparisons; planned endpoint comparisons are4096 baculovirus vs general/unadapted and4096-minus512.
- LoRA is an EXPLORATORY single-seed100M(multispecies) pilot, both corpora, query/value rank8, alpha16, no adapter dropout, frozen original parameters including MLM head/embeddings, lr1e-4. Compare its seed42 with full seed42, not a best full seed or an unmatched seed average as proof of superiority. No equal hyperparameter selection was performed; method and learning rate differ. Report feasibility/efficiency, not equivalence or necessity. Original parameters and adapter-only checkpoint reload were checked.
- Positives overlapping a DIFFERENT gene's CDS are accepted. Do not regenerate the downstream data or reopen the previously settled overlap strategy. Earlier audit exceptions are documented separately.
- The real downstream test was excluded from this study, but that cohort was inspected in earlier project history. Do not describe it as a pristine independent external test.

## Next steps, in order

1. Collect remaining compact artifacts from Nibi if needed for figures and local readout replay. Copy scientific/*.json, *.npz, COMPLETE and subdirectories into local runs/2026-10-24/collected/; exclude *.pt weights. Copy development-primary/, development-with-pilot/ and root fresh-replay JSON files to the corresponding local runs/2026-10-24/ paths. Keep all raw server outputs intact.
2. Check verification.json and accounting, compare expected run/readout counts, and independently reproduce metrics from saved probabilities. If needed, replay saved scaler/logistic coefficients against collected train/validation embeddings and verify training-only scaling. The old scripts/verify_adaptation_readouts.py assumes the October23 layout: do not invoke it unchanged against the new structure. Reuse the new evaluator logic or write a narrowly scoped verifier. No GPU training or fresh embedding inference should be necessary.
3. Answer the adequacy question using all six baculovirus trajectories: selection loss and independent audit loss at each budget, endpoint vs selected state, and the plateau diagnostic. Distinguish language-model adaptation from improved regulatory prediction. Check whether general-control loss behaves differently and why selected steps remain early where applicable.
4. Answer the project question using the fixed comparisons: does longer baculovirus training improve the task, does it outperform equally budgeted general continued training, and does it provide practical benefit over k-mers? Report both probability quality (log loss) and fixed-threshold decisions (MCC), plus AUC and paired uncertainty. Do not select the strongest-looking metric or budget as the sole conclusion.
5. Summarize the LoRA pilot separately: parameter fraction, GPU memory, allocated/runtime costs, MLM/audit learning and paired seed42 task metrics. Explain its tuning/replication limitations. Do not launch a formal LoRA search unless the user later authorizes a specifically justified follow-up.
6. Produce figures: MLM selection/audit curves, downstream log-loss/MCC versus budget, planned endpoint effects with95% intervals, and a compact full-vs-LoRA resource/performance comparison. Use standard plotting tools for exportable scientific figures, inspect rendered figures, and label every model's original pretraining scope. scripts/summarize_adaptation_duration.py already generates selection-loss/development curves and tables after compact collection; extend only where needed for audit/effect/cost plots.
7. Write documents/adaptation-2026-10-24/interpretation.md with the evidence, a restrained conclusion about domain-specific adaptation under this regime, limitations, and a concrete project-closure recommendation. Preserve results/ as raw checked evidence. Update the study index/progress handoff to point to completed analysis. Commit project-owned changes and synchronize to Nibi if available; preserve unrelated user files.
8. Brief the user with the main finding, what is resolved, what remains uncertain and the smallest remaining deliverable to finish the scoped project. Prioritize final report/manuscript assembly if the evidence is already sufficient for a defensible bounded conclusion. A new annotated species is necessary for a broader cross-baculovirus claim but is not automatically required to finish an AcMNPV-focused study. Do not promise that adaptation must win or that more training will resolve a small-cohort uncertainty problem.

## Execution context and safeguards

Local Windows workspace: \\wsl.localhost\Ubuntu\home\s34yan\projects\260308_virus-genome-extrapolation
WSL workspace: /home/s34yan/projects/260308_virus-genome-extrapolation
Nibi root: /home/shouyu/projects/def-aucoin/shouyu/virus-genome-extrapolation (resolved /project/6002642/shouyu/virus-genome-extrapolation).
Shared SSH socket: /home/s34yan/.ssh/nibi-codex.sock. Host: shouyu@nibi.alliancecan.ca. Test this existing connection before requesting user action. Do not read provider credentials; optional DeepSeek was not used or needed. No subagents were authorized for this task.

Shell is PowerShell. WSL commands may need the approved escalation path. Use ASCII saved bash helpers under ignored tmp/, normalize CRLF, then run in WSL; avoid multi-shell quoting hazards. rg is unavailable inside this WSL environment, so use an available alternative there. Local analysis environment: /home/s34yan/miniconda3/envs/ece657d/bin/python. Cluster environment and module instructions are in execution.md. GPU/heavy cluster work uses SLURM, never login-node GPU computation.

Current branch: codex/leakage-and-model-fixes. Source/execution handoff commit before this collection:98c9334, matching local/Nibi. The commit containing this file and copied final evidence follows it. Do not assume a GitHub push was made.

Preserve unrelated local scripts/run_training.slurm and remote logs/, run_training.slurm, scripts/mmseqs_cluster_30id.slurm, and old slurm outputs. No recurring monitor or automation exists. No additional time window or token budget has been supplied for the next chat; follow the user's new instruction.
