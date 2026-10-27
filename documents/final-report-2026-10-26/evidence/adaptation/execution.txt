# October 24 execution and collection

The fixed protocol is protocol.md; verification-amendment.md adds an independent MLM audit and permits redistributing the two existing GPU slots after100M finishes. Scientific training source and protocol hashes are preserved in every metadata-initial.json/metadata.json. No environment packages were upgraded.

## Submitted graph

| Role | SLURM job |
|---|---|
| 100M full + LoRA smoke | 24919204 |
| 500M full smoke | 24919211 |
| 100M multispecies full-parameter, six tasks | 24919261 |
| 500M multispecies full-parameter, six tasks | 24919263 |
| 100M full fresh replay + independent MLM audit | 24919264 |
| 500M full fresh replay + independent MLM audit | 24919265 |
| Fixed primary development readout | 24919266 |
| 100M multispecies LoRA, two single-seed corpus tasks | 24919267 |
| LoRA fresh replay + independent MLM audit | 24919268 |
| Development including explicitly exploratory LoRA pilot | 24919269 |

Array tasks0/2/4 are baculovirus seeds42/43/44;1/3/5 are general seeds42/43/44. LoRA array0/1 uses onlyseed42. Each full array initially has concurrency1. Each replay waits for its size's complete array; LoRA waits for both full replay jobs. CPU primary readout waits for both full arrays; final readout waits for primary readout and LoRA replay. This dependency graph bounds GPU concurrency at two. All arrays are afterok-gated and wrappers require successful smoke markers; failures must be investigated before resubmission.

Remote root: /home/shouyu/projects/def-aucoin/shouyu/virus-genome-extrapolation. Local WSL root: /home/s34yan/projects/260308_virus-genome-extrapolation. SSH shared socket: /home/s34yan/.ssh/nibi-codex.sock; host shouyu@nibi.alliancecan.ca. Run commands from the project root. Heavy computation uses SLURM only.

## State and outputs

Full states: remote runs/2026-10-24/scientific/SIZE-METHOD-CONDITION-seedSEED/. Each run contains original before embeddings, history.json, metadata-initial.json, final metadata.json/COMPLETE, and budget-512/1024/2048/4096 directories with selected_state.pt, after embeddings, selection.json/COMPLETE. Selected state files use atomic replacement and hard links so later training cannot overwrite earlier budget states. Full method saves all weights; LoRA saves only adapter matrices and requires the pinned original source plus the same rank/alpha/target modules for reload. Final root selected_state.pt is the best selected state over4096steps, not necessarily the endpoint. Optimizer state is not retained; these are fixed trajectories rather than resumable training checkpoints.

Replay JSON files: runs/2026-10-24/fresh-replay-100m-full.json, fresh-replay-500m-full.json, fresh-replay-100m-lora.json. They record state hashes/replay differences and separate-locus audit-panel identities, masks, groups and loss at original/allbudget selected states. The audit never changes selection.

CPU reports: runs/2026-10-24/development-primary/ and development-with-pilot/. Readout selection is the previously frozen mean/C=.01; no retuning. Both include metrics.json, fitted scalers/logistic coefficients, per-window validation predictions and COMPLETE. Final analysis refuses missing primary runs. The final pilot panel contains59 readouts (two unadapted NT references,48 full-adaptation budget/seed combinations,8 LoRA budget/corpus combinations,one k-mer).

## Collection without large weights

Use rsync with include directories, *.json, *.npz, COMPLETE; exclude everything else. Copy remote scientific/ to local runs/2026-10-24/collected/. Copy the desired development folder to the corresponding local folder and fresh-replay JSON files to local runs/2026-10-24/. Never confuse smoke directories with scientific runs. Preserve failed attempts and any explicitly excluded directories. SLURM logs stay in remote runs/2026-10-24/.

Render with the existing local ece657d Python environment:

```bash
python scripts/summarize_adaptation_duration.py \
  --root runs/2026-10-24 \
  --development runs/2026-10-24/development-with-pilot \
  --output documents/adaptation-2026-10-24
```

For a primary-only report substitute development-primary. The summary needs compact collected histories and complete development metrics; it never reads downstream test data. Curves/table refer to original multispecies pretraining explicitly. Report the 4096endpoint and planned4096-minus512 contrast rather than choosing the best downstream budget.

Check queue with squeue -u shouyu; diagnose specific jobs with scontrol show job JOBID and sacct. Capture completed accounting in UTC with TZ=UTC sacct and start/end/elapsed/allocation/memory/exitcode fields. JobArrayTaskLimit and Dependency are expected for this graph. Initial ReqNodeNotAvail/UnavailableNodes:u3 resolved without changing the requests; both smoke jobs then ran successfully. No monitor or recurring automation is installed.

Preserve local unrelated scripts/run_training.slurm and remote untracked logs/, run_training.slurm, scripts/mmseqs_cluster_30id.slurm and old slurm outputs.

The final dependent CPU report job runs scripts/finalize_adaptation_duration.slurm and creates documents/adaptation-2026-10-24/results/ only after final readout succeeds. Its job ID is appended to jobs-submitted.txt. It checks every source and selected-state hash, all replay evidence, panel identity, unchanged train/validation hashes, and paired token counts before writing COMPLETE. It requires no agent to be awake and performs no external messaging. Figures are regenerated locally after compact collection.
