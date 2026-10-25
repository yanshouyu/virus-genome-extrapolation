# October 24 adaptation-duration study and LoRA pilot

All 12 full-parameter trajectories and both single-seed LoRA corpus arms completed, with independent checkpoint replay and a separate-locus MLM audit. No downstream test predictions were generated. These remain development results on one AcMNPV genome.

Of the six full baculovirus runs, 6/6 meet the predeclared practical loss-plateau rule (<0.5% cumulative-best loss improvement during the last 1,024 steps). This describes this schedule and held-out panel; it does not prove sufficient adaptation for all tasks. Inspect the independent MLM audit as well as the selection loss.

## Fixed development scores

| Model / original pretraining scope | Method | Corpus | Budget | Log loss | MCC | ROC AUC |
|---|---|---|---:|---:|---:|---:|
| NTv2 100M (multispecies) | Original frozen | None | 0 | 0.477 | 0.584 | 0.848 |
| NTv2 100M (multispecies) | Full | general | 512 | 0.471 +/- 0.003 | 0.541 +/- 0.030 | 0.856 +/- 0.005 |
| NTv2 100M (multispecies) | Full | baculo | 512 | 0.466 +/- 0.016 | 0.645 +/- 0.030 | 0.855 +/- 0.016 |
| NTv2 100M (multispecies) | Full | general | 1024 | 0.471 +/- 0.003 | 0.541 +/- 0.030 | 0.856 +/- 0.005 |
| NTv2 100M (multispecies) | Full | baculo | 1024 | 0.454 +/- 0.001 | 0.627 +/- 0.015 | 0.866 +/- 0.003 |
| NTv2 100M (multispecies) | Full | general | 2048 | 0.471 +/- 0.003 | 0.541 +/- 0.030 | 0.856 +/- 0.005 |
| NTv2 100M (multispecies) | Full | baculo | 2048 | 0.455 +/- 0.007 | 0.610 +/- 0.026 | 0.861 +/- 0.004 |
| NTv2 100M (multispecies) | Full | general | 4096 | 0.471 +/- 0.003 | 0.541 +/- 0.030 | 0.856 +/- 0.005 |
| NTv2 100M (multispecies) | Full | baculo | 4096 | 0.452 +/- 0.005 | 0.627 +/- 0.040 | 0.864 +/- 0.001 |
| NTv2 500M (multispecies) | Original frozen | None | 0 | 0.524 | 0.506 | 0.824 |
| NTv2 500M (multispecies) | Full | general | 512 | 0.491 +/- 0.012 | 0.558 +/- 0.026 | 0.841 +/- 0.006 |
| NTv2 500M (multispecies) | Full | baculo | 512 | 0.428 +/- 0.005 | 0.655 +/- 0.053 | 0.883 +/- 0.005 |
| NTv2 500M (multispecies) | Full | general | 1024 | 0.491 +/- 0.012 | 0.558 +/- 0.026 | 0.841 +/- 0.006 |
| NTv2 500M (multispecies) | Full | baculo | 1024 | 0.412 +/- 0.022 | 0.697 +/- 0.030 | 0.897 +/- 0.012 |
| NTv2 500M (multispecies) | Full | general | 2048 | 0.491 +/- 0.012 | 0.558 +/- 0.026 | 0.841 +/- 0.006 |
| NTv2 500M (multispecies) | Full | baculo | 2048 | 0.359 +/- 0.009 | 0.717 +/- 0.045 | 0.923 +/- 0.001 |
| NTv2 500M (multispecies) | Full | general | 4096 | 0.491 +/- 0.012 | 0.558 +/- 0.026 | 0.841 +/- 0.006 |
| NTv2 500M (multispecies) | Full | baculo | 4096 | 0.341 +/- 0.010 | 0.724 +/- 0.053 | 0.934 +/- 0.010 |
| Oriented 1-4-mer logistic | Logistic | None | 0 | 0.449 | 0.565 | 0.882 |

Adapted full entries are mean +/- sample SD across seeds 42/43/44. All NT readouts use the previously frozen masked-mean pooling and C=.01, trained on 233 windows and evaluated on 77 validation windows. K-mers retain their frozen C. Threshold>.5; no retuning.

## Planned endpoint effects

| Planned endpoint comparison | Log-loss change [95% interval] | MCC change [95% interval] |
|---|---:|---:|
| 100m-budget4096-baculo-minus-general | -0.0189 [-0.0814, +0.0483] | +0.0864 [-0.0253, +0.2065] |
| 100m-budget4096-baculo-minus-before | -0.0243 [-0.0969, +0.0530] | +0.0434 [-0.0996, +0.1996] |
| 100m-baculo-4096-minus-512 | -0.0138 [-0.0510, +0.0262] | -0.0176 [-0.0845, +0.0401] |
| 100m-general-4096-minus-512 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| 500m-budget4096-baculo-minus-general | -0.1498 [-0.2536, -0.0611] | +0.1664 [+0.0169, +0.3131] |
| 500m-budget4096-baculo-minus-before | -0.1832 [-0.3495, -0.0356] | +0.2176 [+0.0194, +0.4326] |
| 500m-baculo-4096-minus-512 | -0.0873 [-0.1564, -0.0329] | +0.0691 [-0.0267, +0.1575] |
| 500m-general-4096-minus-512 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

Negative log-loss changes and positive MCC changes favor the left condition. Intervals use 2,000 paired resamples of 64 validation groups and condition on these trained models/data. Intervals spanning zero leave the effect unresolved. Multiple duration comparisons are descriptive and unadjusted; do not choose a winning downstream budget.

## Exploratory LoRA feasibility

| Model / original pretraining scope | Method | Corpus | Seed | Log loss | MCC | ROC AUC |
|---|---|---|---:|---:|---:|---:|
| NTv2 100M (multispecies) | full, 4096-step budget | general | 42 | 0.473 | 0.558 | 0.855 |
| NTv2 100M (multispecies) | lora, 4096-step budget | general | 42 | 0.489 | 0.610 | 0.849 |
| NTv2 100M (multispecies) | full, 4096-step budget | baculo | 42 | 0.453 | 0.584 | 0.864 |
| NTv2 100M (multispecies) | lora, 4096-step budget | baculo | 42 | 0.448 | 0.688 | 0.877 |

LoRA uses query/value rank8, alpha16, no dropout and frozen original weights including embeddings/MLM head. Its fixed lr 1e-4 differs from full lr 1e-5. One seed and no equal hyperparameter search prevent claims of superiority or noninferiority. Compare paired seed42 only and report actual compute/memory in training-summary.csv.

## Interpretation limits

The new trajectories restart from original pinned pretrained weights with cosine decay over 4,096 steps. Budget 512 is a prefix of that schedule, not the October 23 separately decayed512-step run or an independently optimized short schedule. Selection uses only fixed viral MLM loss; the extra 256-window MLMaudit excludes both orientations of selection loci and never changes selection. These extra windows share held-out species groups and are not new biological replication.

Approximate MLM saturation plus improved downstream performance would support usefulness under the tested regime; it cannot establish universal necessity. Baculovirus-vs-general and comparisons with k-mers remain essential. The existing downstream test was inspected in historical project work and is not a pristine external cohort.

## Evidence

verification.json records source/selection/replay/data identity checks. development-metrics.json contains every fixed readout and paired comparison. independent-mlm-audit.csv and fresh-replay JSON files retain audit losses, panel identities and embedding checks. run-metadata.json/histories.json retain all scientific provenance. Full state files remain in runs/2026-10-24/scientific/ on Nibi. Figures can be regenerated locally with scripts/summarize_adaptation_duration.py after collecting compact outputs.
