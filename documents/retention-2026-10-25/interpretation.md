# Does baculovirus adaptation preserve general NT benchmark performance?

**NTv2 500M (multispecies) preserves overall benchmark performance within the agreed margin, but preservation on every individual task is not established.** Its category-balanced mean MCC increases from **0.5795 to 0.5923**, a change of **+0.0128 [95% interval +0.0028, +0.0227]**. The one-sided 95% lower bound is **+0.0043**, comfortably above the predeclared allowable decline of -0.02 MCC.

The changes are not uniform. Promoter, enhancer and splice family means increase; the histone family mean decreases slightly. Four histone tasks prevent the stronger simultaneous all-18-task preservation claim. Their observed MCC declines are approximately 0.008–0.018, and none of the 18 task point estimates loses more than 0.05. Nevertheless, their uncertainty bounds do not rule out a decline exceeding that task-level limit.

This supports **useful specialization with broadly retained performance under the tested probing procedure**, with small task-specific costs and unresolved limits. It does not support calling the adapted checkpoint universally better or claiming that every task is preserved.

## Overall and family results

All models below are **NTv2 500M, originally pretrained on multispecies genomes**. Adapted entries average the scores of the three existing adaptation seeds, rather than averaging their predictions into an ensemble. The overall score first averages tasks within each family, then averages the four families equally; ten histone tasks therefore do not dominate the result.

| Benchmark family | Tasks | Original MCC | Baculovirus MCC | General-control MCC | Baculovirus minus original [paired 95% interval] |
|---|---:|---:|---:|---:|---:|
| Promoter | 3 | 0.7359 | 0.7721 | 0.7613 | +0.0363 [+0.0054, +0.0676] |
| Enhancer | 2 | 0.4855 | 0.4948 | 0.4762 | +0.0093 [-0.0095, +0.0282] |
| Splice | 3 | 0.5694 | 0.5826 | 0.6081 | +0.0132 [+0.0029, +0.0232] |
| Histone | 10 | 0.5270 | 0.5195 | 0.5172 | -0.0075 [-0.0161, +0.0010] |
| **Category-balanced overall** | **18** | **0.5795** | **0.5923** | **0.5907** | **+0.0128 [+0.0028, +0.0227]** |

![Family and overall retention](results/family-retention.png)

Intervals are paired genomic-block intervals, conditional on the fitted models, selected readouts and benchmark cohort. The -0.02 reference line is the **overall** acceptance margin; no separate family-level noninferiority threshold was declared.

The overall baculovirus scores are 0.5916, 0.5938 and 0.5914 for seeds 42/43/44. General-control scores are 0.5924, 0.5892 and 0.5906. Adaptation-seed variation is small relative to the cohort uncertainty, but these are not independent biological replicates. [Every seed's MCC, log loss and AUC](results/scores.csv) is retained.

The aggregate gain needs context. The largest observed improvement is the **TATA-promoter subgroup**, whose MCC changes from 0.7101 to 0.8241: +0.1140 [0.0270, 0.2033]. That test set has only **212 windows**. Its large gain contributes strongly to the equally weighted promoter-family mean. The all-promoter and no-TATA tasks instead have small negative point changes (-0.0012 and -0.0039). Thus the positive overall result is not evidence of uniform improvement across all tasks. Individual task effects are descriptive and their ordinary intervals are unadjusted.

## The stricter per-task requirement remains unresolved

We agreed to tolerate at most a **0.05 absolute MCC decline per task**. At ordinary one-sided 95% confidence, 17/18 tasks meet that margin; H3K9me3 remains unresolved. To claim preservation across **all 18 tasks together**, the protocol uses more conservative, Bonferroni-adjusted one-sided bounds. Under that rule, **14/18 tasks meet the margin**, and these four remain unresolved:

| Histone task | Observed MCC change | Ordinary paired 95% interval | Simultaneous one-sided lower bound | Required lower bound |
|---|---:|---:|---:|---:|
| H3K27ac | -0.0182 | [-0.0462, +0.0083] | -0.0588 | > -0.0500 |
| H3K4me3 | -0.0114 | [-0.0419, +0.0211] | -0.0544 | > -0.0500 |
| H3K9ac | -0.0078 | [-0.0401, +0.0243] | -0.0535 | > -0.0500 |
| H3K9me3 | -0.0156 | [-0.0612, +0.0318] | -0.0803 | > -0.0500 |

These are **unresolved preservation claims**, rather than four demonstrated losses exceeding 0.05. No task meets the protocol's criterion for supported material degradation beyond that limit. Conversely, an interval that includes zero cannot establish preservation by itself; the relevant lower bound must exclude the unacceptable decline.

![All task effects](results/task-retention.png)

Filled points show mean seed-specific effects; faint rings show individual adaptation-seed effects. Bars are ordinary paired two-sided 95% intervals, which are less conservative than the simultaneous one-sided bounds in the table. [All task retention decisions and bounds](results/task-retention.csv) explain why a task can pass its marginal check while remaining unresolved for an all-task claim.

## Probability quality and ranking give a consistent, nonuniform picture

The histone concern is not merely a threshold artifact. Nine of ten histone tasks have worse log-loss and AUC point estimates after baculovirus adaptation. The changes are generally small, and most individual intervals include zero, but their direction warrants reporting. In H3K27ac, log loss increases by **+0.0096 [+0.0009, +0.0182]**, providing nominal, unadjusted evidence of a small probability-quality cost even though its MCC interval includes zero.

Enhancer and splice tasks generally improve in probability quality and ranking. For enhancer detection, log loss changes by **-0.0090 [-0.0156, -0.0025]** and AUC by **+0.0072 [+0.0022, +0.0122]**. For splice acceptors, the corresponding changes are **-0.0117 [-0.0184, -0.0050]** and **+0.0052 [+0.0015, +0.0090]**. These are supporting, unadjusted comparisons; they do not replace the primary MCC preservation rule.

[The complete paired-effect tables](results/paired-effects.md) show MCC, log loss and AUC for every task and all three planned condition comparisons. Negative log-loss changes and positive MCC/AUC changes favor the left condition.

General continued training also raises overall MCC: **+0.0113 [+0.0018, +0.0207]** versus original. Baculovirus minus general is only **+0.0016 [-0.0046, +0.0077]** overall. The general control performs better on the splice family, whereas baculovirus performs better on the enhancer family. The benchmark therefore supplies no clear overall human-task superiority attributable specifically to baculovirus exposure.

## What was evaluated

The study used the authors' prepared [revised NT benchmark on Hugging Face](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised), pinned to revision `851f9946252e90c665cdb3cc3eedb78f1f26197c`. It contains **493,242 labelled training windows and 38,822 labelled test windows** across 18 tasks. The official chromosome-held-out splits, labels and examples were preserved. Test windows come from chromosomes 20/21. The audit found no train/test coordinate overlap or exact/reverse-complement sequence matches, including across tasks. Reusing identical sequences reduced inference to 434,501 unique training sequences and 32,388 unique test sequences per backbone without removing labelled examples.

Seven frozen backbones were evaluated: the pinned original NTv2 500M multispecies checkpoint and all three baculovirus/general full-parameter seeds from the earlier study. Original model revision is `06615c1660c892fc199840c18123f8385b3542a8`. Every adapted state is the already fixed **4096-maximum-budget, viral-MLM-selected checkpoint**, with its file hash matched to the earlier verified evidence. Baculovirus selected step 4096 in all seeds; general selected steps 256/128/128. Maximum training budgets match, while selected-state exposure differs. The general control is a specified yeast/worm/fly subset, not a reconstruction of the original pretraining mixture.

Only lightweight logistic classifiers were fitted in this study. Backbone weights were frozen throughout. On the original model's training features only, three chromosome-grouped folds selected mean-pooled layer depth and regularization from the fixed grid. Candidate layers were 8/15/22/29; four tasks chose layer 8 and fourteen chose layer 15. Each task's selected layer and C were then held fixed across all seven backbones. Each scaler and classifier was independently fitted using that backbone's training features. Binary decisions use probability >0.5; multiclass decisions use argmax. No test-driven layer, C, threshold, seed or checkpoint selection occurred.

All 18 settings were saved and independently collected **before benchmark test prediction**, with a [pre-test hash record](pre-test-freeze-verification.json) and [complete frozen settings](frozen-settings-before-test.json). The [protocol](protocol.md) gives the exact grid, tie rules, pooling, precision and metrics. The [NT paper](https://www.nature.com/articles/s41592-024-02523-z) and [official repository](https://github.com/instadeepai/nucleotide-transformer) provide the benchmark context. These results evaluate a common frozen linear-probe procedure; they do not reproduce the paper's best nonlinear probing or task-fine-tuned leaderboard scores.

Uncertainty uses **20,000 paired resamples of 107 common genomic blocks**, based on chromosome/1-Mb bins with bins joined when windows cross boundaries. Common weights preserve dependence across related tasks and pair every model on the same genomic resample. Blocks approximate spatial dependence; they are not independent genomes or biological experiments. Intervals condition on the trained states, readout choices and supplied cohort and do not include all training/selection uncertainty. The 0.02/0.05 tolerances were review choices for this study, not standards established by the NT authors.

## Connection to the AcMNPV result

![Viral benefit and benchmark retention](results/benefit-retention.png)

The same baculovirus-adapted states previously increased AcMNPV validation MCC from about 0.506 to 0.724, versus about 0.558 for general continued training. The scatter shows adaptation-seed point estimates and means, not new biological-replication intervals. Its horizontal axis is **AcMNPV validation** performance; its vertical axis is the prepared NT benchmark's held-out-test performance.

The combined evidence supports a bounded claim: **the tested 500M specialization improves the AcMNPV development probe while maintaining overall performance on these human benchmark probes, with small nonuniform costs and unresolved per-task limits.** It does not establish that adaptation is necessary, that it beats k-mers reliably on AcMNPV, or that it generalizes to new baculovirus species. The old AcMNPV test remains historically exposed and was not opened or predicted in this follow-up.

Retention here is measured **after refitting a small linear classifier**. It does not guarantee compatibility with an existing task head used without refitting, preservation of every layer, or retention of long-context and other untested genomic functions. Human benchmark chromosomes may also have appeared in original unlabeled pretraining; they are held out from supervised probe fitting, not necessarily from pretraining.

## Verification, resources and project decision

All **126 readouts** passed server verification and independent local recomputation. Every raw dataset and feature-file hash matched. Full training-only means/scales checked out, and every test probability replayed exactly from its saved coefficients and features. Fresh-process checks reproduced all four layers for the checked 64-row train/test batches in every backbone with maximum difference **0.0**. Locally, genomic blocks were independently reconstructed from coordinates, all bootstrap weights regenerated, and every task/family/overall interval and simultaneous task bound reproduced. [Server verification](results/verification.json) and [local verification](local-verification.json) retain the evidence.

Total GPU allocation was **1.813 H100 hours**, including engineering smoke and replay, with at most two study GPUs active simultaneously. [Compute accounting](compute-cost.md) separates phases. The [execution record](execution.md) preserves two initial CPU module-path failures and a cancelled I/O-bound verifier attempt; its replacement changed file-read order and memory allocation only. The replacement retained the same numerical checks and reused the completed predictions. All figures were rendered locally using the existing plotting environment and visually inspected. Large feature caches, fitted classifiers and model states remain on Nibi; compact checked probabilities and summaries are retained with this report.

The retention extension is complete. I recommend assembling the final AcMNPV-focused report with this qualified finding, while keeping the original checkpoint available for general use. **If preservation on every individual task is a hard requirement, it remains uncertified by this study.** The appropriate report language is “overall preservation supported; four task-level limits unresolved,” rather than claiming universal preservation or launching more training to obtain a favorable result.
