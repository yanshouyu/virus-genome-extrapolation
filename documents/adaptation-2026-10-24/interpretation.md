# Interpretation of the completed October 24 adaptation study

Baculovirus adaptation is useful for the **NTv2 500M (multispecies)** regulatory-window probe under this fixed regime. At the planned 4,096-step budget, its validation log loss, MCC and ROC AUC improve over both the unadapted model and the equally budgeted general continued-training control, with conditional paired 95% intervals excluding zero. Extending the baculovirus trajectory beyond its 512-step prefix improves log loss and AUC; its additional MCC gain remains uncertain. The corresponding **NTv2 100M (multispecies)** effects are smaller and unresolved. Neither size establishes superiority over the frozen k-mer reference on this small cohort.

The language-model adaptation question is sufficiently resolved to close this experiment: all six full baculovirus trajectories improve on a separate-locus audit, and all meet the predeclared practical plateau rule. This is evidence of meaningful adaptation and diminishing returns under the tested schedule, rather than proof of universal convergence or a requirement that adaptation always help. The smallest next deliverable is a bounded AcMNPV-focused final report, using these results and the existing data/model audits. No new training, embedding extraction, or downstream test inference was performed for this analysis.

## Evidence and independent checks

The checked server evidence remains intact in [results/](results/README.md). The completed compact scientific outputs, both development analyses, fitted readout coefficients and replay records were collected from Nibi; model weights remain there. The local analysis used the completed corpus, replacing the incomplete early collection snapshot for these purposes.

The new [verification](analysis/verification.json) independently checks all 14 scientific runs, 59 readouts and 66 accounting rows. Both six-task full arrays and the two-task pilot are present; every recorded job/step completed with exit code 0:0. Eleven distinct scientific/evaluation source paths match their saved hashes; protocol, amendment, frozen readout and unchanged train/validation data hashes match. Checkpoint choices at all four budgets and the final-1,024-step plateau calculations reproduce from histories. Paired full-corpus input/masked-token counts match. Each run processed 33,685,504 input tokens at its maximum budget.

All saved validation probabilities replay from collected embeddings and saved scaler/logistic coefficients within **4.44e-16** maximum absolute difference. Scaler means and population standard deviations match **training-only** features for every readout; all seven stored metrics recompute. The original log-loss/MCC paired bootstrap intervals reproduce to 1e-12. The new analysis adds AUC intervals and supplementary frozen k-mer comparisons using the same resamples. This coefficient replay is separate from the already completed server fresh-process embedding replay (maximum difference 0.0); no GPU replay was repeated locally. Large checkpoint-file hashes are supported by the retained server verification rather than rehashed locally without weights.

The 256-window independent MLM panel was regenerated from saved held-out windows with seed 2718. Its exact identifiers, group counts and sequence-inclusive panel hash match all three replay records; no selection locus appears in either orientation. Audit CSV losses match the replay JSON. The [audit panel](analysis/audit-panel.json) retains identities and the common saved mask hash; masks/losses were not recomputed with a model locally. The panel contains nine held-out species groups, shared with the selection corpus, and therefore adds loci rather than independent biological replication.

## Was the adaptation budget adequate?

![Selection and separate-locus audit losses](analysis/mlm-selection-audit.png)

Each line is one full-parameter seed; all checkpoints originate from multispecies NTv2. Upper panels show every selection-panel evaluation, including the actual 4,096-step endpoint. Lower panels show the original model and the MLM-selected state available at each budget on the fixed audit panel. The two panels use different loci/masks, so their absolute loss levels are not interchangeable. General audit lines are flat because the selected state remains the same, rather than because every later state was audited.

| Model (original scope: multispecies) | Seed | Initial selection loss | Selected step at budget 4096 | Selected loss | Actual endpoint loss | Final-1024 best-loss gain | Audit loss, original -> selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| NTv2 100M | 42 | 8.103702 | 3584 | 6.791215 | 6.791229 | 0.0316% | 8.223145 -> 6.901613 |
| NTv2 100M | 43 | 8.103702 | 4096 | 6.790113 | 6.790113 | 0.0432% | 8.223145 -> 6.904797 |
| NTv2 100M | 44 | 8.103702 | 4096 | 6.790505 | 6.790505 | 0.0209% | 8.223145 -> 6.904985 |
| NTv2 500M | 42 | 7.223035 | 4096 | 6.479429 | 6.479429 | 0.1394% | 7.369603 -> 6.561136 |
| NTv2 500M | 43 | 7.223035 | 4096 | 6.473786 | 6.473786 | 0.1513% | 7.369603 -> 6.563913 |
| NTv2 500M | 44 | 7.223035 | 4096 | 6.473745 | 6.473745 | 0.1303% | 7.369603 -> 6.565888 |

All six improve cumulative-best selection loss by less than the predeclared 0.5% during steps 3072–4096. Actual endpoints closely match the selected states, so cumulative-best reporting is not hiding substantial late deterioration in the baculovirus arms. The independent audit improves at every successive budget for each baculovirus seed. Audit losses at budgets 512/1024/2048/4096 are approximately 6.95/6.94/6.92/6.90 for 100M and 6.80/6.73/6.63/6.56 for 500M. [All per-seed, per-budget selection, endpoint and audit values](analysis/mlm-all-seeds.md) retain the exact differences, including the pilot.

General continued training also improves viral MLM loss initially. However, its selected steps remain early: 128/128/256 for 100M and 256/128/128 for 500M, in seed order 42/43/44. At the 4,096 endpoint, selection loss deteriorates from approximately 6.96 to 6.98–6.99 for 100M and 6.87 to 6.90 for 500M. The same early checkpoint is selected at every reported budget, making all general downstream duration changes exactly zero. This is a property of selection by viral MLM loss; it does not show that general-domain learning stopped, nor measure downstream performance of the unselected endpoint. General selected audit losses remain about 7.07 and 6.98 respectively, worse than the corresponding baculovirus states.

Together, selection and audit losses support actual viral-language adaptation beyond the repeatedly consulted panel. Nevertheless, cosine decay to a near-zero learning rate contributes to late flattening, and the audit has no 3072-step state. The plateau is conditional on this schedule, corpus and panel, not an independent audit-plateau test. It supplies a reasonable stopping point for this experiment; it cannot guarantee that a changed schedule would yield no further improvement. Language-model loss alone also cannot establish improved regulatory prediction: 100M adapts clearly in MLM while downstream changes remain uncertain.

## Does it improve the fixed regulatory-window probe?

The downstream split remains **233 train / 77 validation / 78 test**, from one AcMNPV genome. Only train/validation were opened here. Masked-mean pooling, logistic C=.01 for NT, the frozen k-mer C, and probability >.5 remain unchanged. Training seeds 42/43/44 are not independent genomes. Adapted scores below are means +/- sample SD of seed-specific scores, not an ensemble or confidence interval.

| Model / original pretraining scope | Additional training (4096 maximum steps) | Log loss | MCC | ROC AUC |
|---|---|---:|---:|---:|
| NTv2 100M (multispecies) | None | 0.477 | 0.584 | 0.848 |
| NTv2 100M (multispecies) | General | 0.471 +/- 0.003 | 0.541 +/- 0.030 | 0.856 +/- 0.005 |
| NTv2 100M (multispecies) | Baculovirus | 0.452 +/- 0.005 | 0.627 +/- 0.040 | 0.864 +/- 0.001 |
| NTv2 500M (multispecies) | None | 0.524 | 0.506 | 0.824 |
| NTv2 500M (multispecies) | General | 0.491 +/- 0.012 | 0.558 +/- 0.026 | 0.841 +/- 0.006 |
| NTv2 500M (multispecies) | Baculovirus | 0.341 +/- 0.010 | 0.724 +/- 0.053 | 0.934 +/- 0.010 |
| Oriented 1–4-mer logistic | None | 0.449 | 0.565 | 0.882 |

![Fixed development outcomes versus budget](analysis/development-duration.png)

Thin lines show individual adaptation seeds; error bars show sample SD, not cohort uncertainty. Lower log loss and higher MCC/AUC are better. No budget or seed is selected using these downstream curves. The 4,096 budget and within-trajectory 4,096-minus-512 effects were planned. The new 512-step prefix uses a 4,096-step cosine schedule restarted from pretrained weights; it is not the October 23 independently decayed 512-step experiment and does not compare independently optimized short/long schedules.

![Planned endpoint effects and supplementary k-mer reference](analysis/endpoint-effects.png)

Intervals use 2,000 paired group-bootstrap resamples of 64 validation groups, averaging seed-specific metrics within conditions. They condition on the fitted models and this cohort, omit full training/selection uncertainty, and do not represent new genomes. All are unadjusted development intervals. The k-mer effects and newly added AUC intervals are supplementary fixed comparisons; they did not change selection. [Complete effect table](analysis/paired-effects.md) and [all readout scores with conditional intervals](analysis/readout-scores.csv) provide the values.

For **500M (multispecies)** at the planned endpoint, baculovirus minus general changes are log loss **-0.1498 [-0.2536, -0.0611]**, MCC **+0.1664 [+0.0169, +0.3131]**, and AUC **+0.0930 [+0.0360, +0.1648]**. Corresponding comparisons against unadapted 500M also exclude zero for all three outcomes. These concordant results support a domain-specific benefit beyond the specified general continued-training control, under the accepted sampling/grouping policy.

Within the 500M baculovirus trajectory, 4096 minus 512 changes are log loss **-0.0873 [-0.1564, -0.0329]**, MCC **+0.0691 [-0.0267, +0.1575]**, and AUC **+0.0507 [+0.0148, +0.0988]**. Longer adaptation therefore benefits probability quality and ranking, while the incremental fixed-threshold decision benefit remains unresolved. This sharpens the October 23 short-budget conclusion, without retrospectively making that earlier study conclusive.

For **100M (multispecies)**, all these endpoint and duration intervals cross zero. Its 4096-minus-512 log-loss change is -0.0138 and MCC change is -0.0176; the metrics do not support a consistent larger-budget improvement. It would be misleading to use its clear MLM reduction as proof of a regulatory-task gain, or to call these unresolved intervals evidence of no effect.

The practical **k-mer comparison remains unresolved**. For 500M baculovirus minus k-mer, log loss changes by **-0.1076 [-0.2200, +0.0042]**, MCC by **+0.1593 [-0.0254, +0.3708]**, and AUC by **+0.0520 [-0.0230, +0.1356]**. All point estimates favor adapted 500M, but all intervals cross zero. For 100M, log loss is slightly worse (+0.0038), MCC better (+0.0623), and AUC worse (-0.0180), with all intervals crossing zero. These observations support 500M adaptation as a promising option, while leaving a clear cost-adjusted advantage over the inexpensive reference unproven. No k-mer training-cost measurement was added.

## Exploratory LoRA feasibility, kept separate

The pilot uses only **NTv2 100M (multispecies), seed 42**, both corpora, query/value rank 8, alpha 16, zero dropout, frozen original parameters including the head/embeddings, and learning rate 1e-4 versus full training at 1e-5. Each run pays the full 4096-step budget; checkpoint selection remains viral MLM only.

| Corpus / method (NTv2 100M multispecies, seed 42) | Trainable parameters | Peak GPU GB | Training / program / allocated H100 min | Selected step | Audit loss | Log loss / MCC / AUC |
|---|---:|---:|---:|---:|---:|---|
| Baculovirus / full | 97,889,132 | 6.882 | 11.08 / 11.36 / 11.88 | 3584 | 6.9016 | 0.453 / 0.584 / 0.864 |
| Baculovirus / LoRA | 360,448 | 5.057 | 10.36 / 10.64 / 11.35 | 3328 | 6.9734 | 0.448 / 0.688 / 0.877 |
| General / full | 97,889,132 | 6.882 | 11.08 / 11.31 / 12.38 | 128 | 7.0689 | 0.473 / 0.558 / 0.855 |
| General / LoRA | 360,448 | 5.057 | 10.44 / 10.67 / 11.25 | 256 | 7.0524 | 0.489 / 0.610 / 0.849 |

![Exploratory paired resource and performance comparison](analysis/pilot-resource-performance.png)

LoRA trains **0.367% of its 98,249,580 total parameters**, approximately 272 times fewer trainable parameters than full training. Peak allocated GPU memory falls by about **26.5%**. Training-section time falls only around 6%; allocated job time falls approximately 4.5% for baculovirus and 9.2% for general. These are observed full-H100 allocation minutes, not monetary prices or portable speedup estimates; startup and allocation overhead vary. [Resource evidence](analysis/pilot-resources.json) distinguishes training-section, program and whole-job time. Training time includes scheduled evaluation/save overhead; replay and downstream-analysis allocations are separate.

Baculovirus LoRA improves selection loss from 8.1037 to 6.8642 (endpoint 6.8645), and audit loss from 8.2231 to 6.9734. Audit losses at budgets 512/1024/2048/4096 are 6.9984/6.9878/6.9793/6.9734. It adapts, although its viral MLM loss remains worse than paired full training. General LoRA selects step 256, loss 6.9450 versus endpoint 6.9627; its selected audit loss is 7.0524 at every budget. The lower MLM loss method does not automatically give the better task point estimate.

Baculovirus LoRA minus paired full seed 42 changes log loss by **-0.0044 [-0.0562, +0.0455]**, MCC by **+0.1041 [-0.0087, +0.2383]**, and AUC by **+0.0128 [-0.0360, +0.0670]**. General LoRA yields worse log-loss/AUC point estimates despite higher MCC. All paired pilot intervals include zero. The pilot establishes implementation feasibility and measured memory savings; one seed, unequal learning rates and no equal tuning budget prevent claims of superiority, equivalence, noninferiority or necessity. No formal LoRA search is warranted merely to complete this scoped study.

## Boundaries of the claim

- Both primary models were originally pretrained on **multispecies** genomes. This study cannot attribute transfer limits to human-only pretraining. The earlier frozen NTv1 2.5B multispecies result is an exploratory reference from another generation/pretraining history; it was not adapted and does not isolate model size.
- Adaptation excludes AcMNPV NC_001623.1 under the existing target-name/duplicate exclusions. Species-grouped viral MLM validation and the separate-locus audit improve control, but related species remain and the audit shares nine held-out groups.
- The general control matches maximum steps, strand-paired locus exposure and paired token budgets. Its yeast/worm/fly composition and diversity differ from baculovirus and the original multispecies mixture. This comparison tests these domain corpora, not a specific promoter mechanism. Different MLM-selected exposure is part of the fixed selection procedure; it does not compare equal selected training durations.
- Downstream evidence concerns one genome, a small cohort and the accepted group/strand/sampling policy. Positives overlapping another gene's CDS remain accepted. Data were not regenerated and settled overlap decisions were not reopened.
- Repeated development evaluation, unadjusted comparisons and conditional intervals limit confirmatory interpretation. Training-seed stability does not supply independent biological replication. No universal necessity, cross-baculovirus generalization, mechanistic discovery, or broad model-size claim follows.
- The existing 78-window test was inspected in earlier project history. It remains excluded from prediction here and must not be called a pristine independent external test later.

## Concrete closure recommendation

**Close the adaptation experiment and assemble the final AcMNPV report/manuscript.** The bounded question now has a defensible answer: longer baculovirus adaptation improves 500M probability quality and ranking, and the fixed endpoint outperforms the specified general control; 100M and k-mer-superiority questions remain unresolved. Integrate the existing dataset/exposure audit, repaired model/source fidelity, frozen-readout rationale, these four figures and their limitations. Rewrite the original aspirational framing around the executed controlled probing study. Attribution, additional models, datasets and training sweeps are optional future work, not prerequisites for this deliverable.

The current evidence is sufficient for a development-only scoped report. If a final internal test evaluation is later requested, first freeze this exact panel: unadapted NTv2 100M/500M (multispecies); all three full-training baculovirus and general seeds at the **4096 maximum budget**, with each state already selected only by viral MLM; and the frozen oriented 1–4-mer readout. Apply the existing fitted training-only scaler/logistic coefficients, frozen C/pooling, and >.5 threshold to the 78 test windows once, report seed-specific and mean metrics with group-aware intervals, and make no test-driven revision. Keep LoRA separately exploratory. Document the cohort's historical inspection and describe it as a historically exposed internal evaluation. That is a separate authorization/decision, and no implementation or inference was launched here.

An independently annotated additional species would be necessary to support a broader cross-baculovirus claim; it is not required to finish this AcMNPV-focused report. More optimization alone would not resolve uncertainty caused by the small cohort.

## Reproduction and artifacts

Run `OPENBLAS_NUM_THREADS=1 /home/s34yan/miniconda3/envs/ece657d/bin/python scripts/interpret_adaptation_duration.py` from the repository root after compact collection described in [execution.md](execution.md). The script reads only saved training/validation features, probabilities, coefficients, MLM histories/audit and accounting; it never loads weights or opens downstream test data. It writes derived artifacts under `analysis/`, preserving `results/` as raw checked evidence. Four figures are available in PNG and exportable PDF; all rendered PNGs were visually inspected for labels, overlap and clipping. Complete per-seed metrics, effects, audit identities and verification accompany the report.
