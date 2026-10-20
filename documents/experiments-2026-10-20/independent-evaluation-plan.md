# Independent evaluation and exposure policy proposal

Prepared October 20, 2026. This is a plan, not a claim that an independent cohort has been assembled or that adaptation is authorized.

## What is available and what it can establish

The current curated annotations originate from Chen et al. (2013), which maps AcMNPV transcription starts. Reusing its supplemental tables is the same annotation source, not an external cohort. [Original study](https://pmc.ncbi.nlm.nih.gov/articles/PMC3648101/).

A concrete candidate for independent annotation validation is Boldogkői et al. (2018): AcMNPV long-read data, including cap-selected reads, mapped to KM667940.1, with ENA projects PRJEB24943 and PRJEB27818. This could test annotation robustness across studies after reference-coordinate reconciliation. It is still AcMNPV, so it cannot establish unseen-virus transfer. The study uses a recombinant virus; altered regions require explicit exclusion or reconciliation. [Data descriptor](https://www.nature.com/articles/sdata2018276).

Xue et al. (2012) is a BmNPV transcriptome lead, not a verified TSS benchmark. Its abstract establishes RNA-seq transcriptome analysis, not a ready genome-wide base-resolution start-site truth set. Do not label expression coverage or annotated translation starts as experimental TSS. Full annotation/supplement review is still required. [Original study](https://pubmed.ncbi.nlm.nih.gov/22532689/).

A targeted literature search in this session did not verify a ready, compatible genome-wide TSS dataset from a different baculovirus. That is a search outcome, not evidence none exists. Individual-gene start-site studies are too selected and sparse to substitute for a whole-genome cohort without a stated sampling design.

## Admission requirements before any external scoring

Record exact genome accession/version, strain and reference sequence; study and assay provenance; host/cell and observation conditions; strand; coordinate conventions; experimentally supported starts and confidence criteria. Reconcile coordinates against the actual study reference rather than assuming accession equivalence. Freeze mapping and ambiguous-region exclusions before predictions.

Build windows under a declared common rule. The present positives have variable start offsets; a centered external dataset would change the task. Evaluate a matched window rule first, and identify any alternate task separately. Unknown/unobserved starts cannot be treated as experimentally confirmed negatives. Specify assay coverage and negative ascertainment; use context-matched negatives or explicit context strata in a future protocol to address the CDS asymmetry documented here. Do not retrofit clean_v2 or score newly built cohorts within this round.

Preserve genome/ORF/overlap groups. For multiple genomes, hold out entire genomes and define treatment of related genomes before fitting. Record positive/negative counts and effective genomic units. Select all settings internally; score the frozen external cohort once. Multiple conditions on the same genomic sequence are not independent sequence replicates.

## Proposed corpus policy for a future unseen-genome claim

Keep accession, version, source-genome hash, parent accession and coordinates for every adaptation window. Normalize accession versions for identity checks and retain exact versions for reproducibility. Assign partitions at the genome or declared related-genome group level before creating overlapping windows. Never randomly partition overlapping windows across training and validation.

Exclude every downstream evaluation genome and alternative records of the same genome from the adaptation corpus. Exclude the complete declared related-genome group for a strict unseen-group claim; create those groups with a documented taxonomy/whole-genome similarity method fixed before adaptation. Do not choose a similarity cutoff from downstream performance. Report results separately if a future protocol intentionally permits target-genome exposure; such a protocol evaluates adaptation with target exposure.

The historical corpus's 44 NC_001623.1 windows make it ineligible as-is for a target-excluded claim. Unknown foundation-model pretraining exposure remains a separate limitation even after adapting on a filtered corpus. Repair positional-token budgeting and accession-aware validation before any adaptation restart.

## Matched human comparison, if retained as a project objective

First harmonize the scientific task: annotated TSS presence in a 300-base oriented window is not automatically comparable to an existing promoter/non-promoter benchmark. Record window offsets, negative ascertainment, label evidence, chromosome/locus grouping and sampling prevalence for both domains. Use matched training counts and the same declared readout/pooling protocol; assess sensitivity to sample size without consulting external tests. A performance gap would still combine biological and annotation/distribution differences. Architecture and pretraining corpus effects require separately matched checkpoints; the current NTv1-human-reference versus NTv2-multispecies pair does not isolate either factor.

## Next decision

Prioritize annotation reconciliation and negative-sampling design over additional adaptation. The 2018 AcMNPV study is an actionable annotation-validation lead. An unseen-virus claim remains contingent on obtaining a compatible different-genome truth set. No external predictions, bulk sequencing downloads, dataset replacement or adaptation were performed in this round.
