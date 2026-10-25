# October 24 study: analysis complete

[Read the interpreted report](interpretation.md). All 14 scientific runs and 59 readouts are complete. Local independent coefficient/scaler/metric checks passed; existing paired intervals reproduce, and the separate-locus audit panel regenerates exactly. No new training or downstream test inference was launched.

NTv2 500M (multispecies) baculovirus adaptation improves planned endpoint log loss, MCC and AUC versus both general continued training and its unadapted reference. Longer adaptation improves log loss and AUC; incremental MCC remains uncertain. NTv2 100M (multispecies) effects and superiority over k-mers remain unresolved. The single-seed NTv2 100M (multispecies) LoRA pilot demonstrates feasibility and lower memory, with no established performance superiority.

The next deliverable is final AcMNPV-focused report/manuscript assembly. Additional training is not needed to state the bounded result. A historically exposed internal test evaluation is a separate decision, with the exact proposed frozen comparison documented in the interpretation.

- [Raw checked server report](results/README.md), [server verification](results/verification.json), [completed job accounting](jobs-completed.psv).
- [Independent local analysis verification](analysis/verification.json), [paired effects including AUC/k-mers](analysis/paired-effects.md), [all per-seed MLM states](analysis/mlm-all-seeds.md), [all readout scores and conditional intervals](analysis/readout-scores.csv).
- [MLM selection/audit figure](analysis/mlm-selection-audit.pdf), [development duration figure](analysis/development-duration.pdf), [endpoint effects figure](analysis/endpoint-effects.pdf), [exploratory pilot figure](analysis/pilot-resource-performance.pdf).
- [Protocol](protocol.md), [verification amendment](verification-amendment.md), [execution/collection guide](execution.md).

Scientific raw evidence is preserved under results/. Derived analysis and figures are under analysis/. The early running-status handoff is historical; use this index and the interpretation for current status. Model weights remain on Nibi. No new recurring monitoring was installed.
