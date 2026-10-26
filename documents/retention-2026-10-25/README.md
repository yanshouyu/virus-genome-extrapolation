# NTv2 500M multispecies retention study - complete

[Read the interpreted report](interpretation.md). The approved comparison of seven frozen checkpoints on all18 revised NT tasks is complete, including126 readouts and independent statistical replay.

Overall category-balanced MCC increases0.5795 ->0.5923 after baculovirus adaptation: change+0.0128 [95% interval+0.0028,+0.0227], supporting the predeclared overall preservation limit of-.02. The simultaneous all-task criterion is unresolved:14/18 tasks meet the-.05 bound; H3K27ac, H3K4me3, H3K9ac and H3K9me3 do not exclude larger losses. Their observed declines are small, and no task shows supported degradation exceeding the margin.

- [All checked task scores](results/README.md), [paired effects for MCC/log loss/AUC](results/paired-effects.md), [every seed's scores](results/scores.csv).
- [Server verification](results/verification.json), [independent local verification](local-verification.json), [pre-test freeze](pre-test-freeze-verification.json).
- [Task figure](results/task-retention.pdf), [family figure](results/family-retention.pdf), [viral benefit / retention](results/benefit-retention.pdf).
- [Protocol](protocol.md), [execution and preserved attempts](execution.md), [compute costs](compute-cost.md), [UTC job accounting](jobs.psv).

The authors' supplied dataset and chromosome splits were preserved. No backbone training or AcMNPV test prediction was performed. The study evaluates retention after fitting lightweight task-specific readouts; it does not guarantee unchanged behavior of existing task heads or all untested capabilities. No scientific jobs remain running. The next deliverable is final report/manuscript assembly.
