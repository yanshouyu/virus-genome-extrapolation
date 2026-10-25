# October 24 study: all jobs completed

All 14 scientific runs, replay/audit, analysis and final report jobs completed successfully. [Checked results](results/README.md) and [verification](results/verification.json) are available locally and on Nibi. The next work is scientific interpretation and figures; follow [the saved new-chat instructions](../handoff-adaptation-2026-10-24-next-chat.md).

The earlier running-status text below is a historical snapshot.

The longer adaptation experiment and bounded LoRA pilot are now executing on Nibi. This page is a status entry, not a completed results claim. The server will create [the final checked report](results/README.md) only after all14 scientific runs, independent replay/audit and fixed-readout analysis pass. Until results/COMPLETE exists, treat the study as unfinished.

Primary panel: NTv2 100M (multispecies) and NTv2 500M (multispecies), full-parameter MLM on baculovirus/general corpora, seeds42/43/44,4096steps. Fixed prefix budgets512/1024/2048/4096; all downstream settings remain frozen. Original AcMNPV data is unchanged and excluded from adaptation; no downstream test inference is run. The exploratory LoRA pilot uses100M(multispecies), query/value rank8, seed42, both corpora and a fixed separate learning rate.

[Protocol](protocol.md), [verification amendment](verification-amendment.md), [job graph and collection guide](execution.md), and [progress handoff](../handoff-adaptation-2026-10-24-progress.md) provide exact details. The queued jobs produce compact predictions, readout coefficients, checkpoint provenance, source checks, MLM curves, independent audit and a final report. Full model states stay on Nibi.

GPU smoke checks passed for100M full,500M full and100M LoRA. LoRA zero-update loss matches the original model exactly; its adapters change while original weights remain fixed. The final local regression suite passed 31 tests with 2 optional skips, including the independent-audit and synthetic end-to-end analysis checks. These engineering checks do not establish the scientific outcome.

At the early handoff, the first 100M full baculovirus run (seed 42) has completed all 4,096 steps and exact checkpoint reloads. Its best selection loss is 6.7912 at step 3,584; the last 1,024 steps improved cumulative-best loss by 0.0316%, meeting the practical plateau rule for this run. This is not a downstream performance result or a conclusion across seeds. Other primary runs are active or queued by the deliberate array limit; verification, pilot and report jobs wait on dependencies. See jobs-at-handoff.psv for the timestamped snapshot and jobs-submitted.txt for all identifiers.
