# October24 study: submitted, running

The longer adaptation experiment and bounded LoRA pilot are now executing on Nibi. This page is a status entry, not a completed results claim. The server will create [the final checked report](results/README.md) only after all14 scientific runs, independent replay/audit and fixed-readout analysis pass. Until results/COMPLETE exists, treat the study as unfinished.

Primary panel: NTv2 100M (multispecies) and NTv2 500M (multispecies), full-parameter MLM on baculovirus/general corpora, seeds42/43/44,4096steps. Fixed prefix budgets512/1024/2048/4096; all downstream settings remain frozen. Original AcMNPV data is unchanged and excluded from adaptation; no downstream test inference is run. The exploratory LoRA pilot uses100M(multispecies), query/value rank8, seed42, both corpora and a fixed separate learning rate.

[Protocol](protocol.md), [verification amendment](verification-amendment.md), [job graph and collection guide](execution.md), and [progress handoff](../handoff-adaptation-2026-10-24-progress.md) provide exact details. The queued jobs produce compact predictions, readout coefficients, checkpoint provenance, source checks, MLM curves, independent audit and a final report. Full model states stay on Nibi.

GPU smoke checks passed for100M full,500M full and100M LoRA. LoRA zero-update loss matches the original model exactly; its adapters change while original weights remain fixed. Local regression suite passed29tests with2optional skips; additional audit and synthetic end-to-end analysis checks passed. These engineering checks do not establish the scientific outcome.
