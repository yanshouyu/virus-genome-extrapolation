# October 24 duration study — running checkpoint

Authorized window: 08:57:11–10:57:11 Toronto (12:57:11–14:57:11 UTC). Source/protocol and scientific jobs have been saved and submitted. This is a progress checkpoint, not a completed scientific result.

Read [the fixed protocol](adaptation-2026-10-24/protocol.md), [verification amendment](adaptation-2026-10-24/verification-amendment.md), and [execution/collection guide](adaptation-2026-10-24/execution.md).

Both original-source and real GPU smoke checks passed, including full100M/500M and100M LoRA. All-source MLM parameter equality, finite gradients, changed trainable weights, selected-state reload, and LoRA frozen-weight identity are checked. LoRA starts at the exact same smoke loss as full100M, trains360448parameters vs97889132, with peak allocated smoke memory5.057GB vs6.882GB. These short-smoke figures do not establish scientific performance or full-study compute savings. Full500M smoke peak19.067GB.

Primary full-parameter arrays24919261/24919263 run both multispecies sizes, both corpora, seeds42/43/44 through4096steps. At most two GPU jobs are active. Follow-on replays24919264/65, CPU primary readout24919266, LoRA pilot24919267, LoRA replay24919268 and final CPU readout24919269 are submitted with dependencies. Scientific paths and ordering are in the execution guide. All before/after comparisons use the existing fixed readout. The real downstream test remains excluded.

At this snapshot the first scientific tasks are training normally and have crossed the512-step extraction/restore boundary. No adapted downstream outcome from the new study has been used to alter the protocol. Full local suite:29passed2skipped, followed by a separate fifth verification test and one synthetic end-to-end analysis test. Training source and original protocol hashes match local/Nibi records. Independent audit lives in a separate module so the scientific training-source hashes remain identical across tasks.

Next: collect finished primary/pilot outputs; verify allreplay jobs; run/report independent MLM audit and frozen-readout budget comparison. If100M array and replay are finished while500M remains, raising500M array throttle to2 is permitted only after confirming no other GPU job is active. This reallocates existing capacity without increasing the two-job bound. Leave pending/running jobs intact at the deadline, report exact states, and do not claim completion from submitted dependencies. No monitor is installed.

A final CPU reporting job also waits for24919269 and generates a checked Markdown report at documents/adaptation-2026-10-24/results/. Source/protocol/data/replay identity gates must pass before its COMPLETE marker is written. This allows scientific jobs to finish after the chat without requiring an agent or an automation. Collect and inspect its outputs on the next user prompt.
