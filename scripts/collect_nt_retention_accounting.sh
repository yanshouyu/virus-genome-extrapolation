#!/bin/bash
# Run after completed study collection; preserve all actual engineering/scientific jobs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/shouyu/projects/def-aucoin/shouyu/virus-genome-extrapolation}"
TZ=UTC sacct -P -j 25004286,25004391,25004436,25004444,25004455,25004589,25004631,25004632,25004633,25004643,25004644,25004662,25004813,25004863,25005147,25009860 --format=JobID,JobName,State,Start,End,Elapsed,ElapsedRaw,AllocCPUS,CPUTimeRAW,TotalCPU,MaxRSS,ExitCode,AllocTRES > documents/retention-2026-10-25/jobs.psv
