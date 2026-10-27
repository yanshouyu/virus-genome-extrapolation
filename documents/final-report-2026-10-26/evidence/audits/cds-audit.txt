# Positive CDS overlap audit — October 21, 2026

The broad CDS-overlap counts from October 20 did not distinguish the assigned downstream coding region from other genes. They are not sufficient to call positive windows incorrectly constructed or to justify replacing the dataset. Under the user's clarified rule, overlap with another gene is acceptable. No dataset or model output was changed.

All 203 positive windows across training, validation and test were checked against their assigned TSS/ORF in data/interim/tss_seq_range.csv and the GenBank CDS features of NC_001623.1. Test inspection was limited to sequence/annotation coordinates; no test predictions or model selection occurred. Every assigned ORF matches a GenBank CDS segment, including compound CDS parts. Every window sequence matches the oriented genome.

| Split | Positives | No assigned downstream-segment overlap | Exactly 1 bp assigned overlap | Other same-strand gene overlap |
|---|---:|---:|---:|---:|
| Train | 122 | 119 | 3 | 65 |
| Validation | 40 | 37 | 3 | 25 |
| Test | 41 | 39 | 2 | 20 |

Every positive counted in the earlier same-strand CDS-overlap totals also overlaps a different annotated gene. These overlaps are acceptable under the clarified policy. The eight 1-bp boundary cases are additional overlaps with the assigned downstream CDS; there are no larger overlaps with the specifically assigned downstream segment.

## Eight boundary cases

Coordinates below are one-based inclusive. Each window includes just the first coding nucleotide in transcriptional orientation, not the complete start codon.

| Split | Window | Assigned gene | Shared base |
|---|---|---|---:|
| Train | 63527–63826 (-) | AcOrf-75 | 63527 |
| Train | 96099–96398 (-) | AcOrf-110 | 96099 |
| Train | 133292–133591 (+) | AcOrf-154 | 133591 |
| Validation | 59811–60110 (+) | AcOrf-70 | 60110 |
| Validation | 84540–84839 (+) | AcOrf-97 | 84839 |
| Validation | 86879–87178 (-) | Ac-p6.9 | 86879 |
| Test | 44829–45128 (+) | Ac-lef10 | 45128 |
| Test | 60717–61016 (+) | Ac-IAP2 | 61016 |

The notebook's pick_range function declares one-based closed intervals. On '+' it uses ed=min(orf_st+1, st+length) and chooses region_ed from range(st,ed), which can include orf_st. On '-' it chooses region_st from range(max(orf_ed,ed-length),ed), which can include orf_ed. Both permit exactly the assigned CDS boundary base. Strictly upstream windows would require region_ed < orf_st on '+' and region_st > orf_ed on '-'. This audit documents the boundary issue without altering source notebooks, windows or splits.

## Compound CDS exception

Training window 122590–122889 (+), named 122755|122589:122889|+|1, is assigned to ORF segment 127150–128946. GenBank places that segment in compound CDS Ac-IE-01, with another coding part at 122832–122945. The window overlaps that earlier part by 58 bp (122832–122889). Thus it is upstream of the selected downstream segment but is not wholly upstream of the entire compound coding feature. It also overlaps separately annotated Ac-IE-0 and AcOrf-140. This should be distinguished from an unrelated-gene overlap and from the eight 1-bp cases. The table's assigned segment is not necessarily the translation-start segment of the complete spliced CDS.

Another training window, 122532–122831 (+), is assigned to the first Ac-IE-01 segment, ends immediately before it, and has no assigned-gene overlap.

## Conclusion and reproduction

Retain the dataset. Other-gene overlap is expected and acceptable under the user policy. The concrete findings to retain are eight 1-bp extraction-boundary exceptions and one compound-CDS annotation exception. The earlier recommendation to generate new negatives based solely on aggregate CDS overlap is withdrawn. These measurements do not establish a causal model shortcut, nor does CDS overlap alone define TSS presence; the existing negative sampling policy is preserved.

Run from the repository root with the existing Biopython environment:

```bash
/home/s34yan/miniconda3/envs/cpen_14/bin/python scripts/audit_positive_cds.py
```

[Every positive window](positive-windows.csv), [every CDS intersection](cds-overlaps.csv), and [summary with source hashes and exception identifiers](summary.json) provide the audit trail. No cluster computation was needed.
