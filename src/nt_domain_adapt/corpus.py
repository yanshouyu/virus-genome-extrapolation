"""Corpus preparation: GenBank parsing, windowing, tokenisation, train/val split."""
import hashlib
import re
import statistics
from pathlib import Path

from Bio import SeqIO
from datasets import Dataset, DatasetDict

_NON_ACGT_RE = re.compile(r"[^ACGTacgt]")

# NT-v2 uses 6-mer tokenisation; each window of window_nt nucleotides produces
# (window_nt // 6) 6-mer tokens plus one [CLS] token.
_KMERS = 6
_MAX_LENGTH = 1026   # (6144 // 6) + [CLS] + 1 buffer for any [EOS]


def _ambiguous_frac(seq: str) -> float:
    """Fraction of characters that are not A/C/G/T."""
    if not seq:
        return 1.0
    return len(_NON_ACGT_RE.findall(seq)) / len(seq)


def _tile_windows(
    seq_str: str,
    accession: str,
    window_nt: int,
    stride_nt: int,
) -> list[dict]:
    """Tile a sequence into overlapping windows, plus one circular wrap-around window.

    The wrap-around window straddles the genome origin (position 0) to give
    complete coverage of the circular dsDNA topology.
    """
    L = len(seq_str)
    windows: list[dict] = []

    # Regular (non-wrapping) windows
    for start in range(0, L - window_nt + 1, stride_nt):
        windows.append({
            "sequence": seq_str[start : start + window_nt].upper(),
            "accession": accession,
            "start": start,
            "end": start + window_nt,
            "is_wraparound": False,
        })

    # Wrap-around window: last (window_nt // 2) nt + first (window_nt // 2) nt
    half = window_nt // 2
    wrap_seq = seq_str[L - half :] + seq_str[: window_nt - half]
    windows.append({
        "sequence": wrap_seq.upper(),
        "accession": accession,
        "start": L - half,
        "end": window_nt - half,    # coordinate wraps past genome end
        "is_wraparound": True,
    })

    return windows


def build_corpus(
    gb_dir: str,
    window_nt: int = 6144,
    stride_nt: int = 3072,
    max_ambiguous_frac: float = 0.05,
) -> tuple[list[dict], list[dict]]:
    """Parse all GenBank files in *gb_dir* and tile them into sequence windows.

    Args:
        gb_dir:             Path to directory containing ``*.gb`` files.
        window_nt:          Window length in nucleotides.
        stride_nt:          Stride between consecutive windows.
        max_ambiguous_frac: Drop any window whose non-ACGT fraction exceeds this.

    Returns:
        Tuple of (windows, manifest) where *windows* is the list of window dicts
        and *manifest* is one entry per genome with accession/length/quality info.
    """
    gb_path = Path(gb_dir)
    all_windows: list[dict] = []
    manifest: list[dict] = []

    gb_files = sorted(gb_path.glob("*.gb"))
    if not gb_files:
        raise FileNotFoundError(f"No .gb files found in {gb_dir!r}")

    for gb_file in gb_files:
        record = SeqIO.read(str(gb_file), "genbank")
        accession = record.id
        seq_str = str(record.seq)
        L = len(seq_str)
        n_frac = _ambiguous_frac(seq_str)

        manifest.append({
            "accession": accession,
            "file": gb_file.name,
            "length": L,
            "ambiguous_frac": round(n_frac, 6),
        })

        if L < window_nt:
            print(
                f"  WARNING: {accession} skipped — genome length {L} < "
                f"window size {window_nt}"
            )
            continue

        raw_windows = _tile_windows(seq_str, accession, window_nt, stride_nt)
        kept = [w for w in raw_windows if _ambiguous_frac(w["sequence"]) <= max_ambiguous_frac]
        n_dropped = len(raw_windows) - len(kept)
        if n_dropped:
            print(
                f"  {accession}: dropped {n_dropped}/{len(raw_windows)} windows "
                f"(>{max_ambiguous_frac:.0%} ambiguous bases)"
            )

        all_windows.extend(kept)

    return all_windows, manifest


def corpus_hash(windows: list[dict]) -> str:
    """Stable SHA-256 of window coordinates, suitable for reproducibility metadata."""
    h = hashlib.sha256()
    for w in sorted(windows, key=lambda x: (x["accession"], x["start"])):
        h.update(f"{w['accession']}:{w['start']}:{w['end']}".encode())
    return h.hexdigest()


def tokenize_and_split(
    windows: list[dict],
    tokenizer,
    val_frac: float = 0.10,
    seed: int = 42,
    max_length: int = _MAX_LENGTH,
) -> DatasetDict:
    """Tokenise windows and return a train/validation :class:`DatasetDict`.

    Args:
        windows:    Output of :func:`build_corpus`.
        tokenizer:  HuggingFace tokenizer for the NT model.
        val_frac:   Fraction of windows held out for validation.
        seed:       Random seed for the split.
        max_length: Token budget per sample (default covers 6144 nt + special tokens).

    Returns:
        :class:`DatasetDict` with ``"train"`` and ``"validation"`` splits.
    """
    raw_ds = Dataset.from_dict({"sequence": [w["sequence"] for w in windows]})

    def _tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["sequence"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

    tokenized = raw_ds.map(
        _tokenize,
        batched=True,
        remove_columns=["sequence"],
        num_proc=4,
        desc="Tokenising",
    )

    # Keep only the columns the DataCollatorForLanguageModeling needs
    cols_to_keep = [c for c in ["input_ids", "attention_mask"] if c in tokenized.column_names]
    tokenized = tokenized.select_columns(cols_to_keep)

    split = tokenized.train_test_split(test_size=val_frac, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def corpus_report(
    windows: list[dict],
    manifest: list[dict],
    dataset: DatasetDict,
) -> dict:
    """Build a summary dict suitable for writing to ``corpus_report.json``."""
    sequences = [w["sequence"] for w in windows]
    gc_fracs = [(s.count("G") + s.count("C")) / len(s) for s in sequences if s]
    ambig_fracs = [_ambiguous_frac(s) for s in sequences]

    per_accession: dict[str, int] = {}
    for w in windows:
        per_accession[w["accession"]] = per_accession.get(w["accession"], 0) + 1

    return {
        "total_windows": len(windows),
        "total_tokens": len(windows) * (_MAX_LENGTH - 2),  # excludes special tokens
        "train_windows": len(dataset["train"]),
        "val_windows": len(dataset["validation"]),
        "mean_gc_frac": round(statistics.mean(gc_fracs), 4) if gc_fracs else 0.0,
        "mean_ambiguous_frac": round(statistics.mean(ambig_fracs), 6) if ambig_fracs else 0.0,
        "n_genomes": len(manifest),
        "per_accession_windows": per_accession,
        "manifest": manifest,
    }
