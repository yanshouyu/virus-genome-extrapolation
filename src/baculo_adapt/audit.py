"""Secondary MLM audit independent of checkpoint-selection loci."""


def independent_audit_rows(validation_rows, selection_rows, count=256, seed=2718):
    """Exclude entire selection loci, including opposite orientations."""
    import random
    def locus(row):
        return row['accession'], row['start'], row['end']
    excluded = {locus(row) for row in selection_rows}
    eligible = [row for row in validation_rows if locus(row) not in excluded]
    if len(eligible) < count:
        raise ValueError('Insufficient independent audit windows')
    return random.Random(seed).sample(eligible, count)
