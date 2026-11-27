"""Encode one canonical classroom sequence from panel.csv."""
AA="ACDEFGHIKLMNPQRSTVWY"

def features(s):
    n = len(s)
    a = [0.5, n] + [s.count(k) for k in AA if k != 'I']
    a += [s[:2].count(k) + s[-2:].count(k) for k in 'DEKRFWY']
    return a
