"""Encode one canonical classroom sequence from panel.csv."""
NT="ACGT"
def rc(s):
    return s.translate(str.maketrans("ACGT","TGCA"))[::-1]

def features(s):
    n = (len(s) - 4) // 2
    stem = s[:n]
    loop = s[n:n + 4]
    gc = sum((stem.count(k) for k in 'GC'))
    a = [0.5, n, gc, gc / n, 1 / n] + [stem.count(k) for k in NT]
    a.extend((int(c == k) for c in (stem[0], stem[-1], *loop) for k in NT))
    stacks = sorted({min(x + y, rc(x + y)) for x in NT for y in NT})
    pairs = [min(stem[i:i + 2], rc(stem[i:i + 2])) for i in range(n - 1)]
    cnt = [pairs.count(z) for z in stacks]
    a += cnt
    return a
