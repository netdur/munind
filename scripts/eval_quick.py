#!/usr/bin/env python3
import subprocess, re, h5py, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
truth = np.array(h5py.File(ROOT/'benches/data/glove-100-angular.hdf5','r')['neighbors'])[:,:10]
print(f'Ground truth: {truth.shape[0]} queries, top-10')

for eps in [0.1, 0.2, 0.4]:
    r = subprocess.run([str(ROOT/'target/release/munind'),'search','-n','10','-e',str(eps),
        str(ROOT/'benches/indexes/glove-100-angular-munind'),
        str(ROOT/'benches/data/glove-100-angular.test.tsv')],
        capture_output=True, text=True)
    lines = r.stdout.splitlines()
    results, cur = [], []
    for l in lines:
        m = re.match(r'^\s*(\d+)\s+(\d+)\s+', l)
        if m:
            rank, idx = int(m.group(1)), int(m.group(2))
            if rank==1 and cur: results.append(cur); cur=[]
            cur.append(idx)
            if len(cur)==10: results.append(cur); cur=[]
    if cur: results.append(cur)
    found = np.array(results)-1
    hits = sum(len(set(found[i].tolist())&set(truth[i].tolist())) for i in range(found.shape[0]))
    recall = hits/(found.shape[0]*10)
    avg_re = re.search(r'Average query time:\s*([0-9.]+)', r.stderr)
    avg_ms = float(avg_re.group(1)) if avg_re else 0
    print(f'-e {eps}  recall@10: {recall:.6f}  avg_query_ms: {avg_ms:.6f}')
