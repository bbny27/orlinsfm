"""Optional independent dataset verification: pip install scipy numpy.
Checks checksums, every published optimum, and derived crop optima via Dinic.
This tests datasets, not performance of the three generic solvers.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow
root=Path(__file__).resolve().parents[1]/'datasets'
for item in json.loads((root/'manifest.json').read_text()):
    path=root/item['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],path
    aa=np.array([list(map(int,line.split()[1:])) for line in path.read_text().splitlines() if line.startswith('a ')],dtype=np.int64)
    graph=coo_matrix((aa[:,2],(aa[:,0]-1,aa[:,1]-1)),shape=(item['nodes'],item['nodes'])).tocsr()
    result=maximum_flow(graph,0,1,method='dinic')
    solution=int(next(line.split()[1] for line in path.with_suffix('.sol').read_text().splitlines() if line.startswith('s ')))
    assert result.flow_value==solution==item['optimum'],path
    print(path.name,solution,'PASS')
