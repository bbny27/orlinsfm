"""Exact adapter checks and runner failure handling; no Julia/MATLAB needed."""
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'benchmarks'))
import run_crosscheck as runner
from instances import load,evaluate,write_sfm

class CrosscheckTests(unittest.TestCase):
    def test_small_optima(self):
        for path in [root/'datasets/small_cut.sfm',root/'datasets/small_coverage.sfm',
                     root/'tests/data/tiny.max',
                     root/'datasets/varied/concave_n10.sfm',
                     root/'datasets/varied/matching_3x3.sfm',
                     root/'datasets/varied/facility_4x6.sfm']:
            d=load(path)
            best=min(evaluate(d,[i for i in range(d['n']) if mask>>i&1]) for mask in range(1<<d['n']))
            self.assertEqual(best,d['known'])
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'copy.sfm';write_sfm(d,p);self.assertEqual(d,load(p))
    def test_dimacs_terminal_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.max'
            p.write_text('p max 4 7\nn 4 s\nn 2 t\na 4 1 5\na 1 3 7\na 3 2 6\na 4 2 2\na 2 1 99\na 3 4 99\na 1 1 88\n')
            d=load(p)
            for s in [[],[0],[1],[0,1]]:
                nodes={4}|{[1,3][i] for i in s}
                arcs=[(4,1,5),(1,3,7),(3,2,6),(4,2,2),(2,1,99),(3,4,99),(1,1,88)]
                self.assertEqual(evaluate(d,s),sum(c for u,v,c in arcs if u in nodes and v not in nodes))
    def test_known_failure_not_hidden(self):
        d=load(root/'datasets/small_cut.sfm')
        r=dict(exit_code=0,samples=[dict(selected=[],minimum=d['offset'])])
        with self.assertRaises(ValueError): runner.validate(d,r)
    def test_invalid_set(self):
        d=load(root/'datasets/small_cut.sfm')
        for s in [[-1],[5],[1,1]]:
            with self.assertRaises(ValueError): evaluate(d,s)
    def test_reported_value_checked(self):
        d=dict(n=1,kind='cut',offset=0,unary=[0],arcs=[],features=[],
               scale=0,right_n=0,matching=[],known=0)
        with self.assertRaises(ValueError): runner.validate(d,dict(exit_code=0,samples=[dict(selected=[],minimum=1)]))
    def test_exit_code_preserved(self):
        p=subprocess.CompletedProcess([],2,'notice\n{"samples":[]}\n','')
        with patch.object(subprocess,'run',return_value=p):self.assertEqual(runner.run(['dummy'],1)['exit_code'],2)
    def test_timeout_and_missing_solver_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'results.csv'
            argv=['runner',str(root/'datasets/small_cut.sfm'),'--solvers','orlin','julia','--output',str(out)]
            with patch.object(sys,'argv',argv),patch.object(runner,'run',side_effect=[subprocess.TimeoutExpired('dummy',1),FileNotFoundError('julia')]):
                with self.assertRaises(SystemExit): runner.main()
            self.assertIn('TIMEOUT',out.read_text());self.assertIn('FAILED',out.read_text())
    def test_size_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'r.csv'
            with patch.object(sys,'argv',['runner',str(root/'datasets/small_cut.sfm'),'--solvers','orlin','--max-n','1','--output',str(out)]),patch.object(runner,'run') as call:
                with self.assertRaises(SystemExit):runner.main()
                call.assert_not_called()
            self.assertIn('SKIPPED_SIZE',out.read_text())

if __name__=='__main__':unittest.main()
