"""Inventory and independent certificates for the varied benchmark data."""

import csv
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from instances import evaluate, load


def scipy_cut_certificate(instance):
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import maximum_flow

    ground_n = instance["n"]
    feature_n = len(instance["features"])
    source, sink = ground_n + feature_n, ground_n + feature_n + 1
    rows, columns, capacities = [], [], []
    constant = instance["offset"]

    def arc(tail, head, capacity):
        if capacity:
            rows.append(tail); columns.append(head); capacities.append(capacity)

    for item, weight in enumerate(instance["unary"]):
        if weight >= 0:
            arc(item, sink, weight)
        else:
            constant += weight
            arc(source, item, -weight)
    for tail, head, capacity in instance["arcs"]:
        arc(tail, head, capacity)

    finite_sum = sum(capacities) + sum(weight for weight, _ in instance["features"])
    infinity = finite_sum + 1
    for feature_index, (weight, items) in enumerate(instance["features"]):
        feature = ground_n + feature_index
        arc(feature, sink, weight)
        for item in items:
            arc(item, feature, infinity)

    shape = ground_n + feature_n + 2
    graph = coo_matrix(
        (np.asarray(capacities, dtype=np.int64), (rows, columns)),
        shape=(shape, shape), dtype=np.int64).tocsr()
    return constant + int(maximum_flow(graph, source, sink, method="dinic").flow_value)


class VariedDatasetTests(unittest.TestCase):
    def test_catalog_and_embedded_references(self):
        with (ROOT / "datasets/varied/catalog.csv").open(
                newline="", encoding="utf-8") as stream:
            catalog = list(csv.DictReader(stream))
        files = sorted((ROOT / "datasets/varied").glob("*.sfm"))
        self.assertEqual(len(catalog), len(files))
        by_name = {Path(row["file"]).name: row for row in catalog}
        self.assertEqual(set(by_name), {path.name for path in files})
        for path in files:
            instance = load(path)
            row = by_name[path.name]
            self.assertEqual(instance["n"], int(row["n"]))
            self.assertEqual(instance["known"], int(row["optimum"]))

    def test_concave_certificates(self):
        for path in (ROOT / "datasets/varied").glob("concave_*.sfm"):
            instance = load(path)
            ordered = sorted(instance["unary"])
            values = [instance["offset"]]
            prefix = instance["offset"]
            for cardinality, weight in enumerate(ordered, 1):
                prefix += weight
                values.append(prefix - instance["scale"] * cardinality *
                              (cardinality - 1) // 2)
            self.assertEqual(min(values), instance["known"], path.name)

    def test_matching_certificates(self):
        for path in (ROOT / "datasets/varied").glob("matching_*.sfm"):
            instance = load(path)
            optimum = min(
                evaluate(instance, [i for i in range(instance["n"])
                                    if mask & (1 << i)])
                for mask in range(1 << instance["n"]))
            self.assertEqual(optimum, instance["known"], path.name)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy is optional")
    def test_graph_reduction_certificates(self):
        for path in (ROOT / "datasets/varied").glob("*.sfm"):
            instance = load(path)
            if instance["kind"] in ("cut", "coverage"):
                self.assertEqual(scipy_cut_certificate(instance),
                                 instance["known"], path.name)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy is optional")
    def test_all_small_tsukuba_crops(self):
        for side in range(1, 13):
            path = ROOT / "datasets/scaling" / f"tsukuba_crop_{side}x{side}.max"
            self.assertTrue(path.is_file(), path.name)
            instance = load(path)
            self.assertEqual(instance["n"], side * side)
            self.assertEqual(scipy_cut_certificate(instance),
                             instance["known"], path.name)


if __name__ == "__main__":
    unittest.main()
