"""Regression test for tools/create_crop.py; skips when SciPy is unavailable."""

from pathlib import Path
import importlib.util
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CropToolTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("scipy") is not None,
        "SciPy is optional",
    )
    def test_induced_crop_and_solution(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "grid.max"
            rows = ["p max 14 32", "n 1 s", "n 2 t", "c 4 by 3 grid"]
            arcs = []
            for pixel in range(3, 15):
                arcs.extend([(1, pixel, pixel - 2), (pixel, 2, 2)])
            for pixel in range(3, 14):
                arcs.append((pixel, pixel + 1, 3))
                if len(arcs) == 32:
                    break
            self.assertEqual(len(arcs), 32)
            rows.extend(f"a {tail} {head} {capacity}" for tail, head, capacity in arcs)
            source.write_text("\n".join(rows) + "\n", encoding="utf-8")

            prefix = folder / "crop"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/create_crop.py"),
                    "--input",
                    str(source),
                    "--output",
                    str(prefix),
                    "--grid-width",
                    "4",
                    "--grid-height",
                    "3",
                    "--x",
                    "1",
                    "--y",
                    "1",
                    "--crop-width",
                    "2",
                    "--crop-height",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            max_rows = [
                line.split()
                for line in prefix.with_suffix(".max").read_text().splitlines()
                if line and not line.startswith("c ")
            ]
            self.assertEqual(max_rows[0][:3], ["p", "max", "6"])
            crop_arcs = [tuple(map(int, row[1:])) for row in max_rows if row[0] == "a"]
            self.assertTrue(
                all(1 <= u <= 6 and 1 <= v <= 6 and c >= 0 for u, v, c in crop_arcs)
            )

            optimum = min(
                sum(
                    capacity
                    for tail, head, capacity in crop_arcs
                    if tail in ({1} | {3 + i for i in range(4) if mask & (1 << i)})
                    and head not in ({1} | {3 + i for i in range(4) if mask & (1 << i)})
                )
                for mask in range(1 << 4)
            )
            solution = int(
                next(
                    line.split()[1]
                    for line in prefix.with_suffix(".sol").read_text().splitlines()
                    if line.startswith("s ")
                )
            )
            self.assertEqual(solution, optimum)


if __name__ == "__main__":
    unittest.main()
