#!/usr/bin/env python3
"""Sequential exact/double Orlin and external-library comparison; stdlib only."""

import argparse
import csv
from fractions import Fraction
import hashlib
import glob
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from instances import load, evaluate, write_sfm


def run(command, timeout):
    env = dict(
        os.environ,
        JULIA_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    start = time.perf_counter()
    p = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, env=env
    )
    if p.returncode not in (0, 2):
        raise RuntimeError(
            f"exit {p.returncode}: {p.stderr[-2000:]} {p.stdout[-1000:]}"
        )
    lines = [line for line in p.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError("solver produced no JSON result")
    result = json.loads(lines[-1])
    result["exit_code"] = p.returncode
    result["process_seconds"] = time.perf_counter() - start
    result["stderr"] = p.stderr[-2000:]
    return result


def validate(d, result):
    samples = result["samples"]
    if not samples:
        raise ValueError("empty measurements")
    vals = []
    for sample in samples:
        value = evaluate(d, sample["selected"])
        if Fraction(str(sample["minimum"])) != value:
            raise ValueError("reported objective differs from returned set")
        if (
            "returned_value" in sample
            and Fraction(str(sample["returned_value"])) != value
        ):
            raise ValueError("library return value differs from returned set")
        if sample.get("iteration_limit_reached"):
            raise ValueError("library iteration limit reached")
        if d["known"] is not None and value != d["known"]:
            raise ValueError(f"known optimum mismatch: {value} != {d['known']}")
        vals.append(value)
    if len(set(vals)) != 1:
        raise ValueError("repeats disagree")
    if result["exit_code"] != 0:
        raise ValueError("solver reported failure")
    return vals[0]


def mq(s):
    return "'" + str(s).replace("'", "''") + "'"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("instances", nargs="+", type=Path)
    p.add_argument("--cpp", type=Path, default=Path("build/orlin_benchmark"))
    p.add_argument("--julia", default="julia")
    p.add_argument("--matlab", default="matlab")
    p.add_argument("--sfo-dir", type=Path)
    p.add_argument(
        "--solvers",
        nargs="+",
        choices=["orlin", "orlin_double", "julia", "matlab"],
        default=["orlin", "orlin_double", "julia", "matlab"],
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--orlin-absolute-tolerance", type=float, default=1e-10)
    p.add_argument("--orlin-relative-tolerance", type=float, default=1e-12)
    p.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="per solver process, includes startup and warmups",
    )
    p.add_argument(
        "--max-n",
        type=int,
        default=256,
        help="explicitly increase for dense large-instance experiments",
    )
    p.add_argument("--output", type=Path, default=Path("results/crosscheck.csv"))
    a = p.parse_args()
    if (
        a.repeats < 1
        or a.warmups < 1
        or a.epsilon <= 0
        or a.timeout <= 0
        or a.orlin_absolute_tolerance < 0
        or a.orlin_relative_tolerance < 0
    ):
        p.error(
            "positive repeats/warmups/epsilon/timeout and nonnegative Orlin tolerances required"
        )
    if "matlab" in a.solvers and not a.sfo_dir:
        p.error("--sfo-dir must point to the extracted SFO toolbox")
    folder = Path(__file__).resolve().parent
    a.output.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for pattern in a.instances:
        matches = sorted(glob.glob(str(pattern)))
        if not matches:
            p.error(f"no instance matches {pattern}")
        paths.extend(Path(x) for x in matches)
    rows = []
    raw = []
    failed = False
    for path in paths:
        d = load(path)
        values = {}
        with tempfile.TemporaryDirectory(prefix="sfm_") as tmp:
            canonical = Path(tmp) / "instance.sfm"
            write_sfm(d, canonical)
            for solver in a.solvers:
                arithmetic = (
                    "Boost cpp_rational"
                    if solver == "orlin"
                    else "IEEE 754 binary64" if solver == "orlin_double" else "Float64"
                )
                row = dict(
                    instance=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    n=d["n"],
                    solver=solver,
                    arithmetic=arithmetic,
                    known=d["known"],
                    status="",
                    minimum="",
                    median_seconds="",
                    detail="",
                )
                if d["n"] > a.max_n:
                    row.update(
                        status="SKIPPED_SIZE", detail=f"exceeds --max-n {a.max_n}"
                    )
                    rows.append(row)
                    failed = True
                    continue
                if solver in ("orlin", "orlin_double"):
                    backend = "exact" if solver == "orlin" else "double"
                    cmd = [
                        str(a.cpp.resolve()),
                        "--instance",
                        str(canonical),
                        "--backend",
                        backend,
                        "--repeats",
                        str(a.repeats),
                        "--warmups",
                        str(a.warmups),
                        "--absolute-tolerance",
                        str(a.orlin_absolute_tolerance),
                        "--relative-tolerance",
                        str(a.orlin_relative_tolerance),
                    ]
                elif solver == "julia":
                    cmd = [
                        a.julia,
                        f"--project={folder}",
                        "--threads=1",
                        str(folder / "compare_julia.jl"),
                        str(canonical),
                        str(a.repeats),
                        str(a.warmups),
                        str(a.epsilon),
                    ]
                else:
                    expression = f"addpath({mq(folder)}); compare_matlab({mq(canonical)},{mq(a.sfo_dir.resolve())},{a.repeats},{a.warmups},{a.epsilon:.17g});"
                    cmd = [a.matlab, "-singleCompThread", "-batch", expression]
                try:
                    result = run(cmd, a.timeout)
                    raw.append(dict(instance=str(path), solver=solver, result=result))
                    value = validate(d, result)
                    values[solver] = value
                    samples = result["samples"]
                    seconds = sorted(x["elapsed_seconds"] for x in samples)
                    import statistics

                    row.update(
                        status=(
                            "PASS_KNOWN" if d["known"] is not None else "UNCERTIFIED"
                        ),
                        minimum=value,
                        median_seconds=statistics.median(seconds),
                    )
                except subprocess.TimeoutExpired:
                    row.update(status="TIMEOUT", detail=f"{a.timeout}s process budget")
                    failed = True
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as e:
                    row.update(status="FAILED", detail=str(e))
                    failed = True
                rows.append(row)
                print(f'{path.name} {solver}: {row["status"]}', flush=True)
            if len(set(values.values())) > 1:
                failed = True
                for row in rows:
                    if row["instance"] == str(path) and row["solver"] in values:
                        row["status"] = "DISAGREEMENT"
        # Checkpoint after each dataset, including failures and skips.
        with a.output.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)
        a.output.with_suffix(".json").write_text(
            json.dumps(
                dict(
                    platform=platform.platform(),
                    python=platform.python_version(),
                    settings={k: str(v) for k, v in vars(a).items()},
                    results=raw,
                ),
                indent=2,
            )
            + "\n"
        )
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
