#!/usr/bin/env python3
"""Create an induced rectangular DIMACS crop and its exact integer .sol file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def read_header(path: Path) -> tuple[int, int, int, int]:
    nodes = declared_arcs = source = sink = None
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields or fields[0] == "c":
                continue
            if fields[:2] == ["p", "max"] and len(fields) == 4:
                if nodes is not None:
                    raise ValueError("more than one DIMACS problem line")
                nodes, declared_arcs = map(int, fields[2:])
            elif fields[0] == "n" and len(fields) == 3:
                if fields[2] == "s":
                    source = int(fields[1])
                elif fields[2] == "t":
                    sink = int(fields[1])
            elif fields[0] not in {"a"}:
                raise ValueError(
                    f"unsupported row at line {line_number}: {line.rstrip()}"
                )
    if None in (nodes, declared_arcs, source, sink):
        raise ValueError("DIMACS file needs p max, n ... s, and n ... t rows")
    if source == sink or not (1 <= source <= nodes and 1 <= sink <= nodes):
        raise ValueError("invalid source/sink declarations")
    return nodes, declared_arcs, source, sink


def induced_arcs(path: Path, retained: dict[int, int], nodes: int, declared_arcs: int):
    arcs: list[tuple[int, int, int]] = []
    seen_arcs = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields or fields[0] != "a":
                continue
            if len(fields) != 4:
                raise ValueError(f"invalid arc row at line {line_number}")
            tail, head, capacity = map(int, fields[1:])
            seen_arcs += 1
            if not (1 <= tail <= nodes and 1 <= head <= nodes):
                raise ValueError(
                    f"arc endpoint outside node range at line {line_number}"
                )
            if not (0 <= capacity <= (1 << 63) - 1):
                raise ValueError(
                    f"capacity outside signed 64-bit range at line {line_number}"
                )
            if tail in retained and head in retained:
                arcs.append((retained[tail], retained[head], capacity))
    if seen_arcs != declared_arcs:
        raise ValueError(
            f"problem line declares {declared_arcs} arcs but file has {seen_arcs}"
        )
    return arcs


def solve(nodes: int, arcs: list[tuple[int, int, int]]) -> int:
    try:
        import numpy as np
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import maximum_flow
    except ImportError as error:
        raise RuntimeError(
            "creating .sol requires NumPy and SciPy; run: "
            "python -m pip install numpy scipy"
        ) from error

    if arcs:
        rows = np.fromiter((tail - 1 for tail, _, _ in arcs), dtype=np.int64)
        columns = np.fromiter((head - 1 for _, head, _ in arcs), dtype=np.int64)
        capacities = np.fromiter((capacity for _, _, capacity in arcs), dtype=np.int64)
        graph = coo_matrix(
            (capacities, (rows, columns)), shape=(nodes, nodes), dtype=np.int64
        ).tocsr()
    else:
        graph = coo_matrix((nodes, nodes), dtype=np.int64).tocsr()
    return int(maximum_flow(graph, 0, 1, method="dinic").flow_value)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="source DIMACS .max file"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output prefix (writes PREFIX.max and PREFIX.sol)",
    )
    parser.add_argument(
        "--grid-width",
        required=True,
        type=int,
        help="width of the source row-major pixel grid",
    )
    parser.add_argument(
        "--grid-height",
        type=int,
        help="height; inferred from nonterminal count if omitted",
    )
    parser.add_argument(
        "--first-pixel-id",
        type=int,
        default=3,
        help="DIMACS ID of pixel (0,0), default 3",
    )
    parser.add_argument(
        "--x", required=True, type=int, help="zero-based left coordinate"
    )
    parser.add_argument(
        "--y", required=True, type=int, help="zero-based top coordinate"
    )
    parser.add_argument("--crop-width", required=True, type=int)
    parser.add_argument("--crop-height", required=True, type=int)
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing output files"
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    source_path = arguments.input.resolve()
    if not source_path.is_file():
        raise ValueError(f"input does not exist: {source_path}")
    if (
        arguments.grid_width <= 0
        or arguments.crop_width <= 0
        or arguments.crop_height <= 0
    ):
        raise ValueError("grid and crop dimensions must be positive")
    if arguments.x < 0 or arguments.y < 0 or arguments.first_pixel_id < 1:
        raise ValueError("coordinates must be nonnegative and first pixel ID positive")

    nodes, declared_arcs, source, sink = read_header(source_path)
    grid_height = arguments.grid_height
    if grid_height is None:
        pixel_count = nodes - 2
        if pixel_count <= 0 or pixel_count % arguments.grid_width:
            raise ValueError(
                "cannot infer grid height; supply --grid-height explicitly"
            )
        grid_height = pixel_count // arguments.grid_width
    if grid_height <= 0:
        raise ValueError("grid height must be positive")
    if (
        arguments.x + arguments.crop_width > arguments.grid_width
        or arguments.y + arguments.crop_height > grid_height
    ):
        raise ValueError("crop rectangle is outside the source grid")

    last_pixel_id = arguments.first_pixel_id + arguments.grid_width * grid_height - 1
    if last_pixel_id > nodes:
        raise ValueError("declared grid pixel IDs exceed the DIMACS node count")
    if source in range(arguments.first_pixel_id, last_pixel_id + 1) or sink in range(
        arguments.first_pixel_id, last_pixel_id + 1
    ):
        raise ValueError("terminal ID overlaps the declared pixel-ID range")

    retained = {source: 1, sink: 2}
    next_id = 3
    for row in range(arguments.y, arguments.y + arguments.crop_height):
        for column in range(arguments.x, arguments.x + arguments.crop_width):
            old_id = arguments.first_pixel_id + row * arguments.grid_width + column
            retained[old_id] = next_id
            next_id += 1

    arcs = induced_arcs(source_path, retained, nodes, declared_arcs)
    output_nodes = len(retained)

    prefix = arguments.output
    if prefix.suffix in {".max", ".sol"}:
        prefix = prefix.with_suffix("")
    max_path = Path(str(prefix) + ".max")
    sol_path = Path(str(prefix) + ".sol")
    if not arguments.force:
        existing = [str(path) for path in (max_path, sol_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing file(s): " + ", ".join(existing)
            )
    optimum = solve(output_nodes, arcs)
    max_path.parent.mkdir(parents=True, exist_ok=True)
    with max_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "c Induced rectangular crop; boundary arcs omitted; "
            "solution computed with SciPy integer Dinic\n"
        )
        stream.write(
            f"c source={source_path.name} x={arguments.x} y={arguments.y} "
            f"width={arguments.crop_width} height={arguments.crop_height}\n"
        )
        stream.write(f"p max {output_nodes} {len(arcs)}\n")
        stream.write("n 1 s\nn 2 t\n")
        for tail, head, capacity in arcs:
            stream.write(f"a {tail} {head} {capacity}\n")
    with sol_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("c SciPy integer Dinic optimum for the induced crop\n")
        stream.write(f"s {optimum}\n")

    print(f"wrote {max_path} ({output_nodes} nodes, {len(arcs)} arcs)")
    print(f"wrote {sol_path} (maximum flow {optimum})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileExistsError) as error:
        print(f"create_crop: {error}", file=sys.stderr)
        raise SystemExit(1)
