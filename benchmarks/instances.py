"""Common integer objective and exact verification, outside timed solver calls."""

from pathlib import Path


def load(path):
    path = Path(path)
    rows = [
        s.split()
        for s in path.read_text().splitlines()
        if s.strip() and not s.lstrip().startswith(("c", "#"))
    ]
    header = rows[0]
    if header[:2] == ["p", "max"]:
        n, m = map(int, header[2:])
        terminals = {r[2]: int(r[1]) for r in rows if r[0] == "n"}
        source, sink = terminals["s"], terminals["t"]
        if source == sink:
            raise ValueError("identical terminals")
        ids = {
            v: i
            for i, v in enumerate(v for v in range(1, n + 1) if v not in (source, sink))
        }
        d = dict(
            n=n - 2,
            kind="cut",
            offset=0,
            unary=[0] * (n - 2),
            arcs=[],
            features=[],
            scale=0,
            right_n=0,
            matching=[],
            known=None,
        )
        arcs = [tuple(map(int, r[1:])) for r in rows if r[0] == "a"]
        if len(arcs) != m:
            raise ValueError("DIMACS arc count mismatch")
        for u, v, c in arcs:
            if not (1 <= u <= n and 1 <= v <= n and c >= 0):
                raise ValueError("invalid arc")
            if u == v or u == sink or v == source:
                continue
            if u == source:
                d["offset"] += c
                if v != sink:
                    d["unary"][ids[v]] -= c
            elif v == sink:
                d["unary"][ids[u]] += c
            else:
                d["arcs"].append((ids[u], ids[v], c))
        solution = path.with_suffix(".sol")
        if solution.exists():
            d["known"] = int(
                next(
                    r.split()[1]
                    for r in solution.read_text().splitlines()
                    if r.startswith("s ")
                )
            )
    else:
        if (
            len(header) not in (3, 4)
            or header[0] != "p"
            or header[1] not in ("cut", "coverage", "concave", "matching")
            or (header[1] == "matching") != (len(header) == 4)
        ):
            raise ValueError("invalid SFM header")
        n = int(header[2])
        right_n = int(header[3]) if header[1] == "matching" else 0
        if n <= 0 or (header[1] == "matching" and right_n <= 0):
            raise ValueError("invalid SFM dimensions")
        d = dict(
            n=n,
            kind=header[1],
            offset=0,
            unary=[0] * n,
            arcs=[],
            features=[],
            scale=0,
            right_n=right_n,
            matching=[],
            known=None,
        )
        has_scale = False
        for r in rows[1:]:
            if r[0] == "u":
                i, value = int(r[1]), int(r[2])
                if not 0 <= i < n:
                    raise ValueError("invalid unary item")
                d["unary"][i] += value
            elif r[0] == "a":
                if d["kind"] != "cut":
                    raise ValueError("arc in non-cut instance")
                u, v, c = map(int, r[1:])
                if not (0 <= u < n and 0 <= v < n and c >= 0):
                    raise ValueError("invalid arc")
                d["arcs"].append((u, v, c))
            elif r[0] == "f":
                if d["kind"] != "coverage":
                    raise ValueError("feature in non-coverage instance")
                weight, items = int(r[1]), list(map(int, r[2:]))
                if weight < 0 or not items or any(not 0 <= i < n for i in items):
                    raise ValueError("invalid feature")
                d["features"].append((weight, items))
            elif r[0] == "q":
                if d["kind"] != "concave" or has_scale or len(r) != 2:
                    raise ValueError("invalid concavity row")
                d["scale"] = int(r[1])
                has_scale = True
                if d["scale"] < 0:
                    raise ValueError("negative concavity scale")
            elif r[0] == "e":
                if d["kind"] != "matching" or len(r) != 3:
                    raise ValueError("invalid matching edge row")
                left, right = int(r[1]), int(r[2])
                if not (0 <= left < n and 0 <= right < right_n):
                    raise ValueError("invalid matching edge")
                d["matching"].append((left, right))
            elif r[0] == "offset":
                d["offset"] = int(r[1])
            elif r[0] == "o":
                d["known"] = int(r[1])
            else:
                raise ValueError("unknown SFM row")
        if d["kind"] == "concave" and not has_scale:
            raise ValueError("concave instance has no q row")
    # All objective sums must be exactly representable in Float64, including
    # intermediate sums, so the independently re-evaluated integer value is meaningful.
    bound = (
        abs(d["offset"])
        + sum(map(abs, d["unary"]))
        + sum(a[2] for a in d["arcs"])
        + sum(f[0] for f in d["features"])
        + d["scale"] * d["n"] * max(0, d["n"] - 1) // 2
        + min(d["n"], d["right_n"])
    )
    if bound >= 2**52:
        raise ValueError("objective magnitude too large for this Float64 comparison")
    return d


def evaluate(d, selected):
    if len(set(selected)) != len(selected) or any(
        type(i) is not int or not 0 <= i < d["n"] for i in selected
    ):
        raise ValueError("invalid returned set")
    s = set(selected)
    value = (
        d["offset"]
        + sum(d["unary"][i] for i in s)
        + sum(c for u, v, c in d["arcs"] if u in s and v not in s)
        + sum(w for w, items in d["features"] if any(i in s for i in items))
    )
    if d.get("kind") == "concave":
        value -= d["scale"] * len(s) * (len(s) - 1) // 2
    elif d.get("kind") == "matching":
        adjacency = [[] for _ in range(d["n"])]
        for left, right in d["matching"]:
            adjacency[left].append(right)
        matched = [-1] * d["right_n"]

        def augment(left, seen):
            for right in adjacency[left]:
                if seen[right]:
                    continue
                seen[right] = True
                if matched[right] < 0 or augment(matched[right], seen):
                    matched[right] = left
                    return True
            return False

        value += sum(augment(left, [False] * d["right_n"]) for left in s)
    return value


def write_sfm(d, path):
    with Path(path).open("w") as f:
        suffix = f" {d['right_n']}" if d["kind"] == "matching" else ""
        f.write(f"p {d['kind']} {d['n']}{suffix}\noffset {d['offset']}\n")
        for i, w in enumerate(d["unary"]):
            if w:
                f.write(f"u {i} {w}\n")
        for u, v, c in d["arcs"]:
            f.write(f"a {u} {v} {c}\n")
        for w, items in d["features"]:
            f.write("f " + str(w) + " " + " ".join(map(str, items)) + "\n")
        if d["kind"] == "concave":
            f.write(f"q {d['scale']}\n")
        for left, right in d["matching"]:
            f.write(f"e {left} {right}\n")
        if d["known"] is not None:
            f.write(f"o {d['known']}\n")
