# Dataset sources, derivations, and certificates

## Tsukuba maximum-flow crops

The original graph is `datasets/vision/BVZ-tsukuba0.max`, extracted from the
University of Waterloo Vision and Image Processing Lab's
[BVZ-tsukuba archive](https://vision.cs.uwaterloo.ca/files/BVZ-tsukuba.tbz2).

It also states that the supplied instances use DIMACS `.max` files and
corresponding `.sol` maximum-flow values. The archive contains 16 related
stereo graph-cut instances. The crop series here uses instance 0 only.

Every `tsukuba_crop_KxK.max`, for `K=1,...,12`, is an induced subgraph made by
`tools/create_crop.py` with:

- original grid: 384 by 288, row-major;
- original source 1, sink 2, and first pixel ID 3;
- top-left crop coordinate `(x,y)=(144,112)`;
- retained pixel ID `3 + (112+r)*384 + 144+c`, for `0<=r,c<K`;
- source and sink retained; an arc retained exactly when both endpoints are;
- retained pixels renumbered row-major from 3; boundary-crossing arcs omitted.

Each crop is a new problem, not a slice of the original solution. Its `.sol`
was computed independently by SciPy's integer `maximum_flow(..., method="dinic")`.
The crops have `n=K^2` free SFM elements; the DIMACS node count is `n+2`.
`tests/test_varied_datasets.py` re-solves all twelve references when SciPy is
installed. Some very small crops genuinely have optimum zero.

## Varied synthetic SFM instances

The function-family selection follows the examples and performance cases in
[SubmodularMinimization.jl](https://github.com/edwinlock/SubmodularMinimization.jl)
and its
[performance analysis](https://github.com/edwinlock/SubmodularMinimization.jl/blob/main/performance_analysis.jl).
The repository describes its examples as:

> “a comprehensive collection of submodular functions covering major application areas”

The numeric instances in `datasets/varied` are newly generated, deterministic
integer data; they are not copied benchmark outputs from that repository.
`catalog.csv` records dimensions, parameters, seed, optimum, selected-set size,
and certificate method for every file.

All objectives include optional modular terms `sum(i in S) u_i` so that the
known minimizer is usually neither empty nor the full ground set:

- **Sparse/dense directed cut:** `F(S)=sum u_i + sum c_ij [i in S,j not in S]`.
  Nonnegative integer capacities; exact optimum certified by integer min-cut.
- **Concave cardinality:**
  `F(S)=sum u_i - q*|S|*(|S|-1)/2`. For each cardinality, the best set contains
  the smallest unary weights; checking all cardinalities is therefore exact.
- **Bipartite matching rank:** `F(S)=rank(S)+sum u_i`, where `rank(S)` is the
  maximum matching size from selected left vertices to the right side. All
  subsets are enumerated, using an exact augmenting-path matching algorithm.
- **Weighted coverage:**
  `F(S)=sum u_i + sum_j w_j [S intersects feature_j]`. The exact certificate
  uses the standard auxiliary-node integer min-cut reduction.
- **Facility location:**
  `F(S)=sum u_i + sum_client max(i in S) similarity(client,i)` with zero for
  an empty maximum. Integer similarity levels are expanded exactly into
  weighted coverage threshold features, so no approximation is introduced;
  the resulting coverage graph is certified by integer min-cut.
- **Special graph cuts:** directed path, star, and complete-graph cases, also
  certified by integer min-cut.

The `.sfm` files contain the optimum in an `o VALUE` row. This is input
metadata, not the only check: the independent methods above are implemented in
`tests/test_varied_datasets.py`.

## License and citation note

The project license covers project code. Third-party datasets remain subject
to their source terms. Cite the Waterloo dataset page and the Tsukuba/Boykov–
Veksler–Zabih provenance when reporting results from the vision graphs. Cite
the Julia package only when discussing the comparator or the choice of
function families; the varied numeric instances themselves were created for
this project.
