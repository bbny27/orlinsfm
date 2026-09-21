# Orlin Submodular Function Minimization (SFM)

A research implementation and benchmarking project for general submodular function minimization, centred on James B. Orlin's strongly polynomial combinatorial algorithm and cross-checked against independent Julia and MATLAB solvers.

The project contains:

- a C++ implementation of Orlin's SFM algorithm using exact rational arithmetic;
- a floating-point (`double`) version of the same implementation for performance experiments;
- benchmark algorithm: Julia's `SubmodularMinimization.jl`;
- benchmark algotithm: Francis Bach's MATLAB SFO toolbox;
- graph-cut scaling instances derived from the Tsukuba stereo/max-flow benchmark;
- a varied range of submodular objectives including cuts, concave-cardinality functions, weighted coverage, bipartite matching, facility location, and special/edge-case instances;
- scripts for cross-checking correctness, timing solvers, and exporting reproducible CSV results.

---

## 1. Project goals

The project has two main goals.

First, it provides a relatively direct implementation of the algorithm in:

> James B. Orlin, **"A faster strongly polynomial time algorithm for submodular function minimization"**, Mathematical Programming, 118:237-251, 2009.

Orlin's paper gives a strongly polynomial SFM algorithm with running time

$$\[
O(n^5 EO+n^6)
\]$$

where `EO` is the time required for one evaluation of the submodular function. Orlin also writes the bound as

$$\[
O(n^4 EG+n^6)
\]$$

where `EG` is the cost of constructing one greedy extreme base.

Second, the repository is an experimental framework for comparing this combinatorial algorithm with other generic SFM approaches on exactly the same objective functions.

The benchmark currently compares four solver configurations:

| Solver label | Arithmetic | Main method |
|---|---|---|
| `orlin` | Boost `cpp_rational` | Orlin's combinatorial SFM algorithm |
| `orlin_double` | IEEE-754 `double` | Floating-point adaptation of the same Orlin implementation |
| `julia` | `Float64` | Fujishige-Wolfe / minimum-norm-point method from `SubmodularMinimization.jl` |
| `matlab` | MATLAB `double` | Minimum-norm-point based SFM routines from Bach's SFO toolbox |

The exact rational solver is primarily the **reference implementation for correctness**. The floating-point version is included because exact rational arithmetic becomes extremely expensive as the problem size grows.

---

## 2. Mathematical background

For a finite ground set

$$\[
V=\{1,\dots,n\}
\]$$

a set function

$$\[
F:2^V\rightarrow \mathbb{R}
\]$$

is **submodular** if

$$\[
F(A)+F(B)\geq F(A\cup B)+F(A\cap B)
\]$$

for all $\(A,B\subseteq V\)$.

Equivalently, submodularity can be interpreted as **diminishing marginal returns**:

$$\[
F(A\cup\{v\})-F(A)
\geq
F(B\cup\{v\})-F(B)
\]$$

whenever $\(A\subseteq B\)$ and $\(v\notin B\)$.

The SFM problem is

$$\[
\min_{S\subseteq V}F(S).
\]$$

The implementation normalizes the oracle when necessary so that internally

$$\[
F(\varnothing)=0.
\]$$

Adding or subtracting the constant \(F(\varnothing)\) does not change the minimizer.

A central object is the **base polyhedron**

$$\[
F(\varnothing)=0.
\]$$

$$
\[
B(F) =
\left\{
x\in\mathbb{R}^{n}:
x(V)=F(V),\;
x(S)\leq F(S)\quad \forall S\subseteq V
\right\}.
\]
$$

For an ordering $\(L=(v_1,\dots,v_n)\)$, a greedy extreme base is defined by

$$\[
y_L(v_j)
=
F(\{v_1,\dots,v_j\})
-
F(\{v_1,\dots,v_{j-1}\}).
\]$$

Orlin's algorithm maintains a point in the base polyhedron as a convex combination of such greedy extreme bases,

$$\[
x=\sum_{d\in D}\lambda_d y_d,
\qquad
\lambda_d>0,
\qquad
\sum_{d\in D}\lambda_d=1,
\]$$

while manipulating the associated **distance functions**, primary/secondary extreme bases, the zero set \(V^0\), and distance gaps until the optimality conditions are met.


## 3.1 Core C++ files

### `include/orlin/orlin_sfm.hpp`

The main Orlin implementation.

Its responsibilities include:

- representing active elements \(V^{in}\);
- storing distance functions;
- maintaining convex-combination coefficients \(\lambda_d\);
- generating greedy extreme bases;
- maintaining the current base vector \(x\);
- computing primary distance functions \(p(v)\);
- constructing secondary functions \(s(v)=INC(p(v),v)\);
- building columns $y_{s(v)}-y_{p(v)}$;
- solving for the non-negative direction vector \(\gamma\);
- taking the maximum feasible step \(\alpha\);
- updating \(\lambda\) and \(x\);
- removing zero-weight bases;
- performing `Reduce`;
- detecting distance gaps;
- checking algorithmic invariants.


### `include/orlin/types.hpp`

Provides the exact-number layer used by the reference solver.

The current exact implementation uses Boost.Multiprecision rational arithmetic, in particular `boost::multiprecision::cpp_rational`.

This is useful because several parts of Orlin's algorithm depend on exact distinctions such as:

- whether a quantity is exactly zero;
- whether a linear system is singular;
- whether a coefficient is exactly non-negative;
- whether a step length reaches a constraint exactly;
- whether an affine-dependence relation is exact.

Using rational arithmetic avoids having those decisions depend on an arbitrary floating-point tolerance.

### `include/orlin/examples.hpp`

Contains small reproducible example submodular oracles used for testing and demonstrations.

Examples used during development include:

- modular functions;
- concave-cardinality functions;
- directed cut functions;
- coverage functions;
- uniform matroid rank functions.

These are intentionally separated from the solver: the algorithm only needs a value oracle.

### `include/orlin/verification.hpp`

Contains validation helpers for small instances.

Typical uses include:

- exhaustive enumeration of all \(2^n\) subsets for small `n`;
- checking the returned minimum against brute force;
- checking the submodular inequality;
- checking internal invariants independently of the main solver.

Exhaustive verification is exponential and is therefore used only for small test cases.

### `src/main.cpp`

Command-line/demo driver for the built-in examples.

It is useful for quick correctness checks but is not the main cross-solver benchmark executable.

### `src/benchmark_instances.cpp`

Benchmark-oriented executable.

This is the executable normally used by `benchmarks/run_crosscheck.py`.

It reads the benchmark instance, selects the requested arithmetic mode/solver path, runs the implementation, and emits a machine-readable result used by the Python harness.

---

## 3.2 Benchmark files

### `benchmarks/run_crosscheck.py`

Main experiment orchestrator.

It:

1. discovers the requested `.max` or `.sfm` instances;
2. computes/checks the file hash;
3. reads the known optimum when available;
4. runs selected solvers;
5. applies warm-up runs;
6. performs repeated timed runs;
7. enforces a per-process timeout;
8. records status and objective value;
9. writes a common CSV.

Typical CSV columns are:

```text
instance
sha256
n
solver
arithmetic
known
status
minimum
median_seconds
detail
```

Common statuses include:

- `PASS_KNOWN`
- `FAILED`
- `TIMEOUT`
- `SKIPPED`

Using a single harness is important because all solvers are then evaluated against the same instance file and the same known optimum.

### `benchmarks/compare_julia.jl`

Adapter between the project `.sfm` representation and `SubmodularMinimization.jl`.

It defines a `FileFunction <: SubmodularFunction`, implements `ground_set_size`, implements the objective evaluator for the supported instance families, calls the Fujishige-Wolfe solver, restores the constant offset where required, and prints the objective in a format understood by the Python harness.

The adapter currently supports the same benchmark-function families used by the C++ benchmark.

### `benchmarks/compare_matlab.m`

MATLAB adapter for Francis Bach's SFO toolbox.

It constructs the corresponding MATLAB oracle and calls the SFO minimization routines, with the result passed back to the common Python harness.

### `benchmarks/Project.toml` and `benchmarks/Manifest.toml`

Julia environment files.

These are important for reproducibility: they record the Julia project dependencies independently of the user's global Julia installation.

### `benchmarks/vendor/SubmodularMinimization.jl/`

A local development copy of the Julia package.

The repository uses a vendored/developed copy rather than silently modifying the globally installed package, because a correctness patch is required for the benchmark.

### `benchmarks/patches/fix_singleton_minor_cycle.patch`

Patch applied to the vendored Julia solver.

See **Section 8** for the exact reason for this patch.

---

## 3.3 Dataset files

### `datasets/scaling/`

Contains Tsukuba-derived graph-cut instances used for scaling experiments.

Small induced square crops were generated for:

\[
1\times1,\;
2\times2,\;
\dots,\;
12\times12,
\]

giving problem sizes up to \(n=144\).

Larger scaling instances include:

- `16x16` (\(n=256\));
- `32x32` (\(n=1024\));
- `64x64` (\(n=4096\)).

The small square sequence is especially useful because exact Orlin becomes expensive rapidly, so a dense series of small sizes gives much more information than testing only very large vision instances.

### `datasets/varied/`

Contains non-Tsukuba instances intended to test whether the implementation works across different kinds of submodular structure rather than only graph cuts.

The current suite includes families such as:

- sparse cut functions;
- dense cut functions;
- concave-cardinality functions;
- weighted coverage;
- bipartite matching rank functions;
- facility-location functions;
- special/edge cases.

These files use the project's `.sfm` format.

---

# 4. Requirements

The project was developed primarily on **Windows 11**, using PowerShell and Visual Studio 2022.

Recommended versions:

| Dependency | Recommended setup |
|---|---|
| C++ compiler | Visual Studio 2022 / MSVC 19.x |
| CMake | 3.20 or newer |
| C++ standard | C++17 or newer |
| Boost | 1.75+ |
| Eigen | Eigen3 package discoverable by CMake |
| Python | 3.11 or 3.12 |
| Julia | 1.10+ |
| MATLAB | recent MATLAB release |
| MATLAB toolbox | SFO 1.2.0.0 or compatible |
| Git | recent version |

The core C++ implementation does **not** require MATLAB or Julia. Those are only required for the cross-solver comparison.

---

# 5. Detailed installation

## 5.1 Clone or copy the repository

Open PowerShell:

```powershell
cd "C:\Users\<USERNAME>\Downloads\FYP_Implementation"
git clone <YOUR_REPOSITORY_URL> Orlin_SFM
cd Orlin_SFM
```

If the project is already present locally:

```powershell
cd "<Path to File>"
```

---

## 5.2 Install Visual Studio C++ tools

Install **Visual Studio 2022** or **Build Tools for Visual Studio 2022**.

In the Visual Studio Installer, enable:

```text
Desktop development with C++
```

Make sure the installation includes:

- MSVC compiler;
- Windows SDK;
- CMake tools for Windows.

Check:

```powershell
cmake --version
```

For a normal PowerShell terminal, `cl.exe` may not be on `PATH`. CMake can still locate Visual Studio by generator name.

---

## 5.3 Install Boost

The exact solver uses Boost.Multiprecision.

A typical local installation may look like:

```text
C:\local\boost_1_92_0
```

The code uses header-based Boost components, so no Boost binary library is normally required for `cpp_rational`.

You can point CMake to Boost using:

```powershell
$env:BOOST_ROOT="C:\local\boost_1_92_0"
```

or supply the root directly during CMake configuration.

If CMake warns that `BOOST_ROOT` is ignored because of policy `CMP0144`, this is a CMake package-discovery warning rather than an Orlin-algorithm error. The project CMake configuration can explicitly handle the relevant Boost policy.

---

## 5.4 Install Eigen

Install Eigen3 somewhere on the machine, for example:

```text
C:\local\eigen
```

The important point is that CMake must be able to find `Eigen3Config.cmake`.

A typical configuration is:

```powershell
cmake -S . -B build `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DEigen3_DIR="C:/local/eigen/share/eigen3/cmake"
```

Adjust the path to the location containing `Eigen3Config.cmake`.

To locate it:

```powershell
Get-ChildItem C:\ -Filter Eigen3Config.cmake -Recurse -ErrorAction SilentlyContinue
```

If Eigen is found successfully, CMake should configure the `Eigen3::Eigen` target.

---

## 5.5 Configure the C++ project

From the repository root:

```powershell
cmake -S . -B build-varied `
  -G "Visual Studio 17 2022" `
  -A x64
```

If Boost or Eigen are not found automatically:

```powershell
cmake -S . -B build-varied `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DBOOST_ROOT="C:/local/boost_1_92_0" `
  -DEigen3_DIR="C:/local/eigen/share/eigen3/cmake"
```

A clean reconfiguration is often preferable after dependency changes:

```powershell
Remove-Item -Recurse -Force build-varied
cmake -S . -B build-varied -G "Visual Studio 17 2022" -A x64
```

---

## 5.6 Build

Build the release configuration:

```powershell
cmake --build build-varied --config Release
```

The benchmark executable should then appear at approximately:

```text
build-varied\Release\orlin_benchmark.exe
```

Check:

```powershell
Test-Path .\build-varied\Release\orlin_benchmark.exe
```

A result of `True` confirms the executable exists.

---

## 5.7 Install Python benchmark dependencies

Create a virtual environment:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the packages required by the benchmark scripts:

```powershell
pip install numpy
```

If additional scripts in the repository use pandas or plotting:

```powershell
pip install pandas matplotlib
```

Check:

```powershell
python --version
```

The cross-check harness itself uses Python mainly for subprocess management, CSV handling, hashing, validation, and experiment orchestration.

---

# 6. Julia baseline installation

## 6.1 Install Julia

Install Julia 1.10 or later.

Check:

```powershell
julia --version
```

---

## 6.2 Instantiate the benchmark environment

From the repository root:

```powershell
julia --project=benchmarks -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

On PowerShell, quoting can be awkward. The following form is also valid:

```powershell
julia --project=benchmarks -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
```

---

## 6.3 Use the vendored patched package

The benchmark should use the repository copy of `SubmodularMinimization.jl`, not an unrelated globally installed copy.

From PowerShell:

```powershell
julia --project=benchmarks -e 'using Pkg; Pkg.develop(path=raw"benchmarks/vendor/SubmodularMinimization.jl"); Pkg.instantiate(); Pkg.precompile()'
```

If PowerShell quoting causes problems, enter Julia interactively:

```powershell
julia --project=benchmarks
```

then run:

```julia
using Pkg
Pkg.develop(path=raw"benchmarks/vendor/SubmodularMinimization.jl")
Pkg.instantiate()
Pkg.precompile()
```

Verify which package Julia is actually loading:

```powershell
julia --project=benchmarks -e 'using SubmodularMinimization; println(pathof(SubmodularMinimization))'
```

The printed path should point into:

```text
benchmarks/vendor/SubmodularMinimization.jl/
```

and not a separate package copy under the global Julia depot.

---

# 7. MATLAB SFO baseline installation

The MATLAB comparison uses Francis Bach's **Submodular Function Optimization (SFO)** toolbox.

Suppose it is installed at:

```text
C:\Users\Downloads\SFO\sfo
```

Test MATLAB from PowerShell:

```powershell
matlab -batch "addpath(genpath('C:/Users/Downloads/SFO/sfo')); disp(which('sfo_min_norm_point'))"
```

The output should be a real `.m` file path.

For a stronger check:

```powershell
matlab -batch "addpath(genpath('C:/Users/Brian Bong Neng Ye/Downloads/SFO/sfo')); assert(~isempty(which('sfo_min_norm_point'))); disp('MATLAB SFO ready')"
```

If an assertion fails, the most common problem is simply that the `--sfo-dir` path points to the wrong folder level.

When running the benchmark, pass the actual toolbox directory:

```text
--sfo-dir "<path to SFO>"
```

---

# 8. Julia patches and adapter fixes

This project discovered a correctness issue in the Julia comparison package on benchmark instances that trigger a singleton collapse in a Wolfe minor cycle. This section still requires more review to determine if it is a genuine error.

## 8.1 Singleton minor-cycle bug

In the original Julia code in `src/algorithms.jl`, the minor-cycle logic contained the equivalent of:

```julia
if m <= 1
    return (false, m)
end
```

The project patch changes this to:

```julia
if m <= 1
    continue
end
```

### Why the original behaviour is wrong

In Wolfe's algorithm, the active/corral set can legitimately shrink to a single surviving vertex during a minor cycle.

A singleton active set is not, by itself, a failure condition.

When the minor cycle reaches one vertex, the algorithm should continue with that vertex as the current affine minimizer and then return to the next major-cycle optimality test / linear-oracle step.

The original `return (false, m)` exits the Wolfe routine immediately and reports non-convergence.

That premature exit is particularly dangerous because the higher-level SFM wrapper may still construct and return a set from the current non-optimal point. In other words, the call can produce a normal-looking set even though the Wolfe gap is still positive.

In a small regression example used during debugging, the old code stopped after the corral collapsed to the single point

```text
(5, -3, 0)
```

while a strictly positive Wolfe gap remained. The high-level result corresponded to the wrong set and wrong objective.

After the patch, the algorithm proceeds to the correct minimum-norm point

```text
(5, -3/2, -3/2)
```

and the Wolfe gap reaches zero.

### Why upstream tests could still pass

The defect is path-dependent.

Many ordinary test functions:

- converge without a singleton minor-cycle collapse;
- have very simple endpoint minimizers;
- use symmetric/non-negative examples that do not exercise the same geometry;
- check only the returned set/value and not the internal `converged` flag or Wolfe gap.

Therefore thousands of tests can pass while a specific geometric branch remains untested.

The Tsukuba-derived asymmetric graph-cut objectives were useful precisely because they exercised a different part of the solver.

### Applying the patch

If the patch file is tracked in the repository:

```powershell
cd benchmarks\vendor\SubmodularMinimization.jl
git apply ..\..\patches\fix_singleton_minor_cycle.patch
cd ..\..\..
```

If `git apply` says the patch is already applied, inspect the source before applying it again.

---

## 8.2 Julia workspace allocation fix in the benchmark adapter

A separate issue was caused by constructing the Wolfe workspace using the iteration limit as a workspace-size argument.

That can allocate an enormous dense affine workspace.

For example, a value such as:

```julia
MAX_ITERATIONS = 1_000_000
```

is a reasonable *iteration cap* but completely unreasonable as the number of active vertices to pre-allocate in a dense workspace.

The benchmark adapter therefore uses:

```julia
workspace = WolfeWorkspace(instance.n)
```

and passes the iteration cap only to the solver:

```julia
fujishige_wolfe_submodular_minimization!(
    workspace,
    instance;
    ε=epsilon,
    max_iterations=MAX_ITERATIONS,
    verbose=false,
)
```

This separates:

- problem dimension / workspace size; and
- maximum permitted iterations.

This change fixes the earlier `OutOfMemoryError()` behaviour in `AffineWorkspace`.

---

## 8.3 Matching adapter: `BitVector` compatibility

Julia's

```julia
falses(n)
```

returns a `BitVector`, not a `Vector{Bool}`.

The matching evaluator originally used a method signature such as:

```julia
augment(left::Int, selected::Vector{Bool})
```

and therefore failed when the actual argument was a `BitVector`.

The adapter was changed to:

```julia
augment(left::Int, selected::AbstractVector{Bool})
```

so that both ordinary boolean vectors and packed `BitVector`s are accepted.

This was an **adapter type bug**, not a flaw in the underlying SFM algorithm.

---

# 9. Dataset sources

## 9.1 Tsukuba graph-cut data

The scaling experiments use graph-cut/max-flow instances derived from the well-known **Tsukuba stereo** benchmark.

The public max-flow benchmark page at the University of Waterloo / Computer Vision at Western distributes stereo max-flow sequences such as:

```text
BVZ-tsukuba
KZ2-tsukuba
```

The benchmark page explains that these sequences correspond to subproblems produced by the first iteration of alpha-expansion, and that the Tsukuba photo set was provided by Tsukuba University.

Source:

```text
https://vision.cs.uwaterloo.ca/data/maxflow
```

The full benchmark graph is much larger than what is practical for a generic exact SFM implementation.

Therefore this project uses **induced square crops** of the graph.

The crop files are derived data; they are not claimed to be the original raw Tsukuba image dataset.

The crop generation preserves the induced subproblem over the selected vertices and writes a new instance with its own known optimum.

`tools/create_crop.py` is responsible for generating these small instances reproducibly.

---

## 9.2 Testing of SFM of varied data

The varied suite is inspired by the families exercised by:

```text
https://github.com/edwinlock/SubmodularMinimization.jl
```

That package includes test/example families such as:

- sparse and dense cut functions;
- concave-cardinality functions;
- bipartite matching;
- facility location;
- weighted coverage;
- special cases.

The `.sfm` files in this repository are project benchmark instances generated for the common C++ / Julia / MATLAB interface.

They should therefore be described as **generated benchmark instances inspired by / matching these function families**, rather than copied raw datasets, unless a specific file was directly taken from an external repository.

---

# 10. Algorithm sources

## 10.1 Orlin implementation

Primary source:

James B. Orlin, 2009.

```text
A faster strongly polynomial time algorithm
for submodular function minimization
Mathematical Programming 118:237-251
```

The code follows the paper's main objects and terminology closely:

- base polyhedron;
- greedy extreme bases;
- distance functions;
- valid triples \((D,\lambda,x)\);
- \(D_{min}(v)\);
- primary distance function \(p(v)\);
- secondary distance function \(s(v)=INC(p(v),v)\);
- zero set \(V^0\);
- positive set \(V^+\);
- auxiliary matrix;
- vector \(\gamma\);
- maximum step \(\alpha\);
- `Reduce`;
- distance gaps;
- elimination of vertices.

The paper states the final strongly polynomial complexity as

\[
O(n^5EO+n^6)
\]

or

\[
O(n^4EG+n^6).
\]

---

## 10.2 General submodular theory

A second important reference is:

Francis Bach, 2013.

```text
Learning with Submodular Functions:
A Convex Optimization Perspective
Foundations and Trends in Machine Learning
arXiv:1111.6453
```

This monograph is useful for:

- definitions of submodularity;
- Lovasz extensions;
- submodular and base polyhedra;
- greedy extreme bases;
- minimum-norm-point methods;
- examples such as cuts, coverage, matroids, and other submodular functions;
- relationships between SFM and convex optimization.

---

## 10.3 Julia baseline algorithm

The Julia package:

```text
https://github.com/edwinlock/SubmodularMinimization.jl
```

implements SFM using the Fujishige-Wolfe / minimum-norm-point approach.

The package itself cites the Wolfe-based SFM literature and exposes:

```julia
fujishige_wolfe_submodular_minimization
fujishige_wolfe_submodular_minimization!
```

The Julia solver is therefore an algorithmically independent baseline rather than another implementation of Orlin.

---

## 10.4 MATLAB baseline

The MATLAB baseline uses Francis Bach's SFO toolbox and its minimum-norm-point SFM routines.

This is another independent implementation and is especially useful because it provides a mature MATLAB baseline with a very different code path from the C++ Orlin implementation.

---

# 11. File formats

## 11.1 `.max`

`.max` files are graph/max-flow style instances.

They are used mainly for Tsukuba scaling.

For a graph cut with source/sink unary terms and pairwise non-negative capacities, the induced set function is submodular. This makes min-cut instances a structured special case of SFM.

The project converts the graph representation into the common value-oracle form required by the generic solvers.

## 11.2 `.sfm`

`.sfm` is the project's common benchmark format for general submodular functions.

The format stores enough information to reconstruct the objective consistently in:

- C++;
- Julia;
- MATLAB.

Depending on the `kind`, the file may contain fields such as:

- `n`;
- constant offset;
- unary/modular terms;
- graph arcs;
- coverage features;
- concavity parameters;
- matching adjacency;
- facility-location weights.

A major reason for using a shared file representation is to avoid accidentally benchmarking different mathematical functions in the different languages.

---

# 12. Running the benchmarks

## 12.1 Small Tsukuba sequence

PowerShell:

```powershell
$crops = 1..12 | ForEach-Object {
    "datasets/scaling/tsukuba_crop_${_}x${_}.max"
}

python benchmarks/run_crosscheck.py $crops `
  --cpp build-varied/Release/orlin_benchmark.exe `
  --sfo-dir "<PATH TO SFO>" `
  --solvers orlin orlin_double julia matlab `
  --max-n 150 `
  --repeats 3 `
  --warmups 1 `
  --timeout 600 `
  --output results/tsukuba_1_to_12.csv
```

To exclude the exact rational Orlin implementation:

```powershell
python benchmarks/run_crosscheck.py $crops `
  --cpp build-varied/Release/orlin_benchmark.exe `
  --sfo-dir "<PATH TO SFO>" ` `
  --solvers orlin_double julia matlab `
  --max-n 150 `
  --repeats 3 `
  --warmups 1 `
  --timeout 600 `
  --output results/tsukuba_1_to_12_no_exact.csv
```

---

## 12.2 Varied data

```powershell
python benchmarks/run_crosscheck.py "datasets/varied/*.sfm" `
  --cpp build-varied/Release/orlin_benchmark.exe `
  --solvers orlin orlin_double julia matlab `
  --sfo-dir "<PATH TO SFO>" ``
  --max-n 150 `
  --repeats 3 `
  --warmups 1 `
  --timeout 600 `
  --output results/varied_four_way.csv
```

For performance-oriented comparison without exact rational arithmetic:

```powershell
python benchmarks/run_crosscheck.py "datasets/varied/*.sfm" `
  --cpp build-varied/Release/orlin_benchmark.exe `
  --solvers orlin_double julia matlab `
  --sfo-dir "<PATH TO SFO>" ``
  --max-n 150 `
  --repeats 3 `
  --warmups 1 `
  --timeout 600 `
  --output results/varied_three_way.csv
```

---

# 13. Representative results

## 13.1 Tsukuba scaling

For the `16x16` Tsukuba crop:

```text
n = 256
known optimum = 50
```

Representative medians:

| Solver | Result | Median |
|---|---:|---:|
| `orlin_double` | 50 | ~107.18 s |
| `julia` | 50 | ~7.29 s |
| `matlab` | 50 | ~0.027 s |

All three completed solvers agreed with the known optimum.

At `32x32`:

```text
n = 1024
known optimum = 186
timeout = 600 s
```

in the latest scaling experiments, MATLAB completed rapidly while the generic Orlin/Julia runs reached the process-time budget.

This is a useful reminder that **strongly polynomial worst-case complexity does not imply practical competitiveness on structured graph-cut problems**. A max-flow/min-cut algorithm exploits graph structure that a generic SFM solver deliberately ignores.

---

## 13.2 Varied examples

Example current benchmark records include:

### Facility location

```text
facility_8x12.sfm
known optimum = -43
```

Representative medians:

| Solver | Median |
|---|---:|
| exact Orlin | 0.000245 s |
| Orlin double | 0.000162 s |
| Julia | 0.000132 s |
| MATLAB | 0.00721 s |

All four agreed on the minimum.

### Concave-cardinality

For `concave_n25.sfm`:

```text
known optimum = -15
```

representative medians were approximately:

| Solver | Median |
|---|---:|
| exact Orlin | 5.10 s |
| Orlin double | 0.0759 s |
| Julia | 0.000746 s |
| MATLAB | 0.00801 s |

This illustrates the very large cost of exact rational arithmetic even at modest problem sizes.

### Dense cut

For `cut_dense_n120.sfm`:

```text
known optimum = -6048
```

representative medians were approximately:

| Solver | Median |
|---|---:|
| exact Orlin | 0.128 s |
| Orlin double | 0.123 s |
| Julia | 4.16 s |
| MATLAB | 0.0157 s |

The relative ordering therefore depends strongly on the function family. There is no single solver that should be expected to dominate every oracle.


# 14. Verification strategy

Correctness is checked in several layers.

## 14.1 Unit tests

Run:

```powershell
ctest --test-dir build-varied -C Release --output-on-failure
```

or invoke individual test executables.

## 14.2 Exhaustive small-instance checking

For small `n`, enumerate all subsets and compute:

\[
\min_{S\subseteq V}F(S).
\]

Compare the solver result with this exact value.

This is the strongest practical correctness test for small instances.

## 14.3 Cross-solver agreement

For larger instances where exhaustive enumeration is impossible, compare:

- exact Orlin where feasible;
- floating Orlin;
- Julia;
- MATLAB;
- known graph-cut optimum where available.

Independent agreement does not constitute a formal proof, but it is strong implementation evidence.

## 14.4 File hashes

The benchmark records SHA-256 hashes so that two result rows can be traced to exactly the same input file.

# 15. References

## Core algorithm

James B. Orlin.  
**A faster strongly polynomial time algorithm for submodular function minimization.**  
*Mathematical Programming*, 118:237-251, 2009.  
DOI: `10.1007/s10107-007-0189-2`.

## General submodular optimization reference

Francis Bach.  
**Learning with Submodular Functions: A Convex Optimization Perspective.**  
2013.  
arXiv: `1111.6453`.

## Julia baseline

Edwin Lock et al.  
**SubmodularMinimization.jl**  
`https://github.com/edwinlock/SubmodularMinimization.jl`

The package implements a Fujishige-Wolfe / minimum-norm-point approach.

## MATLAB baseline

Francis Bach.  
**SFO - Submodular Function Optimization toolbox.**

The toolbox accompanies Bach's work on submodular optimization and is used here as an independent MATLAB minimum-norm-point baseline.

## Tsukuba / max-flow benchmark data

Computer Vision at Western / University of Waterloo.  
**Max-flow problem instances in vision.**  
`https://vision.cs.uwaterloo.ca/data/maxflow`

The benchmark page includes the `BVZ-tsukuba` and `KZ2-tsukuba` stereo/max-flow sequences and notes that the Tsukuba photo set was provided by Tsukuba University.

