#!/usr/bin/env julia

using SubmodularMinimization
using LinearAlgebra
using JSON3
using Pkg
BLAS.set_num_threads(1)

struct FileFunction <: SubmodularMinimization.SubmodularFunction
    calls::Base.RefValue{Int}
    n::Int
    kind::Symbol
    offset::Float64
    unary::Vector{Float64}
    arcs::Vector{Tuple{Int,Int,Float64}}
    features::Vector{Tuple{Float64,Vector{Int}}}
    concavity_scale::Float64
    matching_right::Int
    matching_adjacency::Vector{Vector{Int}}
end

SubmodularMinimization.ground_set_size(f::FileFunction) = f.n

function SubmodularMinimization.evaluate(f::FileFunction, selected::AbstractVector{Bool})
    f.calls[] += 1
    value = 0.0 # normalize F(empty) to zero for all solver internals
    @inbounds for item in 1:f.n
        selected[item] && (value += f.unary[item])
    end
    if f.kind == :cut
        @inbounds for (tail, head, capacity) in f.arcs
            selected[tail] && !selected[head] && (value += capacity)
        end
    elseif f.kind == :coverage
        @inbounds for (weight, items) in f.features
            any(item -> selected[item], items) && (value += weight)
        end
    elseif f.kind == :concave
        cardinality = count(identity, selected)
        value -= f.concavity_scale * cardinality * (cardinality - 1) / 2
    elseif f.kind == :matching
        matched_left = zeros(Int, f.matching_right)
        function augment(left::Int, seen::AbstractVector{Bool})
            @inbounds for right in f.matching_adjacency[left]
                seen[right] && continue
                seen[right] = true
                if matched_left[right] == 0 || augment(matched_left[right], seen)
                    matched_left[right] = left
                    return true
                end
            end
            return false
        end
        @inbounds for left in 1:f.n
            selected[left] && augment(left, falses(f.matching_right)) && (value += 1)
        end
    else
        error("unsupported instance kind $(f.kind)")
    end
    return value
end

function load_sfm(path::String)
    n = 0
    kind = :invalid
    offset = 0.0
    unary = Float64[]
    arcs = Tuple{Int,Int,Float64}[]
    features = Tuple{Float64,Vector{Int}}[]
    concavity_scale = 0.0
    has_concavity_scale = false
    matching_right = 0
    matching_edges = Tuple{Int,Int}[]
    known = nothing
    for line in eachline(path)
        fields = split(strip(line))
        (isempty(fields) || fields[1] == "c" || startswith(fields[1], "#")) && continue
        if fields[1] == "p"
            length(fields) in (3, 4) || error("invalid .sfm header")
            kind = Symbol(fields[2])
            kind in (:cut, :coverage, :concave, :matching) || error("unsupported .sfm kind")
            n = parse(Int, fields[3])
            if kind == :matching
                length(fields) == 4 || error("matching header needs right-side size")
                matching_right = parse(Int, fields[4])
            else
                length(fields) == 3 || error("unexpected header field")
            end
            unary = zeros(n)
        elseif fields[1] == "offset"
            offset = parse(Float64, fields[2])
        elseif fields[1] == "u"
            unary[parse(Int, fields[2]) + 1] += parse(Float64, fields[3])
        elseif fields[1] == "a"
            push!(arcs, (parse(Int, fields[2]) + 1,
                         parse(Int, fields[3]) + 1,
                         parse(Float64, fields[4])))
        elseif fields[1] == "f"
            push!(features, (parse(Float64, fields[2]),
                             [parse(Int, item) + 1 for item in fields[3:end]]))
        elseif fields[1] == "q"
            kind == :concave || error("q row outside concave instance")
            has_concavity_scale && error("duplicate q row")
            concavity_scale = parse(Float64, fields[2])
            concavity_scale >= 0 || error("negative concavity scale")
            has_concavity_scale = true
        elseif fields[1] == "e"
            kind == :matching || error("e row outside matching instance")
            left = parse(Int, fields[2]) + 1
            right = parse(Int, fields[3]) + 1
            1 <= left <= n && 1 <= right <= matching_right || error("matching edge out of range")
            push!(matching_edges, (left, right))
        elseif fields[1] == "o"
            known = parse(Float64, fields[2])
        else
            error("unknown .sfm tag: $(fields[1])")
        end
    end
    n > 0 || error("instance has no valid p line")
    kind == :concave && !has_concavity_scale && error("concave instance has no q row")
    matching_adjacency = [Int[] for _ in 1:n]
    for (left, right) in matching_edges
        push!(matching_adjacency[left], right)
    end
    return FileFunction(Ref(0), n, kind, offset, unary, arcs, features,
                        concavity_scale, matching_right, matching_adjacency), known
end

length(ARGS) == 4 || error("usage: compare_julia.jl INSTANCE.sfm REPEATS WARMUPS EPSILON")
instance, known = load_sfm(ARGS[1])
repeats, warmups = parse.(Int, ARGS[2:3])
epsilon = parse(Float64, ARGS[4])
samples = []
for trial in (1-warmups):repeats
    GC.gc()
    instance.calls[] = 0
    start = time_ns()
    selected, value, x, iterations = fujishige_wolfe_submodular_minimization(
        instance; ε=epsilon, max_iterations=10000, cache=false, verbose=false)
    elapsed = (time_ns()-start)/1e9
    calls = instance.calls[]
    trial <= 0 && continue
    # Offset restoration and validation are outside the timed region.
    actual = evaluate(instance, selected) + instance.offset
    push!(samples, (; minimum=actual, selected=findall(selected).-1,
        elapsed_seconds=elapsed, oracle_calls=calls, iterations,
        returned_value=value+instance.offset,
        base_lower_bound=sum(min.(x,0))+instance.offset,
        iteration_limit_reached=iterations>=1000000))
end
info = Pkg.dependencies()[Base.PkgId(SubmodularMinimization).uuid]
println(JSON3.write((; implementation="SubmodularMinimization.jl", arithmetic="Float64",
    julia_version=string(VERSION), package_version=string(info.version),
    package_revision=info.git_revision, epsilon, samples)))
