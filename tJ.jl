module TJModels

using LinearAlgebra
using SparseArrays

export Lattice,
    TJ,
    tJ,
    periodic_site,
    neighbours_of_point,
    kneighbours,
    dist,
    hole_dist,
    fermion_sign_hop,
    cicjdagger,
    XiXj,
    ZiZj,
    ninj,
    nh,
    spin_corr,
    hole_corr,
    assemblecorr,
    reorder_basis!,
    reduce_basis!,
    translate_state,
    rotate_state,
    assemblematrix,
    Hamiltonian,
    assembleH,
    hamiltonian_by_degree,
    assemble_by_degree

const Point = NTuple{2,Float64}

_point(x::Real, y::Real)::Point = (Float64(x), Float64(y))
_add(a::Point, b::Point)::Point = (a[1] + b[1], a[2] + b[2])
_sub(a::Point, b::Point)::Point = (a[1] - b[1], a[2] - b[2])
_scale(c::Real, a::Point)::Point = (Float64(c) * a[1], Float64(c) * a[2])
_dot(a::Point, b::Point)::Float64 = a[1] * b[1] + a[2] * b[2]
_norm(a::Point)::Float64 = sqrt(_dot(a, a))
_matvec(A::AbstractMatrix{<:Real}, x::Point)::Point =
    (A[1, 1] * x[1] + A[1, 2] * x[2], A[2, 1] * x[1] + A[2, 2] * x[2])
_round_key(x::Point)::Tuple{Int,Int} = (round(Int, x[1]), round(Int, x[2]))
_round_point(x::Point; digits::Int=0)::Point =
    (round(x[1]; digits=digits), round(x[2]; digits=digits))
_distance(a::Point, b::Point)::Float64 = _norm(_sub(a, b))

struct Lattice
    ux::Point
    uy::Point
    a::Float64
    a1::Point
    a2::Point
    b1::Point
    b2::Point
    b3::Point
    A1::Point
    A2::Point
    theta::Float64
    A1_frac::Point
    A2_frac::Point
    M::Int
    m::Int
    n::Int
    m_::Int
    n_::Int
    F2R::Matrix{Float64}
    F2R_inv::Matrix{Float64}
    M_frac::Matrix{Float64}
    M_frac_inv::Matrix{Float64}
    pmin::Int
    pmax::Int
    qmin::Int
    qmax::Int
    Asites::Vector{Point}
    Bsites::Vector{Point}
    lattice_sites::Vector{Point}
    lattice_hash::Dict{Tuple{Int,Int},Int}
    lattice_hash_inv::Dict{Int,Tuple{Int,Int}}
end

function Lattice(m::Integer, n::Integer, m_::Integer, n_::Integer)
    a = 3.0
    ux = _point(a, 0)
    uy = _point(0, a)
    a1 = ux
    a2 = _scale(0.5, _add(ux, _scale(sqrt(3), uy)))
    b1 = _scale(1 / 3, _add(a1, a2))
    b2 = _scale(1 / 3, _sub(a1, _scale(2, a2)))
    b3 = _scale(1 / 3, _sub(a2, _scale(2, a1)))

    A1 = _add(_scale(m, a1), _scale(n, a2))
    A2 = _add(_scale(m_, a1), _scale(n_, a2))
    theta = acos(_dot(A1, A2) / (_norm(A1) * _norm(A2)))

    A1_frac = _add(_scale(m, ux), _scale(n, uy))
    A2_frac = _add(_scale(m_, ux), _scale(n_, uy))
    M = 2 * abs(m * n_ - m_ * n)

    F2R = [1.0 0.5; 0.0 sqrt(3) / 2]
    F2R_inv = inv(F2R)
    M_frac = a .* [m m_; n n_]
    M_frac_inv = inv(M_frac)

    pmin = round(Int, minimum((0.0, A1_frac[1], A2_frac[1], A1_frac[1] + A2_frac[1])))
    pmax = round(Int, maximum((0.0, A1_frac[1], A2_frac[1], A1_frac[1] + A2_frac[1])))
    qmin = round(Int, minimum((0.0, A1_frac[2], A2_frac[2], A1_frac[2] + A2_frac[2])))
    qmax = round(Int, maximum((0.0, A1_frac[2], A2_frac[2], A1_frac[2] + A2_frac[2])))

    function site_in_parallelogram(r::Point)
        lambd, mu = _matvec(M_frac_inv, r)
        lambd = round(lambd; digits=4)
        mu = round(mu; digits=4)
        return (0 <= lambd < 1 && 0 <= mu < 1) ? r : nothing
    end

    Asites = Point[]
    for p in pmin:pmax
        for q in qmin:qmax
            rA = _add(_scale(p, ux), _scale(q, uy))
            site = site_in_parallelogram(rA)
            isnothing(site) || push!(Asites, site)
        end
    end
    sort!(Asites; by = r -> (r[2], r[1]))

    lattice = Lattice(
        ux,
        uy,
        a,
        a1,
        a2,
        b1,
        b2,
        b3,
        A1,
        A2,
        theta,
        A1_frac,
        A2_frac,
        M,
        Int(m),
        Int(n),
        Int(m_),
        Int(n_),
        F2R,
        F2R_inv,
        M_frac,
        M_frac_inv,
        pmin,
        pmax,
        qmin,
        qmax,
        Point[],
        Point[],
        Point[],
        Dict{Tuple{Int,Int},Int}(),
        Dict{Int,Tuple{Int,Int}}(),
    )

    Bsites = [periodic_site(lattice, r[1] + 1, r[2] + 1) for r in Asites]
    lattice_sites = Point[]
    for k in eachindex(Asites)
        push!(lattice_sites, Asites[k])
        push!(lattice_sites, Bsites[k])
    end

    lattice_hash = Dict{Tuple{Int,Int},Int}()
    lattice_hash_inv = Dict{Int,Tuple{Int,Int}}()
    for (index, coords) in enumerate(lattice_sites)
        key = _round_key(coords)
        lattice_hash[key] = index
        lattice_hash_inv[index] = key
    end

    return Lattice(
        ux,
        uy,
        a,
        a1,
        a2,
        b1,
        b2,
        b3,
        A1,
        A2,
        theta,
        A1_frac,
        A2_frac,
        M,
        Int(m),
        Int(n),
        Int(m_),
        Int(n_),
        F2R,
        F2R_inv,
        M_frac,
        M_frac_inv,
        pmin,
        pmax,
        qmin,
        qmax,
        Asites,
        Bsites,
        lattice_sites,
        lattice_hash,
        lattice_hash_inv,
    )
end

function periodic_site(lattice::Lattice, p::Real, q::Real)::Point
    lambd, mu = _matvec(lattice.M_frac_inv, _point(p, q))
    lambd = round(lambd; digits=5)
    mu = round(mu; digits=5)
    lambd = mod(lambd + 1, 1)
    mu = mod(mu + 1, 1)

    p_new = round(lambd * lattice.A1_frac[1] + mu * lattice.A2_frac[1]; digits=0)
    q_new = round(lambd * lattice.A1_frac[2] + mu * lattice.A2_frac[2]; digits=0)
    return _point(p_new, q_new)
end

function neighbours_of_point(
    lattice::Lattice,
    point::Point,
    k::Integer;
    Xmin::Integer=-7,
    Xmax::Integer=7,
    Ymin::Integer=-7,
    Ymax::Integer=7,
)
    i = lattice.lattice_hash[_round_key(point)]
    pos = _matvec(lattice.F2R, point)
    distances = Float64[]
    sites = Point[]

    for x in Xmin:Xmax
        for y in Ymin:Ymax
            frac = _add(_scale(x, lattice.ux), _scale(y, lattice.uy))
            r = _matvec(lattice.F2R, frac)
            r_prime = _add(r, _matvec(lattice.F2R, _point(1, 1)))

            push!(distances, round(_distance(pos, r); digits=4))
            push!(distances, round(_distance(pos, r_prime); digits=4))
            push!(sites, r)
            push!(sites, r_prime)
        end
    end

    order = sortperm(distances)
    sorted_distances = distances[order][2:end]
    sorted_sites = [_round_point(_matvec(lattice.F2R_inv, sites[idx]); digits=0) for idx in order[2:end]]

    counts = Int[]
    last_distance = nothing
    current_count = 0
    for d in sorted_distances
        if last_distance === nothing || d == last_distance
            current_count += 1
        else
            push!(counts, current_count)
            current_count = 1
        end
        last_distance = d
    end
    current_count == 0 || push!(counts, current_count)

    if k < 1 || k > length(counts)
        throw(ArgumentError("Requested neighbour degree $k, but only $(length(counts)) shells were found."))
    end

    neighbour_indices = cumsum(counts)
    first_index = k == 1 ? 1 : neighbour_indices[k - 1] + 1
    last_index = neighbour_indices[k]

    kN = Tuple{Int,Int}[]
    for neighbour_coords in sorted_sites[first_index:last_index]
        j = lattice.lattice_hash[_round_key(periodic_site(lattice, neighbour_coords[1], neighbour_coords[2]))]
        push!(kN, (i, j))
    end
    return kN
end

function kneighbours(
    lattice::Lattice,
    k::Integer;
    Xmin::Integer=-7,
    Xmax::Integer=7,
    Ymin::Integer=-7,
    Ymax::Integer=7,
)
    neighbours = Tuple{Int,Int}[]
    for point in lattice.lattice_sites
        append!(
            neighbours,
            neighbours_of_point(lattice, point, k; Xmin=Xmin, Xmax=Xmax, Ymin=Ymin, Ymax=Ymax),
        )
    end
    return sort(unique(filter(pair -> pair[1] < pair[2], neighbours)))
end

dist(::Lattice, r::Point, r_prime::Point)::Float64 = _distance(r, r_prime)

function hole_dist(lattice::Lattice, state::AbstractVector{<:Integer})::Point
    holes = findall(==(0), state)
    length(holes) == 2 || throw(ArgumentError("hole_dist expects a state with exactly two holes."))

    i, j = holes
    ri_frac = _point(lattice.lattice_hash_inv[i]...)
    rj_frac = _point(lattice.lattice_hash_inv[j]...)
    ri = _scale(1 / lattice.a, _matvec(lattice.F2R, ri_frac))
    rj = _scale(1 / lattice.a, _matvec(lattice.F2R, rj_frac))

    Rs = _scale.(
        1 / lattice.a,
        [
            _point(0, 0),
            lattice.A1,
            _scale(-1, lattice.A1),
            lattice.A2,
            _scale(-1, lattice.A1),
            _add(lattice.A1, lattice.A2),
            _scale(-1, _add(lattice.A1, lattice.A2)),
        ],
    )
    rj_trans = [_add(rj, R) for R in Rs]
    distances = [round(_distance(ri, rj_); digits=4) for rj_ in rj_trans]
    midpoint = _scale(0.5, _add(rj_trans[argmin(distances)], ri))
    return midpoint
end

mutable struct TJ
    lattice::Lattice
    nup::Int
    ndown::Int
    h::Int
    lattice_type::String
    PBC::Bool
    Sz::Float64
    N::Int
    phase::Int
    dim::Int
    basis::Vector{Vector{Int8}}
    hashmap::Dict{Tuple{Vararg{Int8}},Int}
    reduced_basis::Vector{Vector{Int8}}
    indices::Vector{Vector{Int}}
    translate_flip_basis::Vector{Vector{Int8}}
    translate_flip_indices::Vector{Vector{Int}}
    subdim::Int
    reduced_map::Dict{Int,Int}
    S::SparseMatrixCSC{Float64,Int}
end

const LATTICE_FIELDS = fieldnames(Lattice)

function Base.getproperty(model::TJ, sym::Symbol)
    if sym in LATTICE_FIELDS
        return getproperty(getfield(model, :lattice), sym)
    end
    return getfield(model, sym)
end

function Base.propertynames(model::TJ; private::Bool=false)
    return private ? (fieldnames(TJ)..., LATTICE_FIELDS...) : (fieldnames(TJ)..., LATTICE_FIELDS...)
end

function generate_basis(M::Integer, nup::Integer, ndown::Integer)
    holes = M - nup - ndown
    holes >= 0 || throw(ArgumentError("nup + ndown cannot exceed the number of sites."))

    counts = Dict(Int8(-1) => Int(ndown), Int8(0) => Int(holes), Int8(1) => Int(nup))
    values = (Int8(-1), Int8(0), Int8(1))
    state = Vector{Int8}(undef, M)
    basis = Vector{Vector{Int8}}()

    function fill_state!(pos::Int)
        if pos > M
            push!(basis, copy(state))
            return
        end

        for value in values
            if counts[value] > 0
                counts[value] -= 1
                state[pos] = value
                fill_state!(pos + 1)
                counts[value] += 1
            end
        end
    end

    fill_state!(1)
    return basis
end

state_key(state::AbstractVector{<:Integer}) = Tuple(Int8.(state))

function hash_table(basis::AbstractVector{<:AbstractVector{<:Integer}})
    hashmap = Dict{Tuple{Vararg{Int8}},Int}()
    for (index, fock_state) in enumerate(basis)
        hashmap[state_key(fock_state)] = index
    end
    return hashmap
end

function TJ(nup::Integer, ndown::Integer, m::Integer, n::Integer, m_::Integer, n_::Integer)
    lattice = Lattice(m, n, m_, n_)
    nup = Int(nup)
    ndown = Int(ndown)
    h = lattice.M - nup - ndown
    h >= 0 || throw(ArgumentError("nup + ndown cannot exceed lattice.M."))

    phase = iseven(nup) ? 1 : -1
    dim = binomial(lattice.M, nup) * binomial(lattice.M - nup, ndown)
    basis = generate_basis(lattice.M, nup, ndown)
    hashmap = hash_table(basis)

    return TJ(
        lattice,
        nup,
        ndown,
        h,
        "honeycomb",
        true,
        -0.5 * ndown + 0.5 * nup,
        nup + ndown,
        phase,
        dim,
        basis,
        hashmap,
        Vector{Vector{Int8}}(),
        Vector{Vector{Int}}(),
        Vector{Vector{Int8}}(),
        Vector{Vector{Int}}(),
        0,
        Dict{Int,Int}(),
        spzeros(Float64, 0, dim),
    )
end

const tJ = TJ

periodic_site(model::TJ, p::Real, q::Real)::Point = periodic_site(model.lattice, p, q)
neighbours_of_point(model::TJ, point::Point, k::Integer; kwargs...) =
    neighbours_of_point(model.lattice, point, k; kwargs...)
kneighbours(model::TJ, k::Integer; kwargs...) = kneighbours(model.lattice, k; kwargs...)

function fermion_sign_hop(::TJ, i::Integer, j::Integer, state::AbstractVector{<:Integer})::Int
    i == j && return 1
    a, b = i < j ? (i, j) : (j, i)
    occupied_between = 0
    for site in (a + 1):(b - 1)
        state[site] == 0 || (occupied_between += 1)
    end
    return isodd(occupied_between) ? -1 : 1
end

function cicjdagger(model::TJ, sigma::Integer, i::Integer, j::Integer, state::AbstractVector{<:Integer})
    if state[i] == sigma && state[j] == 0
        sign = fermion_sign_hop(model, i, j, state)
        newstate = Int8.(copy(state))
        newstate[i] = 0
        newstate[j] = Int8(sigma)
        return newstate, sign
    end
    return nothing, 0
end

function XiXj(::TJ, i::Integer, j::Integer, state::AbstractVector{<:Integer})
    if state[i] * state[j] == -1
        newstate = Int8.(copy(state))
        newstate[i] *= -1
        newstate[j] *= -1
        return newstate
    end
    return nothing
end

ZiZj(::TJ, i::Integer, j::Integer, state::AbstractVector{<:Integer})::Float64 =
    0.25 * state[i] * state[j]

ninj(::TJ, i::Integer, j::Integer, state::AbstractVector{<:Integer})::Int =
    abs(state[i] * state[j])

nh(::TJ, i::Integer, state::AbstractVector{<:Integer})::Int = 1 - abs(state[i])

function spin_corr_values(model::TJ, basis::AbstractVector{<:AbstractVector{<:Integer}})
    NNs = kneighbours(model, 1)
    expected = 3 * model.M ÷ 2
    if length(NNs) != expected
        throw(ArgumentError("Number of neighbors $(length(NNs)) is not equal to 3 * M / 2 = $expected."))
    end

    Css = zeros(Float64, length(basis))
    for (k, state) in enumerate(basis)
        for (i, j) in NNs
            Css[k] += ZiZj(model, i, j, state)
        end
    end
    return Css, NNs
end

function spin_corr(model::TJ, basis::AbstractVector{<:AbstractVector{<:Integer}}=model.basis)
    Css, NNs = spin_corr_values(model, basis)
    return spdiagm(0 => Css ./ length(NNs))
end

function hole_corr(model::TJ, basis::AbstractVector{<:AbstractVector{<:Integer}}, n::Integer)
    neighbours = kneighbours(model, n)
    Chh = zeros(Float64, length(basis))

    for (k, state) in enumerate(basis)
        for (i, j) in neighbours
            Chh[k] += nh(model, i, state) * nh(model, j, state)
        end
    end

    return spdiagm(0 => Chh ./ length(neighbours))
end

hole_corr(model::TJ, n::Integer) = hole_corr(model, model.basis, n)

assemblecorr(::TJ, diag) = nothing

function reorder_basis!(model::TJ)
    Css, _ = spin_corr_values(model, model.basis)
    idx = sortperm(Css)
    basis_sorted = [copy(model.basis[i]) for i in idx]
    hashmap = hash_table(basis_sorted)

    if model.Sz == 0
        used = falses(length(basis_sorted))
        paired = Vector{Vector{Int8}}()

        for (k, state) in enumerate(basis_sorted)
            used[k] && continue
            spin_flip_key = state_key(-state)
            ksf = get(hashmap, spin_flip_key, 0)
            ksf == 0 && throw(ArgumentError("Missing spin-flip partner in Sz=0 basis."))

            used[k] = true
            used[ksf] = true
            push!(paired, copy(state))
            push!(paired, copy(basis_sorted[ksf]))
        end

        model.basis = paired
    else
        model.basis = basis_sorted
    end

    model.hashmap = hash_table(model.basis)
    return model
end

translate_state(::TJ, state::AbstractVector{<:Integer}, n::Integer) = Int8.(circshift(state, 2 * n))
rotate_state(::TJ, state::AbstractVector{<:Integer}) = Int8.(-reverse(state))

function reduce_basis!(model::TJ, basis::AbstractVector{<:AbstractVector{<:Integer}}=model.basis; offset::Integer=0)
    reduced_basis = Vector{Vector{Int8}}()
    indices = Vector{Vector{Int}}()
    translate_flip_basis = Vector{Vector{Int8}}()
    translate_flip_indices = Vector{Vector{Int}}()
    checked = falses(model.dim)
    S_val = Float64[]

    for (index, state) in enumerate(model.basis)
        checked[index] && continue

        idx_temp = [index]
        S_val_temp = [1.0]
        checked[index] = true
        translates_to_flip = false

        for N in 1:(model.M ÷ 2 - 1)
            trindex = get(model.hashmap, state_key(translate_state(model, state, N)), 0)
            if trindex != 0 && !checked[trindex]
                push!(idx_temp, trindex)
                push!(S_val_temp, 1.0)
                checked[trindex] = true
                trindex == index + 1 && (translates_to_flip = true)
            end
        end

        if !translates_to_flip
            for idx_value in copy(idx_temp)
                spin_flip_index = idx_value + (isodd(idx_value) ? 1 : -1)
                if 1 <= spin_flip_index <= model.dim
                    checked[spin_flip_index] = true
                    push!(S_val_temp, model.phase)
                    push!(idx_temp, spin_flip_index)
                end
            end
            push!(reduced_basis, copy(state))
            append!(S_val, S_val_temp ./ norm(S_val_temp))
            push!(indices, idx_temp)
        elseif model.phase == 1
            push!(reduced_basis, copy(state))
            append!(S_val, S_val_temp ./ norm(S_val_temp))
            push!(indices, idx_temp)
        else
            push!(translate_flip_basis, copy(state))
        end
    end

    S_row = Int[]
    S_col = Int[]
    reduced_map = Dict{Int,Int}()
    for (row, idx_values) in enumerate(indices)
        for col in idx_values
            push!(S_row, row)
            push!(S_col, col)
            reduced_map[col] = row
        end
    end

    model.reduced_basis = reduced_basis
    model.indices = indices
    model.translate_flip_basis = translate_flip_basis
    model.translate_flip_indices = translate_flip_indices
    model.subdim = length(indices)
    model.reduced_map = reduced_map
    model.S = sparse(S_row, S_col, S_val, model.subdim, model.dim)

    return model
end

function assemblematrix(::TJ, shape::Tuple{Integer,Integer}, rcv)
    rows, cols, vals = rcv
    return sparse(rows, cols, vals, Int(shape[1]), Int(shape[2]))
end

struct SparseTriplet
    row::Vector{Int}
    col::Vector{Int}
    val::Vector{Float64}
end

SparseTriplet() = SparseTriplet(Int[], Int[], Float64[])

function _push!(triplet::SparseTriplet, row::Integer, col::Integer, val::Real)
    push!(triplet.row, Int(row))
    push!(triplet.col, Int(col))
    push!(triplet.val, Float64(val))
    return triplet
end

_state_index(::TJ, ::Nothing)::Int = 0
_state_index(model::TJ, state::AbstractVector{<:Integer})::Int = get(model.hashmap, state_key(state), 0)

function _hamiltonian_degree!(
    t_terms::SparseTriplet,
    J_terms::SparseTriplet,
    model::TJ,
    basis::AbstractVector{<:AbstractVector{<:Integer}},
    degree::Integer,
    offset::Integer,
    verbose::Bool,
)
    neighbours = kneighbours(model, degree)

    for (local_col, state) in enumerate(basis)
        col = local_col + offset
        diag = 0.0

        for (i, j) in neighbours
            for sigma in (1, -1)
                newstate, sign = cicjdagger(model, sigma, i, j, state)
                row = _state_index(model, newstate)
                row == 0 || _push!(t_terms, row, col, sign)
            end

            newstate = XiXj(model, i, j, state)
            row = _state_index(model, newstate)
            row == 0 || _push!(J_terms, row, col, -0.5)

            diag += ZiZj(model, i, j, state) - 0.25 * ninj(model, i, j, state)
        end

        diag == 0 || _push!(J_terms, col, col, diag)

        if verbose && col % 5000 == 0
            println("Saved matrix columns $col/$(model.dim) of degree-$degree terms")
        end
    end

    return t_terms, J_terms
end

function hamiltonian_by_degree(
    model::TJ,
    basis::AbstractVector{<:AbstractVector{<:Integer}}=model.basis;
    offset::Integer=0,
    verbose::Bool=false,
    degrees=(1, 2),
)
    t_terms = Dict{Int,SparseTriplet}()
    J_terms = Dict{Int,SparseTriplet}()

    for degree in degrees
        degree_int = Int(degree)
        t_triplet = SparseTriplet()
        J_triplet = SparseTriplet()
        _hamiltonian_degree!(t_triplet, J_triplet, model, basis, degree_int, offset, verbose)
        t_terms[degree_int] = t_triplet
        J_terms[degree_int] = J_triplet
    end

    return (t=t_terms, J=J_terms)
end

function Hamiltonian(
    model::TJ,
    basis::AbstractVector{<:AbstractVector{<:Integer}}=model.basis;
    offset::Integer=0,
    verbose::Bool=false,
)
    terms = hamiltonian_by_degree(model, basis; offset=offset, verbose=verbose, degrees=(2, 1))
    Ht2 = terms.t[2]
    HJ2 = terms.J[2]
    Ht = terms.t[1]
    HJ = terms.J[1]

    return (
        Ht.row,
        Ht.col,
        Ht.val,
        Ht2.row,
        Ht2.col,
        Ht2.val,
        HJ.row,
        HJ.col,
        HJ.val,
        HJ2.row,
        HJ2.col,
        HJ2.val,
    )
end

function assembleH(model::TJ, Hparams)
    (
        Ht_row,
        Ht_col,
        Ht_val,
        Ht2_row,
        Ht2_col,
        Ht2_val,
        HJ_row,
        HJ_col,
        HJ_val,
        HJ2_row,
        HJ2_col,
        HJ2_val,
    ) = Hparams

    Ht = sparse(Ht_row, Ht_col, Ht_val, model.dim, model.dim)
    Ht2 = sparse(Ht2_row, Ht2_col, Ht2_val, model.dim, model.dim)
    HJ = sparse(HJ_row, HJ_col, HJ_val, model.dim, model.dim)
    HJ2 = sparse(HJ2_row, HJ2_col, HJ2_val, model.dim, model.dim)

    Ht = Ht + transpose(Ht)
    Ht2 = Ht2 + transpose(Ht2)

    return Ht, Ht2, HJ, HJ2
end

function assemble_by_degree(model::TJ, terms; symmetrize_hopping::Bool=true)
    t_matrices = Dict{Int,SparseMatrixCSC{Float64,Int}}()
    J_matrices = Dict{Int,SparseMatrixCSC{Float64,Int}}()

    for (degree, triplet) in terms.t
        matrix = sparse(triplet.row, triplet.col, triplet.val, model.dim, model.dim)
        t_matrices[degree] = symmetrize_hopping ? matrix + transpose(matrix) : matrix
    end

    for (degree, triplet) in terms.J
        J_matrices[degree] = sparse(triplet.row, triplet.col, triplet.val, model.dim, model.dim)
    end

    return (t=t_matrices, J=J_matrices)
end

end # module TJModels
