module TJNetRC

# Fourth pass: Allow batching in MC algorithm but no speedup over plain CPU
# Fixed bug when loading CUDA for the first time during first training

# Julia port of the tJ neural-network training helpers from tJ/RC/tJNetRC.py.
# Required external packages: Flux, Optim, Zygote, and LineSearches.
# Optional GPU support: CUDA.jl. Set params["device"] to "auto", "gpu", or "cpu".

using LinearAlgebra
using Libdl
using Random
using Serialization
using SparseArrays
using Statistics
using Flux
using LineSearches
using Optim
using Zygote

export seed_julia!,
    cuda_device_count,
    gpu_available,
    processH,
    variational,
    NN_inputs,
    NeuralNetwork,
    random_tj_state,
    metropolis_samples,
    local_energy,
    mc_energy,
    trainNN,
    trainNNMC,
    testNNMC,
    testNN

const PARAM_MISSING = "__missing__"

function _param(params, key::AbstractString, default=PARAM_MISSING)
    if haskey(params, key)
        return params[key]
    end

    symkey = Symbol(key)
    if haskey(params, symkey)
        return params[symkey]
    end

    default === PARAM_MISSING && throw(KeyError(key))
    return default
end

function _param_present(params, key::AbstractString)
    haskey(params, key) && return true
    return haskey(params, Symbol(key))
end

_isnothing_or_empty(x) = x === nothing || x == ""

const _CUDA_LOAD_ATTEMPTED = Ref(false)
const _CUDA_MODULE = Ref{Any}(nothing)
const _CUDA_LOAD_ERROR = Ref{Any}(nothing)
const _CUDA_DRIVER_CHECKED = Ref(false)
const _CUDA_DRIVER_DEVICES = Ref(-1)
const _CUDA_DRIVER_ERROR = Ref{Any}(nothing)
const _CUDA_PKGID = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")

function _libcuda_handle()
    for libname in ("libcuda.so.1", "libcuda.so")
        handle = Libdl.dlopen_e(libname)
        handle == C_NULL || return handle
    end
    return C_NULL
end

function cuda_device_count()
    """Return the number of CUDA devices reported by the NVIDIA driver without loading CUDA.jl."""
    if _CUDA_DRIVER_CHECKED[]
        return max(_CUDA_DRIVER_DEVICES[], 0)
    end

    _CUDA_DRIVER_CHECKED[] = true
    handle = _libcuda_handle()
    if handle == C_NULL
        _CUDA_DRIVER_DEVICES[] = 0
        return 0
    end

    try
        cuInit = Libdl.dlsym_e(handle, :cuInit)
        cuDeviceGetCount = Libdl.dlsym_e(handle, :cuDeviceGetCount)
        if cuInit == C_NULL || cuDeviceGetCount == C_NULL
            _CUDA_DRIVER_DEVICES[] = 0
            return 0
        end

        init_status = ccall(cuInit, Cint, (Cuint,), 0)
        if init_status != 0
            _CUDA_DRIVER_ERROR[] = init_status
            _CUDA_DRIVER_DEVICES[] = 0
            return 0
        end

        count = Ref{Cint}(0)
        count_status = ccall(cuDeviceGetCount, Cint, (Ref{Cint},), count)
        if count_status != 0
            _CUDA_DRIVER_ERROR[] = count_status
            _CUDA_DRIVER_DEVICES[] = 0
            return 0
        end

        _CUDA_DRIVER_DEVICES[] = Int(count[])
        return _CUDA_DRIVER_DEVICES[]
    catch err
        _CUDA_DRIVER_ERROR[] = err
        _CUDA_DRIVER_DEVICES[] = 0
        return 0
    finally
        Libdl.dlclose(handle)
    end
end

function _load_cuda_module()
    _CUDA_LOAD_ATTEMPTED[] && return _CUDA_MODULE[]
    _CUDA_LOAD_ATTEMPTED[] = true

    if Base.find_package("CUDA") === nothing
        return nothing
    end

    try
        _CUDA_MODULE[] = Base.require(_CUDA_PKGID)
    catch err
        _CUDA_LOAD_ERROR[] = err
    end
    return _CUDA_MODULE[]
end

function gpu_available()
    """Return true when CUDA.jl is installed and the NVIDIA driver reports at least one GPU."""
    return Base.find_package("CUDA") !== nothing && cuda_device_count() > 0
end

function _device_request(params)
    _param_present(params, "device") && return lowercase(string(_param(params, "device")))
    _param_present(params, "nn_device") && return lowercase(string(_param(params, "nn_device")))
    _param_present(params, "use_gpu") && return Bool(_param(params, "use_gpu")) ? "gpu" : "cpu"
    _param_present(params, "gpu") && return Bool(_param(params, "gpu")) ? "gpu" : "cpu"
    return "auto"
end

function _cpu_device(requested::AbstractString="cpu")
    return (name=:cpu, requested=Symbol(replace(requested, "-" => "_")))
end

function _gpu_device(requested::AbstractString="gpu")
    return (name=:gpu, requested=Symbol(replace(requested, "-" => "_")))
end

function _resolve_device(params; context::AbstractString="neural-network training")
    requested = _device_request(params)
    requested in ("auto", "cpu", "gpu", "cuda") ||
        throw(ArgumentError("device must be \"auto\", \"cpu\", \"gpu\", or \"cuda\"."))

    requested == "cpu" && return _cpu_device(requested)

    if requested == "auto"
        return gpu_available() ? _gpu_device(requested) : _cpu_device(requested)
    end

    gpu_available() && return _gpu_device(requested)

    fallback = Bool(_param(params, "gpu_fallback", true))
    fallback || throw(ErrorException("GPU was requested for $context, but CUDA.jl is not functional."))
    @warn "GPU was requested for $context, but CUDA.jl is not functional; falling back to CPU."
    return _cpu_device(requested)
end

function _coerce_device(device)
    if device isa NamedTuple && haskey(device, :name)
        return device
    end

    requested = lowercase(string(device))
    requested in ("auto", "cpu", "gpu", "cuda") ||
        throw(ArgumentError("device must be \"auto\", \"cpu\", \"gpu\", or \"cuda\"."))
    requested == "cpu" && return _cpu_device(requested)
    requested == "auto" && return gpu_available() ? _gpu_device(requested) : _cpu_device(requested)
    return gpu_available() ? _gpu_device(requested) : _cpu_device(requested)
end

_device_label(device) = string(_coerce_device(device).name)
function _cuda_module_or_error()
    cuda = _load_cuda_module()
    cuda !== nothing && return cuda

    msg = "CUDA.jl is installed or GPU hardware was detected, but CUDA.jl could not be loaded."
    if _CUDA_LOAD_ERROR[] !== nothing
        msg *= "\nLast CUDA load error: " * sprint(showerror, _CUDA_LOAD_ERROR[])
    end
    throw(ErrorException(msg))
end

function _cuda_to_device(x)
    cuda = _cuda_module_or_error()
    return Base.invokelatest(getproperty(cuda, :cu), x)
end

_to_device(x, device) = _coerce_device(device).name == :gpu ? _cuda_to_device(x) : x
_to_cpu(x, device) = _coerce_device(device).name == :gpu ? Flux.cpu(x) : x
_scalar_value(x) = x isa AbstractArray ? only(Array(x)) : x
_scalar_float64(x) = Float64(_scalar_value(x))

function _cuda_needed_for_request(params)
    requested = _device_request(params)
    requested == "cpu" && return false
    requested == "auto" && return gpu_available()
    return requested in ("gpu", "cuda") && gpu_available()
end

function _prepare_cuda_world(params)
    _cuda_needed_for_request(params) || return false
    already_loaded = _CUDA_MODULE[] !== nothing
    _cuda_module_or_error()
    return !already_loaded
end

function _run_after_cuda_load_if_needed(f, params, args...; kwargs...)
    cuda_needed = _cuda_needed_for_request(params)
    cuda_needed || return f(args...; kwargs...)

    _prepare_cuda_world(params)
    return Base.invokelatest(f, args...; kwargs...)
end

function _return_network(params, neuralnet, device)
    Bool(_param(params, "return_model_on_device", false)) && return neuralnet
    return _to_cpu(neuralnet, device)
end

function seed_julia!(seed::Integer=12345)
    """Seed Julia RNGs used by the tJ neural network."""
    Random.seed!(seed)
    return seed
end

function processH(H; T::Type{<:AbstractFloat}=Float32, device=:cpu)
    """Convert a tJ Hamiltonian to a numeric matrix type suitable for Flux/Optim training."""
    H_work = issparse(H) ? sparse(T.(H)) : Matrix{T}(H)
    return _to_device(H_work, device)
end

function variational(H, psi)
    """Compute <psi|H|psi>/<psi|psi> for a tJ variational wavefunction."""
    psi_vec = vec(psi)
    norm_sq = dot(psi_vec, psi_vec)
    return real(dot(psi_vec, H * psi_vec) / norm_sq)
end

function _basis(model, trunc=nothing)
    basis = model.basis
    dim = length(basis)

    if trunc === nothing || Int(trunc) <= 0
        return basis, dim
    end

    trunc_dim = min(Int(trunc), dim)
    return basis[1:trunc_dim], trunc_dim
end

function NN_inputs(t, J, model; trunc=nothing, T::Type{<:AbstractFloat}=Float32, device=:cpu)
    """Build neural-network inputs for the tJ basis and the requested (t, J) couplings.

    The returned matrix has shape (M + 2, dim), following Flux's convention that
    columns are samples. Each column contains one Fock state followed by t and J.
    """
    basis, dim = _basis(model, trunc)
    M = model.M

    input_data = Matrix{T}(undef, M + 2, dim)
    for (col, state) in enumerate(basis)
        input_data[1:M, col] .= T.(state)
        input_data[M + 1, col] = T(t)
        input_data[M + 2, col] = T(J)
    end

    return _to_device(input_data, device)
end

function _xavier_uniform_tanh(out::Integer, in::Integer)
    gain = 5f0 / 3f0
    bound = gain * sqrt(6f0 / Float32(in + out))
    return rand(Float32, out, in) .* (2f0 * bound) .- bound
end

function _dense_tanh(in::Integer, out::Integer)
    W = _xavier_uniform_tanh(out, in)
    b = fill(0.01f0, out)
    return Dense(W, b, tanh)
end

function _dense_linear(in::Integer, out::Integer)
    W = _xavier_uniform_tanh(out, in)
    b = fill(0.01f0, out)
    return Dense(W, b)
end

function NeuralNetwork(M::Integer, n_neurons::Integer; dropout::Real=0.0)
    """Fully connected neural network that predicts tJ basis coefficients."""
    return Chain(
        _dense_tanh(M + 2, n_neurons),
        _dense_tanh(n_neurons, n_neurons),
        _dense_tanh(n_neurons, n_neurons),
        _dense_tanh(n_neurons, n_neurons),
        _dense_tanh(n_neurons, n_neurons),
        _dense_linear(n_neurons, 1),
    )
end

function _load_or_initialize_network(params, M::Integer, n_neurons::Integer, dropout::Real, seed::Integer)
    seed_julia!(seed)
    load_from = _param(params, "load_from", nothing)

    if _isnothing_or_empty(load_from)
        return NeuralNetwork(M, n_neurons; dropout=dropout)
    end

    return deserialize(String(load_from))
end

function _save_network(params, neuralnet; device=:cpu)
    model_path = _param(params, "model_path", nothing)
    _isnothing_or_empty(model_path) && return nothing
    save_on_device = Bool(_param(params, "save_model_on_device", false))
    neuralnet_to_save = save_on_device ? neuralnet : _to_cpu(neuralnet, device)
    serialize(String(model_path), neuralnet_to_save)
    return model_path
end

function _slice_hamiltonian(H, dim::Integer)
    size(H, 1) == dim && size(H, 2) == dim && return H
    return H[1:dim, 1:dim]
end

function _symmetry_reduce(H, dim::Integer; phase::Real=1)
    iseven(dim) || throw(ArgumentError("exploit_symmetry requires an even tJ basis dimension."))

    idx = vcat(collect(1:2:dim), collect(2:2:dim))
    H_reordered = H[idx, idx]
    half_dim = div(dim, 2)
    block_left = H_reordered[1:half_dim, 1:half_dim]
    block_cross = H_reordered[1:half_dim, (half_dim + 1):dim]
    return (2 .* block_left .+ phase .* (block_cross .+ transpose(block_cross))) ./ 2
end

function _symmetry_inputs(input_data, dim::Integer)
    iseven(dim) || throw(ArgumentError("exploit_symmetry requires an even tJ basis dimension."))
    return input_data[:, 1:2:dim]
end

function _full_wavefunction(psi, dim::Integer; exploit_symmetry::Bool=false, phase::Real=1)
    psi_vec = vec(Array(psi))

    if exploit_symmetry
        full_psi = similar(psi_vec, dim)
        full_psi[1:2:dim] .= psi_vec
        full_psi[2:2:dim] .= phase .* psi_vec
        psi_vec = full_psi
    end

    psi_vec ./= norm(psi_vec)
    return psi_vec
end

function random_tj_state(model; rng=Random.default_rng())
    """Draw a random tJ Fock state with the same particle counts as `model`."""
    M = Int(model.M)
    nup = Int(model.nup)
    ndown = Int(model.ndown)
    holes = M - nup - ndown
    holes >= 0 || throw(ArgumentError("nup + ndown cannot exceed model.M."))

    state = Vector{Int8}(undef, M)
    state[1:nup] .= Int8(1)
    state[(nup + 1):(nup + ndown)] .= Int8(-1)
    state[(nup + ndown + 1):M] .= Int8(0)
    shuffle!(rng, state)
    return state
end

function _state_input(
    state::AbstractVector{<:Integer},
    t,
    J;
    T::Type{<:AbstractFloat}=Float32,
    device=:cpu,
)
    M = length(state)
    input = Matrix{T}(undef, M + 2, 1)
    input[1:M, 1] .= T.(state)
    input[M + 1, 1] = T(t)
    input[M + 2, 1] = T(J)
    return _to_device(input, device)
end

function _states_input(states::AbstractVector, t, J; T::Type{<:AbstractFloat}=Float32, device=:cpu)
    isempty(states) && throw(ArgumentError("At least one state is required."))
    M = length(first(states))
    input = Matrix{T}(undef, M + 2, length(states))
    for (col, state) in enumerate(states)
        length(state) == M || throw(ArgumentError("All sampled states must have the same length."))
        input[1:M, col] .= T.(state)
        input[M + 1, col] = T(t)
        input[M + 2, col] = T(J)
    end
    return _to_device(input, device)
end

function _amplitudes(neuralnet, states::AbstractVector, t, J; T::Type{<:AbstractFloat}=Float32, device=:cpu)
    return vec(Array(neuralnet(_states_input(states, t, J; T=T, device=device))))
end

function _amplitude(
    neuralnet,
    state::AbstractVector{<:Integer},
    t,
    J;
    T::Type{<:AbstractFloat}=Float32,
    device=:cpu,
)
    return only(vec(Array(neuralnet(_state_input(state, t, J; T=T, device=device)))))
end

function _logabs_amplitude(
    neuralnet,
    state::AbstractVector{<:Integer},
    t,
    J;
    T::Type{<:AbstractFloat}=Float32,
    device=:cpu,
    eps::Real=1.0e-12,
)
    amp = _amplitude(neuralnet, state, t, J; T=T, device=device)
    return log(abs(amp) + T(eps))
end

function _logabs_amplitudes(
    neuralnet,
    states::AbstractVector,
    t,
    J;
    T::Type{<:AbstractFloat}=Float32,
    device=:cpu,
    eps::Real=1.0e-12,
)
    input = Zygote.ignore_derivatives() do
        _states_input(states, t, J; T=T, device=device)
    end
    amps = vec(neuralnet(input))
    return log.(abs.(amps) .+ T(eps))
end

function _logabs_amplitudes_cpu(
    neuralnet,
    states::AbstractVector,
    t,
    J;
    T::Type{<:AbstractFloat}=Float32,
    device=:cpu,
    eps::Real=1.0e-12,
)
    amps = _amplitudes(neuralnet, states, t, J; T=T, device=device)
    return log.(abs.(amps) .+ T(eps))
end

function _propose_exchange(state::AbstractVector{<:Integer}, rng)
    M = length(state)
    proposal = Int8.(copy(state))
    M < 2 && return proposal

    for _ in 1:(4 * M)
        i = rand(rng, 1:M)
        j = rand(rng, 1:M)
        if i != j && state[i] != state[j]
            proposal[i], proposal[j] = proposal[j], proposal[i]
            return proposal
        end
    end

    for i in 1:(M - 1), j in (i + 1):M
        if state[i] != state[j]
            proposal[i], proposal[j] = proposal[j], proposal[i]
            return proposal
        end
    end

    return proposal
end

function _initial_chain_states(model, n_chains::Integer, rng, initial_states)
    if initial_states === nothing
        return [random_tj_state(model; rng=rng) for _ in 1:n_chains]
    end

    length(initial_states) >= n_chains ||
        throw(ArgumentError("initial_states must contain at least n_chains states."))
    return [Int8.(copy(initial_states[chain])) for chain in 1:n_chains]
end

function _accept_metropolis(logratio::Real, rng)::Bool
    isnan(logratio) && return false
    logratio >= 0 && return true
    return log(rand(rng)) < logratio
end

function _use_batched_mc(batched, device)
    batched === nothing && return _coerce_device(device).name == :gpu
    return Bool(batched)
end

function metropolis_samples(
    neuralnet,
    model,
    t,
    J;
    n_samples::Integer=1024,
    n_chains::Integer=16,
    n_discard::Integer=64,
    sweeps_per_sample::Integer=1,
    sweep_size=nothing,
    rng=Random.default_rng(),
    initial_states=nothing,
    T::Type{<:AbstractFloat}=Float32,
    device=:cpu,
    batched=nothing,
    eps::Real=1.0e-12,
)
    """Sample tJ Fock states from |psi(s)|^2 with local Metropolis exchanges.

    The proposal swaps two unlike site occupations, preserving nup, ndown, and
    the hole count. This is the same Markov-chain idea used by NetKet's
    Metropolis samplers: propose a symmetry-sector-preserving local move and
    accept it with probability min(1, |psi(new)/psi(old)|^2).
    """
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    n_chains > 0 || throw(ArgumentError("n_chains must be positive."))
    sweep = sweep_size === nothing ? Int(model.M) : Int(sweep_size)
    sweep > 0 || throw(ArgumentError("sweep_size must be positive."))

    chain_states = _initial_chain_states(model, Int(n_chains), rng, initial_states)
    use_batched = _use_batched_mc(batched, device)
    logamps =
        use_batched ?
        _logabs_amplitudes_cpu(neuralnet, chain_states, t, J; T=T, device=device, eps=eps) :
        [
            _logabs_amplitude(neuralnet, state, t, J; T=T, device=device, eps=eps)
            for state in chain_states
        ]
    accepted = 0
    proposed = 0

    function step!(chain::Int)
        proposal = _propose_exchange(chain_states[chain], rng)
        proposal == chain_states[chain] && return nothing

        proposed += 1
        proposal_logamp = _logabs_amplitude(neuralnet, proposal, t, J; T=T, device=device, eps=eps)
        logratio = 2 * (proposal_logamp - logamps[chain])
        if _accept_metropolis(logratio, rng)
            chain_states[chain] = proposal
            logamps[chain] = proposal_logamp
            accepted += 1
        end
        return nothing
    end

    function step_batch!()
        proposal_chains = Int[]
        proposals = Vector{Vector{Int8}}()
        sizehint!(proposal_chains, length(chain_states))
        sizehint!(proposals, length(chain_states))

        for chain in eachindex(chain_states)
            proposal = _propose_exchange(chain_states[chain], rng)
            proposal == chain_states[chain] && continue
            push!(proposal_chains, chain)
            push!(proposals, proposal)
        end

        isempty(proposals) && return nothing

        proposed += length(proposals)
        proposal_logamps =
            _logabs_amplitudes_cpu(neuralnet, proposals, t, J; T=T, device=device, eps=eps)

        for (idx, chain) in enumerate(proposal_chains)
            proposal_logamp = proposal_logamps[idx]
            logratio = 2 * (proposal_logamp - logamps[chain])
            if _accept_metropolis(logratio, rng)
                chain_states[chain] = proposals[idx]
                logamps[chain] = proposal_logamp
                accepted += 1
            end
        end

        return nothing
    end

    for _ in 1:Int(n_discard)
        if use_batched
            for _ in 1:sweep
                step_batch!()
            end
        else
            for chain in 1:Int(n_chains), _ in 1:sweep
                step!(chain)
            end
        end
    end

    samples = Vector{Vector{Int8}}()
    sizehint!(samples, Int(n_samples))
    if use_batched
        while length(samples) < n_samples
            for _ in 1:Int(sweeps_per_sample), _ in 1:sweep
                step_batch!()
            end
            for chain in 1:Int(n_chains)
                push!(samples, copy(chain_states[chain]))
                length(samples) >= n_samples && break
            end
        end
    else
        while length(samples) < n_samples
            for chain in 1:Int(n_chains)
                for _ in 1:Int(sweeps_per_sample), _ in 1:sweep
                    step!(chain)
                end
                push!(samples, copy(chain_states[chain]))
                length(samples) >= n_samples && break
            end
        end
    end

    acceptance = proposed == 0 ? 0.0 : accepted / proposed
    return (samples=samples, acceptance=acceptance, accepted=accepted, proposed=proposed)
end

_state_key(state::AbstractVector{<:Integer}) = Tuple(Int8.(state))

function _model_state_index(model, state::AbstractVector{<:Integer})
    if !hasproperty(model, :hashmap)
        throw(ArgumentError("Matrix-based local energy requires model.hashmap."))
    end
    idx = get(model.hashmap, _state_key(state), 0)
    idx == 0 && throw(ArgumentError("Sampled state was not found in model.hashmap."))
    return idx
end

function _basis_state(model, idx::Integer)
    if !hasproperty(model, :basis)
        throw(ArgumentError("Matrix-based local energy requires model.basis."))
    end
    return model.basis[idx]
end

function _matrix_connections(model, state::AbstractVector{<:Integer}, H)
    idx = _model_state_index(model, state)
    connected_states = Vector{Vector{Int8}}()
    coeffs = Float64[]

    if issparse(H)
        rows = rowvals(H)
        vals = nonzeros(H)
        for ptr in nzrange(H, idx)
            push!(connected_states, Int8.(copy(_basis_state(model, rows[ptr]))))
            push!(coeffs, Float64(vals[ptr]))
        end
    else
        for row in axes(H, 1)
            val = H[row, idx]
            val == 0 && continue
            push!(connected_states, Int8.(copy(_basis_state(model, row))))
            push!(coeffs, Float64(val))
        end
    end

    return connected_states, coeffs
end

function _fermion_sign_hop(i::Integer, j::Integer, state::AbstractVector{<:Integer})::Int
    i == j && return 1
    a, b = i < j ? (i, j) : (j, i)
    occupied_between = 0
    for site in (a + 1):(b - 1)
        state[site] == 0 || (occupied_between += 1)
    end
    return isodd(occupied_between) ? -1 : 1
end

function _hop_state(sigma::Integer, i::Integer, j::Integer, state::AbstractVector{<:Integer})
    if state[i] == sigma && state[j] == 0
        sign = _fermion_sign_hop(i, j, state)
        newstate = Int8.(copy(state))
        newstate[i] = 0
        newstate[j] = Int8(sigma)
        return newstate, sign
    end
    return nothing, 0
end

function _spin_flip_pair(i::Integer, j::Integer, state::AbstractVector{<:Integer})
    if state[i] * state[j] == -1
        newstate = Int8.(copy(state))
        newstate[i] *= -1
        newstate[j] *= -1
        return newstate
    end
    return nothing
end

_zizj(i::Integer, j::Integer, state::AbstractVector{<:Integer}) = 0.25 * state[i] * state[j]
_ninj(i::Integer, j::Integer, state::AbstractVector{<:Integer}) = abs(state[i] * state[j])

function _kneighbours_for_model(model, degree::Integer)
    model_module = parentmodule(typeof(model))
    if isdefined(model_module, :kneighbours)
        return getproperty(model_module, :kneighbours)(model, degree)
    end
    throw(ArgumentError("Matrix-free local energy requires a kneighbours(model, degree) method."))
end

function _degree_tuple(degrees)
    degrees isa Integer && return (Int(degrees),)
    return Tuple(Int.(collect(degrees)))
end

function _neighbour_cache(model, degrees)
    return Dict(degree => _kneighbours_for_model(model, degree) for degree in _degree_tuple(degrees))
end

function _matrix_free_connections(
    model,
    state::AbstractVector{<:Integer},
    t,
    J,
    neighbour_cache,
)
    connected_states = Vector{Vector{Int8}}()
    coeffs = Float64[]

    for (_, neighbours) in neighbour_cache
        diag = 0.0
        for (i, j) in neighbours
            for sigma in (1, -1)
                newstate, sign = _hop_state(sigma, i, j, state)
                if newstate !== nothing
                    push!(connected_states, newstate)
                    push!(coeffs, Float64(t) * sign)
                end

                newstate, sign = _hop_state(sigma, j, i, state)
                if newstate !== nothing
                    push!(connected_states, newstate)
                    push!(coeffs, Float64(t) * sign)
                end
            end

            flipped = _spin_flip_pair(i, j, state)
            if flipped !== nothing
                push!(connected_states, flipped)
                push!(coeffs, -0.5 * Float64(J))
            end

            diag += Float64(J) * (_zizj(i, j, state) - 0.25 * _ninj(i, j, state))
        end

        if diag != 0
            push!(connected_states, Int8.(copy(state)))
            push!(coeffs, diag)
        end
    end

    return connected_states, coeffs
end

function _local_connections(model, state::AbstractVector{<:Integer}, t, J; H=nothing, neighbour_cache=nothing, degrees=(1,))
    return H === nothing ?
           _matrix_free_connections(
               model,
               state,
               t,
               J,
               neighbour_cache === nothing ? _neighbour_cache(model, degrees) : neighbour_cache,
           ) :
           _matrix_connections(model, state, H)
end

function local_energy(
    neuralnet,
    model,
    state::AbstractVector{<:Integer},
    t,
    J;
    H=nothing,
    neighbour_cache=nothing,
    degrees=(1,),
    T::Type{<:AbstractFloat}=Float32,
    device=:cpu,
    eps::Real=1.0e-12,
)
    """Compute the variational local energy E_loc(s) = (H psi)(s) / psi(s)."""
    connected_states, coeffs =
        _local_connections(model, state, t, J; H=H, neighbour_cache=neighbour_cache, degrees=degrees)

    isempty(connected_states) && return 0.0

    amp0 = _amplitude(neuralnet, state, t, J; T=T, device=device)
    denom = abs(amp0) <= eps ? copysign(T(eps), amp0 == 0 ? one(amp0) : amp0) : amp0
    amps = _amplitudes(neuralnet, connected_states, t, J; T=T, device=device)
    return Float64(sum(coeffs .* (amps ./ denom)))
end

function _local_energies_batched(
    neuralnet,
    model,
    states::AbstractVector,
    t,
    J;
    H=nothing,
    neighbour_cache=nothing,
    degrees=(1,),
    T::Type{<:AbstractFloat}=Float32,
    device=:cpu,
    eps::Real=1.0e-12,
)
    isempty(states) && return Float64[]

    base_amps = _amplitudes(neuralnet, states, t, J; T=T, device=device)
    local_values = zeros(Float64, length(states))
    all_connected_states = Vector{Vector{Int8}}()
    all_coeffs = Float64[]
    starts = Vector{Int}(undef, length(states))
    lengths = Vector{Int}(undef, length(states))

    for (sample_idx, state) in enumerate(states)
        connected_states, coeffs =
            _local_connections(model, state, t, J; H=H, neighbour_cache=neighbour_cache, degrees=degrees)

        starts[sample_idx] = length(all_coeffs) + 1
        lengths[sample_idx] = length(coeffs)

        append!(all_connected_states, connected_states)
        append!(all_coeffs, coeffs)
    end

    isempty(all_connected_states) && return local_values

    connected_amps = _amplitudes(neuralnet, all_connected_states, t, J; T=T, device=device)

    for sample_idx in eachindex(states)
        len = lengths[sample_idx]
        len == 0 && continue

        amp0 = base_amps[sample_idx]
        denom = abs(amp0) <= eps ? copysign(T(eps), amp0 == 0 ? one(amp0) : amp0) : amp0
        first_idx = starts[sample_idx]
        last_idx = first_idx + len - 1
        coeff_view = @view all_coeffs[first_idx:last_idx]
        amp_view = @view connected_amps[first_idx:last_idx]
        local_values[sample_idx] = Float64(sum(coeff_view .* (amp_view ./ denom)))
    end

    return local_values
end

function mc_energy(
    neuralnet,
    model,
    t,
    J;
    H=nothing,
    n_samples::Integer=1024,
    n_chains::Integer=16,
    n_discard::Integer=64,
    sweeps_per_sample::Integer=1,
    sweep_size=nothing,
    degrees=(1,),
    rng=Random.default_rng(),
    initial_states=nothing,
    T::Type{<:AbstractFloat}=Float32,
    device=:cpu,
    batched=nothing,
)
    """Estimate <H> with Metropolis samples and matrix-free or sparse-matrix local energies."""
    use_batched = _use_batched_mc(batched, device)
    neighbour_cache = H === nothing ? _neighbour_cache(model, degrees) : nothing
    sampling = metropolis_samples(
        neuralnet,
        model,
        t,
        J;
        n_samples=n_samples,
        n_chains=n_chains,
        n_discard=n_discard,
        sweeps_per_sample=sweeps_per_sample,
        sweep_size=sweep_size,
        rng=rng,
        initial_states=initial_states,
        T=T,
        device=device,
        batched=use_batched,
    )

    local_values =
        use_batched ?
        _local_energies_batched(
            neuralnet,
            model,
            sampling.samples,
            t,
            J;
            H=H,
            neighbour_cache=neighbour_cache,
            degrees=degrees,
            T=T,
            device=device,
        ) :
        [
            local_energy(
                neuralnet,
                model,
                state,
                t,
                J;
                H=H,
                neighbour_cache=neighbour_cache,
                degrees=degrees,
                T=T,
                device=device,
            )
            for state in sampling.samples
        ]

    energy = mean(local_values)
    variance = length(local_values) > 1 ? var(local_values; corrected=true) : 0.0
    stderr = sqrt(variance / length(local_values))
    return (
        energy=energy,
        stderr=stderr,
        variance=variance,
        local_values=local_values,
        samples=sampling.samples,
        acceptance=sampling.acceptance,
        accepted=sampling.accepted,
        proposed=sampling.proposed,
    )
end

function _variational_loss(neuralnet, H, input_data; exploit_symmetry::Bool=false)
    scale = exploit_symmetry ? inv(sqrt(eltype(input_data)(2))) : one(eltype(input_data))
    psi = vec(neuralnet(input_data)) .* scale
    return variational(H, psi)
end

function _loss_function(rebuild, H, input_data; exploit_symmetry::Bool=false)
    return function(theta)
        return _variational_loss(rebuild(theta), H, input_data; exploit_symmetry=exploit_symmetry)
    end
end

function _vmc_surrogate_loss(
    neuralnet,
    model,
    t,
    J;
    H=nothing,
    n_samples::Integer=1024,
    n_chains::Integer=16,
    n_discard::Integer=64,
    sweeps_per_sample::Integer=1,
    sweep_size=nothing,
    degrees=(1,),
    rng=Random.default_rng(),
    T::Type{<:AbstractFloat}=Float32,
    device=:cpu,
    batched=nothing,
    eps::Real=1.0e-12,
)
    use_batched = _use_batched_mc(batched, device)
    sampling = Zygote.ignore_derivatives() do
        metropolis_samples(
            neuralnet,
            model,
            t,
            J;
            n_samples=n_samples,
            n_chains=n_chains,
            n_discard=n_discard,
            sweeps_per_sample=sweeps_per_sample,
            sweep_size=sweep_size,
            rng=rng,
            T=T,
            device=device,
            batched=use_batched,
            eps=eps,
        )
    end
    neighbour_cache = Zygote.ignore_derivatives() do
        H === nothing ? _neighbour_cache(model, degrees) : nothing
    end
    local_values = Zygote.ignore_derivatives() do
        use_batched ?
        _local_energies_batched(
            neuralnet,
            model,
            sampling.samples,
            t,
            J;
            H=H,
            neighbour_cache=neighbour_cache,
            degrees=degrees,
            T=T,
            device=device,
            eps=eps,
        ) :
        [
            local_energy(
                neuralnet,
                model,
                state,
                t,
                J;
                H=H,
                neighbour_cache=neighbour_cache,
                degrees=degrees,
                T=T,
                device=device,
                eps=eps,
            )
            for state in sampling.samples
        ]
    end

    energy = mean(local_values)
    variance = length(local_values) > 1 ? var(local_values; corrected=true) : 0.0
    stderr = sqrt(variance / length(local_values))
    centered = local_values .- energy
    logamps = _logabs_amplitudes(neuralnet, sampling.samples, t, J; T=T, device=device, eps=eps)
    centered_on_device = Zygote.ignore_derivatives() do
        _to_device(eltype(logamps).(centered), device)
    end
    surrogate = 2 * mean(centered_on_device .* logamps)

    stats = (
        energy=energy,
        stderr=stderr,
        variance=variance,
        acceptance=sampling.acceptance,
        accepted=sampling.accepted,
        proposed=sampling.proposed,
    )
    return surrogate, stats
end

function _maybe_mc_hamiltonian(params, t, J, Ht, HJ, dim)
    use_matrix = Bool(_param(params, "mc_use_matrix_hamiltonian", false))
    use_matrix || return nothing
    (Ht === nothing || HJ === nothing) &&
        throw(ArgumentError("mc_use_matrix_hamiltonian=true requires Ht and HJ."))
    return processH(t .* _slice_hamiltonian(Ht, dim) .+ J .* _slice_hamiltonian(HJ, dim))
end

function _maybe_return_wavefunction(params, neuralnet, model, t, J; trunc=nothing, device=:cpu)
    return_wavefunction = Bool(_param(params, "return_wavefunction", false))
    return_wavefunction || return nothing
    input_data = NN_inputs(t, J, model; trunc=trunc, device=device)
    dim = size(input_data, 2)
    psi_reduced = vec(neuralnet(input_data))
    return _full_wavefunction(psi_reduced, dim)
end

function _optimise_lbfgs(loss_from_theta, theta0, params)
    learning_rate = Float64(_param(params, "learning_rate", 1.0))
    epochs = Int(_param(params, "epochs", 300))
    history_size = Int(_param(params, "history_size", 200))
    loss_diff = Float64(_param(params, "loss_diff", 1.0e-4))
    stagnation = Int(_param(params, "stagnation", 20))
    verbose = Bool(_param(params, "verbose", false))
    print_every = Int(_param(params, "print_every", 5))

    loss_history = Float64[]
    prev_loss = Ref(Inf)
    stagnation_counter = Ref(0)
    start_time = time()

    f(theta) = loss_from_theta(theta)

    function g!(G, theta)
        loss, back = Zygote.pullback(loss_from_theta, theta)
        G .= only(back(one(loss)))
        return G
    end

    callback = state -> begin
        epoch = hasproperty(state, :iteration) ? getfield(state, :iteration) : getfield(state, :pseudo_iteration)
        current_loss = hasproperty(state, :value) ? Float64(getfield(state, :value)) : Float64(getfield(state, :f_x))
        push!(loss_history, current_loss)

        if verbose && epoch != 0 && epoch % print_every == 0
            println("epoch $epoch completed, maximum remaining epochs: $(epochs - epoch)\tCurrent loss: $current_loss")
        end

        if abs(current_loss - prev_loss[]) < loss_diff
            stagnation_counter[] += 1
        else
            stagnation_counter[] = 0
        end
        prev_loss[] = current_loss

        return stagnation_counter[] >= stagnation
    end

    options = Optim.Options(
        iterations=epochs,
        callback=callback,
        show_trace=false,
        allow_f_increases=true,
    )
    optimizer = Optim.LBFGS(;
        m=history_size,
        alphaguess=LineSearches.InitialStatic(alpha=learning_rate),
        linesearch=LineSearches.HagerZhang(),
    )
    result = Optim.optimize(f, g!, theta0, optimizer, options)

    if verbose
        epoch = Optim.iterations(result)
        run_time = time() - start_time
        println("Number of training epochs: $epoch, training time: $(round(run_time; digits=4))s")
    end

    if isempty(loss_history)
        push!(loss_history, Float64(loss_from_theta(Optim.minimizer(result))))
    end

    return result, loss_history
end

function _full_optimizer_name(params, device)
    if _param_present(params, "nn_optimizer")
        return lowercase(string(_param(params, "nn_optimizer")))
    end
    if _param_present(params, "optimizer")
        return lowercase(string(_param(params, "optimizer")))
    end
    return _coerce_device(device).name == :gpu ? "adam" : "lbfgs"
end

function _flux_optimizer(optimizer_name::AbstractString, learning_rate::Real)
    name = lowercase(string(optimizer_name))
    if name == "adam"
        return Flux.Adam(learning_rate)
    elseif name in ("descent", "sgd")
        return Flux.Descent(learning_rate)
    end
    throw(ArgumentError("Flux optimizer \"$optimizer_name\" is not supported; use \"adam\", \"descent\", or \"sgd\"."))
end

function _optimise_full_flux!(
    neuralnet,
    H,
    input_data,
    params;
    exploit_symmetry::Bool=false,
    optimizer_name::AbstractString="adam",
)
    learning_rate = Float64(_param(params, "nn_learning_rate", 1.0e-3))
    epochs = Int(_param(params, "epochs", 300))
    loss_diff = Float64(_param(params, "loss_diff", 1.0e-4))
    stagnation = Int(_param(params, "stagnation", 20))
    verbose = Bool(_param(params, "verbose", false))
    print_every = Int(_param(params, "print_every", 5))

    opt_state = Flux.setup(_flux_optimizer(optimizer_name, learning_rate), neuralnet)
    loss_history = Float64[]
    prev_loss = Ref(Inf)
    stagnation_counter = Ref(0)
    start_time = time()

    for epoch in 1:epochs
        loss_ref = Ref{Any}()
        grads = Zygote.gradient(neuralnet) do net
            loss = _variational_loss(net, H, input_data; exploit_symmetry=exploit_symmetry)
            Zygote.ignore_derivatives() do
                loss_ref[] = loss
            end
            return loss
        end

        grads[1] === nothing && throw(ErrorException("Full-basis gradient was nothing."))
        Flux.update!(opt_state, neuralnet, grads[1])

        current_loss = _scalar_float64(loss_ref[])
        push!(loss_history, current_loss)

        if verbose && epoch % print_every == 0
            println("epoch $epoch completed, maximum remaining epochs: $(epochs - epoch)\tCurrent loss: $current_loss")
        end

        if abs(current_loss - prev_loss[]) < loss_diff
            stagnation_counter[] += 1
        else
            stagnation_counter[] = 0
        end
        prev_loss[] = current_loss
        stagnation_counter[] >= stagnation && break
    end

    if verbose
        epoch = length(loss_history)
        run_time = time() - start_time
        println("Number of training epochs: $epoch, training time: $(round(run_time; digits=4))s")
    end

    return neuralnet, loss_history
end

function _trainNN_impl(params_dict, model, t, J, Ht, HJ; trunc=nothing, seed::Integer=12345)
    """Train a neural-network variational wavefunction for the tJ model.

    `Ht` is the hopping matrix and `HJ` is the nearest-neighbor exchange matrix.
    The Hamiltonian optimized here is `t * Ht + J * HJ`.
    """
    if Bool(_param(params_dict, "use_monte_carlo", false))
        return trainNNMC(params_dict, model, t, J, Ht, HJ; trunc=trunc, seed=seed)
    end

    device = _resolve_device(params_dict; context="trainNN")
    verbose = Bool(_param(params_dict, "verbose", false))
    verbose && println("Neural network training device: $(_device_label(device))")

    input_data = NN_inputs(t, J, model; trunc=trunc, device=device)
    M = model.M
    dim = size(input_data, 2)
    n_neurons = Int(_param(params_dict, "n_neurons"))
    dropout = Float64(_param(params_dict, "dropout", 0.0))
    exploit_symmetry = Bool(_param(params_dict, "exploit_symmetry", false))
    phase = 1

    Ht_work = _slice_hamiltonian(Ht, dim)
    HJ_work = _slice_hamiltonian(HJ, dim)
    H = t .* Ht_work .+ J .* HJ_work

    if exploit_symmetry
        H = _symmetry_reduce(H, dim; phase=phase)
        input_data = _symmetry_inputs(input_data, dim)
    end

    H = processH(H; device=device)
    neuralnet = _to_device(_load_or_initialize_network(params_dict, M, n_neurons, dropout, seed), device)
    optimizer_name = _full_optimizer_name(params_dict, device)

    if device.name == :gpu && optimizer_name == "lbfgs"
        @warn "Optim.LBFGS is CPU-oriented in this training path; using Adam for GPU full-basis training. Set device=\"cpu\" to keep LBFGS."
        optimizer_name = "adam"
    end

    if optimizer_name == "lbfgs"
        theta0, rebuild = Flux.destructure(neuralnet)
        loss_from_theta = _loss_function(rebuild, H, input_data; exploit_symmetry=exploit_symmetry)
        result, loss_history = _optimise_lbfgs(loss_from_theta, theta0, params_dict)
        theta_final = Optim.minimizer(result)
        trained_net = rebuild(theta_final)
        E0 = Float64(loss_from_theta(theta_final))
    else
        trained_net, loss_history = _optimise_full_flux!(
            neuralnet,
            H,
            input_data,
            params_dict;
            exploit_symmetry=exploit_symmetry,
            optimizer_name=optimizer_name,
        )
        E0 = _scalar_float64(_variational_loss(trained_net, H, input_data; exploit_symmetry=exploit_symmetry))
    end

    scale = exploit_symmetry ? inv(sqrt(eltype(input_data)(2))) : one(eltype(input_data))
    psi_reduced = vec(trained_net(input_data)) .* scale
    psi0 = _full_wavefunction(psi_reduced, dim; exploit_symmetry=exploit_symmetry, phase=phase)

    _save_network(params_dict, trained_net; device=device)
    return psi0, E0
end

function trainNN(params_dict, model, t, J, Ht, HJ; trunc=nothing, seed::Integer=12345)
    return _run_after_cuda_load_if_needed(
        _trainNN_impl,
        params_dict,
        params_dict,
        model,
        t,
        J,
        Ht,
        HJ;
        trunc=trunc,
        seed=seed,
    )
end

function _trainNNMC_impl(
    params_dict,
    model,
    t,
    J,
    Ht=nothing,
    HJ=nothing;
    trunc=nothing,
    seed::Integer=12345,
    return_details::Bool=false,
)
    """Train the tJ neural-network ansatz with Metropolis VMC energy estimates.

    The sampler draws Fock states from |psi(s)|^2 and optimizes the standard
    real-valued VMC score-function surrogate, using local energies instead of
    the full many-body wavefunction. By default local energies are computed
    matrix-free from the tJ hopping and exchange rules; set
    `"mc_use_matrix_hamiltonian" => true` to use supplied sparse `Ht` and `HJ`
    columns for small-system validation. Set `"mc_batched" => true` to batch
    proposal and local-energy amplitude evaluations; GPU runs enable this by default.
    """
    trunc === nothing ||
        throw(ArgumentError("trainNNMC does not support trunc; the sampler works in the full sector."))

    exploit_symmetry = Bool(_param(params_dict, "exploit_symmetry", false))
    exploit_symmetry &&
        throw(ArgumentError("trainNNMC currently samples the unreduced basis; set exploit_symmetry=false."))

    M = Int(model.M)
    dim = Int(model.dim)
    n_neurons = Int(_param(params_dict, "n_neurons"))
    dropout = Float64(_param(params_dict, "dropout", 0.0))

    epochs = Int(_param(params_dict, "epochs", 300))
    learning_rate = Float64(_param(params_dict, "mc_learning_rate", 1.0e-3))
    n_samples = Int(_param(params_dict, "mc_samples", 1024))
    n_chains = Int(_param(params_dict, "mc_chains", 16))
    n_discard = Int(_param(params_dict, "mc_burn_in", 64))
    sweeps_per_sample = Int(_param(params_dict, "mc_sweeps_per_sample", 1))
    sweep_size = _param(params_dict, "mc_sweep_size", M)
    degrees = _degree_tuple(_param(params_dict, "mc_degrees", (1,)))
    eval_samples = Int(_param(params_dict, "mc_eval_samples", n_samples))
    eval_burn_in = Int(_param(params_dict, "mc_eval_burn_in", n_discard))
    loss_diff = Float64(_param(params_dict, "loss_diff", 1.0e-4))
    stagnation = Int(_param(params_dict, "stagnation", 20))
    verbose = Bool(_param(params_dict, "verbose", false))
    print_every = Int(_param(params_dict, "print_every", 5))
    device = _resolve_device(params_dict; context="trainNNMC")
    batched_mc = Bool(_param(params_dict, "mc_batched", _coerce_device(device).name == :gpu))
    verbose && println("VMC neural network training device: $(_device_label(device))")
    verbose && println("VMC batched amplitude/local-energy path: $batched_mc")
    neuralnet = _to_device(_load_or_initialize_network(params_dict, M, n_neurons, dropout, seed), device)

    optimizer_name = lowercase(String(_param(params_dict, "mc_optimizer", "adam")))
    optimizer_name == "adam" ||
        throw(ArgumentError("trainNNMC currently supports mc_optimizer=\"adam\"."))

    H = _maybe_mc_hamiltonian(params_dict, t, J, Ht, HJ, dim)
    rng = MersenneTwister(seed)
    opt_state = Flux.setup(Flux.Adam(learning_rate), neuralnet)

    energy_history = Float64[]
    stderr_history = Float64[]
    variance_history = Float64[]
    acceptance_history = Float64[]
    prev_energy = Ref(Inf)
    stagnation_counter = Ref(0)
    start_time = time()

    for epoch in 1:epochs
        stats_ref = Ref{Any}()
        grads = Zygote.gradient(neuralnet) do net
            surrogate, stats = _vmc_surrogate_loss(
                net,
                model,
                t,
                J;
                H=H,
                n_samples=n_samples,
                n_chains=n_chains,
                n_discard=n_discard,
                sweeps_per_sample=sweeps_per_sample,
                sweep_size=sweep_size,
                degrees=degrees,
                rng=rng,
                device=device,
                batched=batched_mc,
            )
            Zygote.ignore_derivatives() do
                stats_ref[] = stats
            end
            return surrogate
        end

        grads[1] === nothing && throw(ErrorException("VMC gradient was nothing."))
        Flux.update!(opt_state, neuralnet, grads[1])

        stats = stats_ref[]
        push!(energy_history, Float64(stats.energy))
        push!(stderr_history, Float64(stats.stderr))
        push!(variance_history, Float64(stats.variance))
        push!(acceptance_history, Float64(stats.acceptance))

        if verbose && epoch % print_every == 0
            println(
                "epoch $epoch completed, maximum remaining epochs: $(epochs - epoch)",
                "\tMC energy: $(stats.energy) +/- $(stats.stderr)",
                "\tacceptance: $(round(stats.acceptance; digits=4))",
            )
        end

        if abs(stats.energy - prev_energy[]) < loss_diff
            stagnation_counter[] += 1
        else
            stagnation_counter[] = 0
        end
        prev_energy[] = stats.energy
        stagnation_counter[] >= stagnation && break
    end

    final_stats = mc_energy(
        neuralnet,
        model,
        t,
        J;
        H=H,
        n_samples=eval_samples,
        n_chains=n_chains,
        n_discard=eval_burn_in,
        sweeps_per_sample=sweeps_per_sample,
        sweep_size=sweep_size,
        degrees=degrees,
        rng=rng,
        device=device,
        batched=batched_mc,
    )
    E0 = Float64(final_stats.energy)
    psi0 = _maybe_return_wavefunction(params_dict, neuralnet, model, t, J; trunc=trunc, device=device)

    if verbose
        epoch = length(energy_history)
        run_time = time() - start_time
        println("Number of VMC training epochs: $epoch, training time: $(round(run_time; digits=4))s")
        println("Final MC energy: $E0 +/- $(final_stats.stderr)")
    end

    _save_network(params_dict, neuralnet; device=device)

    if return_details
        return (
            psi=psi0,
            energy=E0,
            stderr=final_stats.stderr,
            variance=final_stats.variance,
            neuralnet=_return_network(params_dict, neuralnet, device),
            energy_history=energy_history,
            stderr_history=stderr_history,
            variance_history=variance_history,
            acceptance_history=acceptance_history,
            final_stats=final_stats,
        )
    end

    return psi0, E0
end

function trainNNMC(
    params_dict,
    model,
    t,
    J,
    Ht=nothing,
    HJ=nothing;
    trunc=nothing,
    seed::Integer=12345,
    return_details::Bool=false,
)
    return _run_after_cuda_load_if_needed(
        _trainNNMC_impl,
        params_dict,
        params_dict,
        model,
        t,
        J,
        Ht,
        HJ;
        trunc=trunc,
        seed=seed,
        return_details=return_details,
    )
end

function _testNNMC_impl(
    params_dict,
    model,
    t,
    J,
    Ht=nothing,
    HJ=nothing;
    trunc=nothing,
    seed::Integer=12345,
    return_details::Bool=false,
)
    """Evaluate a saved or initialized neural-network wavefunction with MC local energies."""
    trunc === nothing ||
        throw(ArgumentError("testNNMC does not support trunc; the sampler works in the full sector."))

    exploit_symmetry = Bool(_param(params_dict, "exploit_symmetry", false))
    exploit_symmetry &&
        throw(ArgumentError("testNNMC currently samples the unreduced basis; set exploit_symmetry=false."))

    M = Int(model.M)
    dim = Int(model.dim)
    n_neurons = Int(_param(params_dict, "n_neurons"))
    dropout = Float64(_param(params_dict, "dropout", 0.0))
    device = _resolve_device(params_dict; context="testNNMC")
    batched_mc = Bool(_param(params_dict, "mc_batched", _coerce_device(device).name == :gpu))
    neuralnet = _to_device(_load_or_initialize_network(params_dict, M, n_neurons, dropout, seed), device)

    n_samples = Int(_param(params_dict, "mc_eval_samples", _param(params_dict, "mc_samples", 1024)))
    n_chains = Int(_param(params_dict, "mc_chains", 16))
    n_discard = Int(_param(params_dict, "mc_eval_burn_in", _param(params_dict, "mc_burn_in", 64)))
    sweeps_per_sample = Int(_param(params_dict, "mc_sweeps_per_sample", 1))
    sweep_size = _param(params_dict, "mc_sweep_size", M)
    degrees = _degree_tuple(_param(params_dict, "mc_degrees", (1,)))

    H = _maybe_mc_hamiltonian(params_dict, t, J, Ht, HJ, dim)
    rng = MersenneTwister(seed)
    stats = mc_energy(
        neuralnet,
        model,
        t,
        J;
        H=H,
        n_samples=n_samples,
        n_chains=n_chains,
        n_discard=n_discard,
        sweeps_per_sample=sweeps_per_sample,
        sweep_size=sweep_size,
        degrees=degrees,
        rng=rng,
        device=device,
        batched=batched_mc,
    )
    psi0 = _maybe_return_wavefunction(params_dict, neuralnet, model, t, J; trunc=trunc, device=device)

    if return_details
        return (
            psi=psi0,
            energy=Float64(stats.energy),
            stderr=stats.stderr,
            variance=stats.variance,
            neuralnet=_return_network(params_dict, neuralnet, device),
            final_stats=stats,
        )
    end

    return psi0, Float64(stats.energy)
end

function testNNMC(
    params_dict,
    model,
    t,
    J,
    Ht=nothing,
    HJ=nothing;
    trunc=nothing,
    seed::Integer=12345,
    return_details::Bool=false,
)
    return _run_after_cuda_load_if_needed(
        _testNNMC_impl,
        params_dict,
        params_dict,
        model,
        t,
        J,
        Ht,
        HJ;
        trunc=trunc,
        seed=seed,
        return_details=return_details,
    )
end

function _testNN_impl(params_dict, model, t, J, Ht, HJ; trunc=nothing, seed::Integer=12345)
    """Evaluate a saved or newly initialized tJ neural-network wavefunction."""
    device = _resolve_device(params_dict; context="testNN")
    input_data = NN_inputs(t, J, model; trunc=trunc, device=device)
    M = model.M
    dim = size(input_data, 2)
    n_neurons = Int(_param(params_dict, "n_neurons"))
    dropout = Float64(_param(params_dict, "dropout", 0.0))
    exploit_symmetry = Bool(_param(params_dict, "exploit_symmetry", false))
    phase = 1

    H = t .* _slice_hamiltonian(Ht, dim) .+ J .* _slice_hamiltonian(HJ, dim)
    if exploit_symmetry
        H = _symmetry_reduce(H, dim; phase=phase)
        input_data = _symmetry_inputs(input_data, dim)
    end

    H = processH(H; device=device)
    neuralnet = _to_device(_load_or_initialize_network(params_dict, M, n_neurons, dropout, seed), device)
    scale = exploit_symmetry ? inv(sqrt(eltype(input_data)(2))) : one(eltype(input_data))
    psi_reduced = vec(neuralnet(input_data)) .* scale
    E0 = _scalar_float64(variational(H, psi_reduced))
    psi0 = _full_wavefunction(psi_reduced, dim; exploit_symmetry=exploit_symmetry, phase=phase)

    return psi0, E0
end

function testNN(params_dict, model, t, J, Ht, HJ; trunc=nothing, seed::Integer=12345)
    return _run_after_cuda_load_if_needed(
        _testNN_impl,
        params_dict,
        params_dict,
        model,
        t,
        J,
        Ht,
        HJ;
        trunc=trunc,
        seed=seed,
    )
end

end # module TJNetRC
