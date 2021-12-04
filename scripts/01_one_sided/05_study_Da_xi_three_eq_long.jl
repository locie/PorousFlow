##%
using DrWatson
@quickactivate

##%
using Scibelt, DifferentialEquations, Printf, CSV, DataFrames, ProgressMeter
using Distributions, Sundials, Interpolations
using Plots

using TerminalLoggers: TerminalLogger
using Logging: global_logger

global_logger(TerminalLogger())

##%
grid_pars = Dict(
    :L => 2000.0,
    :Δx => 0.5,
    :buffer_len => 0.0,
)

physical_pars = Dict(
    :θ => deg2rad(4.6),
    :Re => 50.0,
    :Ka => 769.8,
    :ξ => [1.0, 1000.0],
    :δ => 0.5,
    :εₕ => 0.78,
    :Da => [0.001, 0.01]
)

solver_pars = Dict(
    :progress => true,
    :tmax => 50.0,
    :Δt => 2.0,
    :fnoise => 10.0,
    :alg => :(QNDF())
)

ps = dict_list(merge(grid_pars, physical_pars, solver_pars))

##%
function process_parameters(d, FSV)
    @unpack θ, Re, Ka, ξ, δ, εₕ, Da = d
    δᵦ = √(Da / εₕ)
    Fr = √(FSV * Re)
    Ct = cos(θ) / sin(θ)
    We = Ka / (Re^(5 / 3) * FSV^(1 / 3))
    Dict{Symbol,Any}(:Re => Re,
        :We => We,
        :Ct => Ct,
        :Fr => Fr,
        :ξ => ξ,
        :FSV => FSV,
        :εₕ => εₕ,
        :δ => δ,
        :δᵦ => δᵦ,
        :Da => Da)
end

function read_table(Da)
    coeff_file = @sprintf("data/inputs/one_sided/2phase_Da_%g", Da)
    coeff_df = CSV.File(coeff_file; delim = ' ', ignorerepeated = true) |> DataFrame
    sort!(coeff_df, :H)
    return coeff_df
end

##%
function run_simulation(d)
    @unpack L, buffer_len, Δx = d
    @unpack Da = d
    @unpack tmax, Δt, fnoise, alg = d
    progress = get(d, :progress, false)
    progress_steps = get(d, :progress_steps, 50)

    x = 0:Δx:(L+buffer_len)
    N = length(x)

    coeff_df = read_table(Da)
    p = process_parameters(d, coeff_df[1, :FSV])
    # p[:Ct] = fill(p[:Ct], size(x))
    # p[:Ct][x .> L] .= 1e-10
    t_sample = range(0, tmax, step = 1 / fnoise)
    p[:signal] = LinearInterpolation(t_sample, rand(Uniform(0.9, 1.1), size(t_sample)))
    odefunc, ints = build_onesided_3eq(N, Δx, p, coeff_df; bc = :noflux, chunk_size = 12)

    I = ints[:I]
    PI = ints[:PI]
    h = ones(size(x))
    qₗ = @. I(h) * h^3
    qₚ = @. PI(h) * h^3
    U = vec_alternate(h, qₗ, qₚ)
    problem = ODEProblem(odefunc, U, tmax, p)
    at = range(problem.tspan..., step = Δt)

    @time sol = solve(
        problem, eval(alg);
        saveat = at,
        progress = progress,
        progress_steps = progress_steps
    )
    @unpack h, qₗ, qₚ = unvec_alternate(
        Array(sol)' |> collect, length(x),
        h = [:, :, 1], qₗ = [:, :, 2], qₚ = [:, :, 3]
    )
    return (t = sol.t, x = x), (h = h, qₗ = qₗ, qₚ = qₚ)
end

# %%
resdir = (args...) -> datadir("sims", "one_sided_long", "three_eq", args...)
mkpath(resdir())

Threads.@threads for p in ps
    filename = resdir(savename(p, "csv"; ignores = ["progress"]))
    if isfile(filename)
        return filename
    end
    coords, fields = run_simulation(p)

    CSV.write(filename, tidify_results(coords, fields))
end

# %%