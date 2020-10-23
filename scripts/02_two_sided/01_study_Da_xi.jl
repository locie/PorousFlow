using DrWatson
@quickactivate

##%
using Scibelt, DifferentialEquations, Printf, CSV, DataFrames, ProgressMeter
using Plots

##%
grid_pars = Dict(
    :L => 100,
    :Δx => 0.2
)

physical_pars = Dict(
    :θ => deg2rad(4.6),
    :Re => 50.0,
    :Ka => 769.8,
    :ξ => [0.01, 1.0],
    :δ => 0.5,
    :εₕ => 0.78,
    :Da => [0.01, 0.001]
)

solver_pars = Dict(
    :progress => true,
    :tmax => 1000.0,
    :Δt => 2.0,
    :alg => :(Rosenbrock23(autodiff=false))
)

parameters = dict_list(merge(grid_pars, physical_pars, solver_pars))

##%
function process_parameters(d, FSV)
    @unpack θ, Re, Ka, ξ, δ, εₕ, Da = d
    δᵦ = √(Da / εₕ)
    Fr = √(FSV * Re)

    Ct = cos(θ) / sin(θ)
    We = Ka / (Re^(5 / 3) * FSV^(1 / 3))
    (Re = Re, We = We, Ct = Ct, Fr = Fr, ξ = ξ, FSV = FSV, εₕ = εₕ, δ = δ, δᵦ = δᵦ, Da = Da);
end

function read_table(Da)
    coeff_file = @sprintf("data/inputs/two_sided/two_sided_Da_%g", Da)
    coeff_df = CSV.File(coeff_file; delim=' ', ignorerepeated=true) |> DataFrame
    sort!(coeff_df, :H)
    return coeff_df
end

##%
function run_simulation(d)
    @unpack L, Δx = d
    @unpack θ, Re, Ka, ξ, δ, εₕ, Da = d
    @unpack tmax, Δt, alg = d
    progress = get(d, :progress, false)
    progress_steps = get(d, :progress_steps, 50)

    x = 0:Δx:L-Δx
    N = length(x)

    coeff_df = read_table(Da)
    p = process_parameters(d, coeff_df[:FSV][1])
    odefunc, ints = build_twosided(N, Δx, p, coeff_df; eval_sparsity=true);

    I = ints[:I]
    h₊ = @. 0.1 * cos(2 * π * x / L) + 1
    h₋ = @. 0.1 * cos(2 * π * (x - L / 2) / L) + 1
    q₊ = @. I(h₊) * h₊^3
    q₋ = @. I(h₋) * h₋^3
    vb = zeros(N)
    U = vec_alternate(h₊, h₋, q₊, q₋, vb)
    problem = ODEProblem(odefunc, U, tmax, p)
    at = range(problem.tspan..., step=2)
    @time sol = solve(
        problem, eval(alg);
        saveat=at, progress=progress, progress_steps=progress_steps, dtmin=1e-6
            )
    @unpack h₊, h₋, q₊, q₋, vb = unvec_alternate(
        Array(sol)' |> collect, length(x),
        h₊ = [:, :, 1], h₋ = [:, :, 2], q₊ = [:, :, 3], q₋ = [:, :, 4], vb = [:, :, 5]
        )
    return (t = sol.t, x = x), (h₊ = h₊, h₋ = h₋, q₊ = q₊, q₋ = q₋, vb = vb)
end

# %%
resdir = (args...) -> datadir("sims", "two_sided", args...)
mkpath(resdir())

# %%
@showprogress map(parameters) do p
    filename = resdir(savename(p, "csv"; ignores=["progress"]))
    if isfile(filename)
        return filename
    end
    coords, fields = run_simulation(p)

    CSV.write(filename, tidify_results(coords, fields))
end

# %%
