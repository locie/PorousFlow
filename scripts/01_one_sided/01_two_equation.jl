using DrWatson
@quickactivate

##%
using Scibelt, DifferentialEquations, Printf, CSV, DataFrames
using Plots

##%
grid_pars = Dict(
    :L => 200,
    :Δx => 0.2
)

physical_pars = Dict(
    :θ => deg2rad(4.6),
    :Re => 50.0,
    :Ka => 769.8,
    :ξ => 1.0,
    :δ => 0.5,
    :εₕ => 0.78,
    :Da => 0.001
)

solver_pars = Dict(
    :tmax => 1000.0,
    :Δt => 2.0,
    :alg => :(Rosenbrock23(autodiff=false))
)

parameters = merge(grid_pars, physical_pars, solver_pars)


##%
function process_parameters(d)
    @unpack θ, Re, Ka, ξ, δ, εₕ, Da = d
    δᵦ = √(Da / εₕ)
    FSV = (1 - δ + δᵦ)^2 / 2
    Fr = sqrt(FSV * Re)

    Ct = cos(θ) / sin(θ)
    We = Ka / (Re^(5 / 3) * FSV^(1 / 3))
    (Re = Re, We = We, Ct = Ct, Fr = Fr, ξ = ξ, FSV = FSV, εₕ = εₕ, δ = δ, δᵦ = δᵦ, Da = Da);
end

function read_table(Da)
    coeff_file = @sprintf("data/inputs/one_sided/1phase_Da_%g", Da)
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
    p = process_parameters(d)
    odefunc, ints = build_onesided_2eq(N, Δx, p, coeff_df; eval_sparsity=true);

    I = ints[:I]
    h = @. cos(x * 2 * π / L) * 0.1 + 1
    q = @. I(h) * h^3
    U = vec_alternate(h, q);
    problem = ODEProblem(odefunc, U, tmax, p);
    at = range(problem.tspan..., step=2)
    @time sol = solve(
        problem, eval(alg);
        saveat=at, progress=progress, progress_steps=progress_steps
            )
    @unpack h, qₗ = unvec_alternate(
        Array(sol)' |> collect, length(x),
        h=[:, :, 1], qₗ=[:, :, 2]
        )
    return (t = sol.t, x = x), (h = h, qₗ = qₗ)
end

##%
coords, fields = run_simulation(parameters)

##%
resdir = (args...) -> datadir("sims", "one_sided", "two_eq", args...)
mkpath(resdir())
filename = resdir(savename(parameters, "csv"))
CSV.write(filename, tidify_results(coords, fields))

##%
plot(coords[:x], fields[:h][end, :])
