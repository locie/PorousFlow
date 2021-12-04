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
    :ξ => [0.001, 1., 10., 1000.],
    :δ => 0.5,
    :εₕ => 0.78,
    :Da => [0.001, 0.01]
)

solver_pars = Dict(
    :progress => false,
    :tmax => 1000.0,
    :Δt => 2.0,
    :alg => :(Rosenbrock23(autodiff=false))
)

ps = dict_list(merge(grid_pars, physical_pars, solver_pars))

##%
function process_parameters(d, FSV)
    @show d
    @unpack θ, Re, Ka, ξ, δ, εₕ, Da = d
    δᵦ = √(Da / εₕ)
    Fr = √(FSV * Re)
    # Fr = √(Re / FSV)

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
    p = process_parameters(d, coeff_df[:FSV][1])
    @show p
    odefunc, ints = build_onesided_2eq(N, Δx, p, coeff_df);

    I = ints[:I]
    h = @. cos(x * 2 * π / L) * 0.1 + 1
    q = @. I(h) * h^3
    U = vec_alternate(h, q);
    problem = ODEProblem(odefunc, U, tmax, p);
    at = range(problem.tspan..., step=2)

    cb = FunctionCallingCallback(
        (u, t, integrator) -> @info("Sim running", Da, ξ, t);
        func_everystep=true,
        )

    @time sol = solve(
        problem, eval(alg);
        saveat=at, progress=progress, progress_steps=progress_steps, callback=cb
            )
    @unpack h, q = unvec_alternate(
        Array(sol)' |> collect, length(x),
        h=[:, :, 1], q=[:, :, 2]
        )
    return (t = sol.t, x = x), (h = h, q = q)
end

# %%
resdir = (args...) -> datadir("sims", "one_sided", "two_eq", args...)
mkpath(resdir())

# %%
Threads.@threads for p in ps
    filename = resdir(savename(p, "csv"; ignores=["progress"]))
    if isfile(filename)
        return filename
    end
    coords, fields = run_simulation(p)

    CSV.write(filename, tidify_results(coords, fields))
end

# %%
