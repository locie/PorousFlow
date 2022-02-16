# %%
using DrWatson
using Scibelt, DifferentialEquations, Printf, CSV, DataFrames
using Plots, SparseArrays, ForwardDiff

using TerminalLoggers: TerminalLogger
using Logging: global_logger

global_logger(TerminalLogger())

# %%
grid_pars = Dict(
    :L => 100,
    :Δx => 0.1
)

physical_pars = Dict(
    :θ => deg2rad(90),
    :Re => 10.0,
    :Ka => 769.8,
    :ξ => 1.0,
    :δ => 0.5,
    :εₕ => 0.78,
    :Da => 0.01,
)

solver_pars = Dict(
    :progress => true,
    :tmax => 400.0,
    :Δt => 2.0,
    # :alg => :(CVODE_BDF(linear_solver = :GMRES)),
    :alg => :(QNDF()),
)

parameters = merge(grid_pars, physical_pars, solver_pars)

# %%
function process_parameters(d, FSV)
    @unpack θ, Re, Ka, ξ, δ, εₕ, Da = d
    δᵦ = √(Da / εₕ)
    Fr = √(FSV * Re)
    # Fr = √(Re / FSV)

    Ct = cos(θ) / sin(θ)
    We = Ka / (Re^(5 / 3) * FSV^(1 / 3))
    return Dict{Symbol,Any}(
        :Re => Re,
        :We => We,
        :Ct => Ct,
        :Fr => Fr,
        :ξ => ξ,
        :FSV => FSV,
        :εₕ => εₕ,
        :δ => δ,
        :δᵦ => δᵦ,
        :Da => Da
    )
end

function read_table(Da)
    coeff_file = @sprintf("data/inputs/one_sided/2phase_Da_%g", Da)
    coeff_df = CSV.File(coeff_file; delim = ' ', ignorerepeated = true) |> DataFrame
    sort!(coeff_df, :H)
    return coeff_df
end

# %%
function run_simulation(d)
    @unpack L, Δx = d
    @unpack θ, Re, Ka, ξ, δ, εₕ, Da = d
    @unpack tmax, Δt, alg = d
    progress = get(d, :progress, false)
    progress_steps = get(d, :progress_steps, 1)

    x = 0:Δx:L-Δx
    N = length(x)

    coeff_df = read_table(Da)
    interps = build_interps(coeff_df)
    p = process_parameters(d, coeff_df[1, :FSV])

    update! = build_onesided_3eq(N, Δx, interps; bc = PeriodicCondition())

    I = interps[:I]
    PI = interps[:PI]
    h = @. cos(x * 2 * π / L) * 0.1 + 1
    qₗ = @. I(h) * h^3
    qₚ = @. PI(h) * h^3
    U = vec_alternate(h, qₗ, qₚ)

    sparsity = sparse(ForwardDiff.jacobian((dU, U) -> update!(dU, U, p, 0.0), similar(U), U))
    odefunc = ODEFunction(update!, jac_prototype = sparsity)

    problem = ODEProblem(odefunc, U, tmax, p)
    at = range(problem.tspan..., step = Δt)

    cb = FunctionCallingCallback(
        (u, t, integrator) -> @info("Sim running", Da, ξ, t);
        funcat = 0:1:tmax |> collect
    )

    @time sol = solve(
        problem, eval(alg);
        saveat = at,
        progress = progress,
        progress_steps = progress_steps,
        callback = cb
    )
    @unpack h, qₗ, qₚ = unvec_alternate(
        Array(sol)' |> collect, length(x),
        h = [:, :, 1], qₗ = [:, :, 2], qₚ = [:, :, 3]
    )
    return (t = sol.t, x = x), (h = h, qₗ = qₗ, qₚ = qₚ)
end

# %%
coords, fields = run_simulation(parameters)

# %%
# save on "data/sims/periodic"
resdir = (args...) -> datadir("sims", "periodic", args...)
mkpath(resdir())
filename = resdir(savename(parameters, "csv"))
CSV.write(filename, tidify_results(coords, fields))

# %%
# shift value to center on max(h)
p = plot(coords[:x], circshift(fields[:h][end, :], -round(argmax(fields[:h][end, :]) / 2)))
p

# %%
@gif for i in 1:length(coords[:t])
    plot(coords[:x], fields[:h][i, :])
end