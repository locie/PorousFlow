##%
using DrWatson
@quickactivate

##%
using Scibelt, DifferentialEquations, Printf, CSV, DataFrames, ProgressMeter
using Distributions, Sundials, Interpolations
using WGLMakie

using TerminalLoggers: TerminalLogger
using Logging: global_logger

global_logger(TerminalLogger())

##%
grid_pars = Dict(
    :L => 2000.0,
    :Δx => 0.25,
    :buffer_len => 0.0,
)

physical_pars = Dict(
    :θ => deg2rad(4.6),
    :Re => 50.0,

    :Ka => 769.8,
    :ξ => 1000.0,
    :δ => 0.5,
    :εₕ => 0.78,
    :Da => 0.001
)

solver_pars = Dict(
    :progress => true,
    :tmax => 5000.0,
    :Δt => 2.0,
    :fnoise => 10.0,
    # :alg => :(QNDF()),
    :alg => :(CVODE_BDF(linear_solver=:GMRES))
)

p = merge(grid_pars, physical_pars, solver_pars)

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
coords, fields = let d=p
    @unpack L, buffer_len, Δx = d
    @unpack Da, ξ = d
    @unpack tmax, Δt, fnoise, alg = d
    progress = get(d, :progress, false)
    progress_steps = get(d, :progress_steps, 50)

    x = 0:Δx:(L+buffer_len)
    N = length(x)

    coeff_df = read_table(Da)
    p = process_parameters(d, coeff_df[1, :FSV])
    t_sample = range(0, tmax, step = 1 / fnoise)

    # inject signal on h, qₚ and qₗ will take the proper value on the model
    p[:signal] = LinearInterpolation(t_sample, rand(Uniform(0.9, 1.1), size(t_sample)))
    odefunc, ints = build_onesided_3eq(N, Δx, p, coeff_df; bc = :noflux, eval_sparsity = false)

    I = ints[:I]
    PI = ints[:PI]
    h = ones(size(x))
    qₗ = @. I(h) * h^3
    qₚ = @. PI(h) * h^3
    U = vec_alternate(h, qₗ, qₚ)
    problem = ODEProblem(odefunc, U, tmax, p)
    at = range(problem.tspan..., step = Δt)

    t_node = Node(0.0)
    U_node = Node(U)
    h_node = lift((U)->U[1:3:end], U_node)
    qₚ_node = lift((U)->U[2:3:end], U_node)
    qₗ_node = lift((U)->U[3:3:end], U_node)

    fig = Figure()
    lines(fig[1, 1], x, h_node, color=:black)
    lines(fig[2, 1], x, qₚ_node, color=:blue)
    lines!(fig[2, 1], x, qₗ_node, color=:red)

    display(fig)

    cb = FunctionCallingCallback(
        (u, t, integrator) -> U_node[] = u;
        # funcat=0:1:tmax |> collect,
        )

    @time sol = solve(
        problem, eval(alg);
        saveat = at,
        progress = progress,
        progress_steps = progress_steps,
        callback=cb
    )
    @unpack h, qₗ, qₚ = unvec_alternate(
        Array(sol)' |> collect, length(x),
        h = [:, :, 1], qₗ = [:, :, 2], qₚ = [:, :, 3]
    )
    (t = sol.t, x = x), (h = h, qₗ = qₗ, qₚ = qₚ)
end

# %%
resdir = (args...) -> datadir("sims", "one_sided_long", "three_eq", args...)
mkpath(resdir())