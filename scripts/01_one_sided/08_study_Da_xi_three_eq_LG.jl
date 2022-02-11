using DrWatson
@quickactivate

##%
using Scibelt, DifferentialEquations, Printf, CSV, DataFrames, ProgressMeter
using Sundials
using Plots

using TerminalLoggers: TerminalLogger
using Logging: global_logger

global_logger(TerminalLogger())

##%
grid_pars = Dict(
    :L => 2000.0,
    :Δx => 1.0,
    :buffer_len => 0.0,
)

physical_pars = Dict(
    :θ => deg2rad(4.6),
    :Re => 43.55,
    :Ka => 526.0,
    :ξ => 0.1,
    :δ => 0.5,
    :εₕ => 0.78,
    :Da => 0.01,
    :freq => 4.5,
    :nu => 6.27e-6,
    :g => 9.81,
    :amp => 0.08
)

solver_pars = Dict(
    :progress => true,
    :tmax => 1000.0,
    :Δt => 1.0,
    # :alg => :(SSPRK432())
    :alg => :(CVODE_BDF(linear_solver = :GMRES))
)

ps = dict_list(merge(grid_pars, physical_pars, solver_pars))

##%
function process_parameters(d, FSV)
    @unpack θ, Re, Ka, ξ, δ, εₕ, Da, nu, g, freq, amp = d
    δᵦ = √(Da / εₕ)
    Fr = √(FSV * Re)

    Ct = cos(θ) / sin(θ)
    We = Ka / (Re^(5 / 3) * FSV^(1 / 3))

    gx = g * sin(θ)
    uN = (Re * Fr^2 * nu * gx)^(1 / 3)
    hN = ((Re^2 / Fr^2) * nu^2 / gx)^(1 / 3)
    fforcage = freq * uN / hN

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
        :Da => Da,
        :fforcage => fforcage,
        :amp => amp
    )
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
    @unpack Da, ξ = d
    @unpack tmax, Δt, alg = d
    progress = get(d, :progress, false)
    progress_steps = get(d, :progress_steps, 50)

    x = 0:Δx:(L+buffer_len)
    N = length(x)

    coeff_df = read_table(Da)
    p = process_parameters(d, coeff_df[1, :FSV])
    @unpack fforcage, amp = p

    # inject signal on h, qₚ and qₗ will take the proper value on the model
    p[:signal] = (t) -> @. 1.0 # + amp * sin(2 * π * fforcage * t)
    odefunc, ints = build_onesided_3eq(N, Δx, p, coeff_df; bc = :noflux, eval_sparsity = false)

    I = ints[:I]
    PI = ints[:PI]
    h = ones(size(x))
    qₗ = @. I(h) * h^3
    qₚ = @. PI(h) * h^3
    U = vec_alternate(h, qₗ, qₚ)
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
resdir = (args...) -> datadir("sims", "one_sided", "three_eq_LG", args...)
mkpath(resdir())

# %%
let p = ps[1]
    global coords, fields
    filename = resdir(savename(p, "csv"; ignores = ["progress"]))
    # if isfile(filename)
    #     return filename
    # end
    coords, fields = run_simulation(p)
    CSV.write(filename, tidify_results(coords, fields))
end

# %%
plot(coords.x, fields.qₗ[end, :])

# %%
# @gif for i in 1:length(coords.t)
#     plot(coords.x, fields.h[i, :])
# end