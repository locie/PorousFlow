# %%
using DrWatson
using Scibelt, DifferentialEquations, Printf, CSV, DataFrames
using SparseArrays, ForwardDiff
using GLMakie

using TerminalLoggers: TerminalLogger
using Logging: global_logger

global_logger(TerminalLogger())

# %%
grid_pars = Dict(
    :L => 800,
    :Δx => 0.5
)

physical_pars = Dict(
    :θ => deg2rad(90),
    :Re => 10.0,
    :Ka => 769.8,
    :ξ => 1.0,
    :δ => 0.5,
    :εₕ => 0.78,
    :Da => 0.01,
    :amp => 0.1,
    :freq => 0.1,
)

solver_pars = Dict(
    :progress => true,
    :tmax => 1000.0,
    :Δt => 0.5,
    :alg => :(Trapezoid()),
)

parameters = merge(grid_pars, physical_pars, solver_pars)

# %%
function process_parameters(d, FSV)
    @unpack θ, Re, Ka, ξ, δ, εₕ, Da = d
    δᵦ = √(Da / εₕ)
    Fr = √(FSV * Re)

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
@unpack L, Δx = parameters
@unpack θ, Re, Ka, ξ, δ, εₕ, Da = parameters
@unpack tmax, Δt, alg = parameters
@unpack freq, amp = parameters
progress = get(parameters, :progress, false)
progress_steps = get(parameters, :progress_steps, 1)

x = 0:Δx:L-Δx
N = length(x)

coeff_df = read_table(Da)
interps = build_interps(coeff_df)
p = process_parameters(parameters, coeff_df[1, :FSV])

p[:signal] = (t) -> 1.0 + amp * sin(2 * π * freq * t)
update! = build_onesided_3eq(N, Δx, interps; bc = NoFluxCondition())

I = interps[:I]
PI = interps[:PI]
h = fill(1.0, N)
qₗ = @. I(h) * h^3
qₚ = @. PI(h) * h^3
U = vec_alternate(h, qₗ, qₚ)

# %%
sparsity = sparse(ForwardDiff.jacobian((dU, U) -> update!(dU, U, p, 0.0), similar(U), U))
odefunc = ODEFunction(update!, jac_prototype = sparsity)

problem = ODEProblem(odefunc, U, tmax, p)
at = range(problem.tspan..., step = Δt)

# %%
# setup real_time vizu, see https://makie.juliaplots.org for more complex plots
# can be long before it displays
# observables are object that trigger a callback when changed
# Makie use them to update the plot : modifying U_node[] with a new U value
# will change h_node and update the vizu.
t_node = Observable(0.0)
U_node = Observable(U)
h_node = @lift($U_node[1:3:end])
fig = lines(
    x, h_node,
    axis = (title = @lift("t = $(round($t_node, digits = 1))"),)
)
display(fig)

# %%
function update_viz!(t, u)
    t_node[] = t
    U_node[] = u
end

viz_update_cb = FunctionCallingCallback(
    (u, t, integrator) -> update_viz!(t, u),
    funcat = at, # comment this line to update at every step
)

# %%

@time sol = solve(
    problem, eval(alg);
    saveat = at,
    progress = progress,
    progress_steps = progress_steps,
    callback = viz_update_cb
)
@unpack h, qₗ, qₚ = unvec_alternate(
    Array(sol)' |> collect, length(x),
    h = [:, :, 1], qₗ = [:, :, 2], qₚ = [:, :, 3]
)
coords = (t = sol.t, x = x)
fields = (h = h, qₗ = qₗ, qₚ = qₚ)

# %%
resdir = (args...) -> datadir("sims", "open", args...)
mkpath(resdir())
filename = resdir(savename(parameters, "csv"))
CSV.write(filename, tidify_results(coords, fields))
