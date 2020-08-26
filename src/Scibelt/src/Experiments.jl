module Experiments
# export run_periodic

module Periodic
# using DifferentialEquations, Parameters, Scibelt
# export run_periodic

# function initial_conditions(x::Vector{M}, d::Dict) where {M}
#     @unpack L, B = d
#     h = @. cospi(2 * x / L) * 0.1 + 1.0
#     q = @. h^3 / 3
#     θ = @. 1 / (1 + B * h)
#     φ = zeros(M, length(x))
#     return (h = h, q = q, θ = θ, φ = φ)
# end

# function initial_conditions(x::Array{M}, y::Array{M}, d::Dict) where {M}
#     @unpack L, B = d
#     h = @. cospi(2 * x / L) * 0.1 + 1.0
#     q = @. h^3 / 3
#     θ = @. 1 / (1 + B * h)
#     φ = zeros(M, length(x))
#     T = @. y * (θ - 1) + 1
#     return (h = h, q = q, θ = θ, φ = φ, T = T)
# end

# function sim_periodic_hydro(d; kwargs...)
#     @unpack L, Δx, Re, Ct, We, tmax, alg, model = d
#     p = (Re = Re, We = We, Ct = Ct)

#     x = 0:Δx:L |> collect
#     nx = length(x)

#     @unpack h, q = initial_conditions(x, d)
#     U = vec_alternate(h, q)

#     odefunc = build_hydro(nx, Δx, p; eval_sparsity=get(d, :eval_sparsity, true))
#     problem = ODEProblem(
#             odefunc, U, tmax, p; kwargs...
#         )
#     @info "simulation ready, running...", p
#     sol = solve(problem, eval(alg), dtmin=1e-6; progress=get(d, :progress, false), progress_steps=1);
#     @info "simulation done", p
#     @unpack h, q, θ = unvec_alternate(
#         Array(sol)' |> collect, length(x),
#         h=[:, :, 1], q=[:, :, 2], θ=[:, :, 3]
#         )
#     return (t = sol.t, x = x), (h = h, q = q, θ = θ)
# end

# function sim_periodic_cel20_θ(d; kwargs...)
#     @unpack L, Δx, Re, Ct, We, Pe, B, tmax, alg, model = d
#     p = (Re = Re, We = We, Ct = Ct, Pe = Pe, B = B)

#     x = 0:Δx:L |> collect
#     nx = length(x)

#     @unpack h, q, θ = initial_conditions(x, d)
#     U = vec_alternate(h, q, θ)

#     odefunc = build_cel20_θ(nx, Δx, p; eval_sparsity=get(d, :eval_sparsity, true))
#     problem = ODEProblem(
#             odefunc, U, tmax, p; kwargs...
#         )
#     @info "simulation ready, running...", p
#     sol = solve(problem, eval(alg), dtmin=1e-6; progress=get(d, :progress, false), progress_steps=1);
#     @info "simulation done", p
#     @unpack h, q, θ = unvec_alternate(
#         Array(sol)' |> collect, length(x),
#         h=[:, :, 1], q=[:, :, 2], θ=[:, :, 3]
#         )
#     return (t = sol.t, x = x), (h = h, q = q, θ = θ)
# end

# function sim_periodic_cel20_θφ(d; kwargs...)
#     @unpack L, Δx, Re, Ct, We, Pe, B, tmax, alg, model = d
#     p = (Re = Re, We = We, Ct = Ct, Pe = Pe, B = B)

#     x = 0:Δx:L |> collect
#     nx = length(x)

#     @unpack h, q, θ, φ = initial_conditions(x, d)
#     U = vec_alternate(h, q, θ, φ)

#     odefunc = build_cel20_θφ(nx, Δx, p; eval_sparsity=get(d, :eval_sparsity, true))
#     problem = ODEProblem(
#             odefunc, U, tmax, p; kwargs...
#         )
#     @info "simulation ready, running...", p
#     sol = solve(problem, eval(alg), dtmin=1e-6; progress=get(d, :progress, false), progress_steps=1);
#     @info "simulation done", p
#     @unpack h, q, θ, φ = unvec_alternate(
#         Array(sol)' |> collect, length(x),
#         h=[:, :, 1], q=[:, :, 2], θ=[:, :, 3], φ=[:, :, 4]
#         )
#     return (x = x, t = sol.t), (h = h, q = q, θ = θ, φ = φ)
# end

# function sim_periodic_fourier(d; kwargs...)
#     @unpack L, Δx, ny, Re, Ct, We, Pe, B, tmax, alg = d
#     p = (Re = Re, We = We, Ct = Ct, Pe = Pe, B = B)

#     y = range(0.0, 1.0, length=ny)' |> collect
#     x = 0:Δx:L |> collect
#     nx = length(x)

#     @info "init simulation", p
#     @unpack h, q, T = initial_conditions(x, y, d)
#     U = vec_alternate(h, q, T)

#     odefunc = build_fourier(nx, ny, Δx, p; eval_sparsity=get(d, :eval_sparsity, false));
#     problem = ODEProblem(
#             odefunc, U, tmax, p; kwargs...
#         )
#     @info "simulation ready, running...", p
#     sol = solve(problem, eval(alg), dtmin=1e-6; progress=get(d, :progress, false), progress_steps=1);
#     @info "simulation done", p
#     @info "process solution"
#     @unpack h, q, T = unvec_alternate(
#         Array(sol)' |> collect, length(x),
#         h=[:, :, 1], q=[:, :, 2], T=[:, :, 3:3 + length(y) - 1])
#     return (x = x, t = sol.t), (h = h, q = q, T = T)
# end

# function run_periodic(d::Dict; kwargs...)
#     @unpack model = d
#     if model == :hydro
#         return sim_periodic_hydro(d; kwargs...)
#     elseif model == :cel20_θ
#         return sim_periodic_cel20_θ(d; kwargs...)
#     elseif model == :cel20_θφ
#         return sim_periodic_cel20_θφ(d; kwargs...)
#     elseif model == :fourier
#         return sim_periodic_fourier(d; kwargs...)
#     else throw(ArgumentError("model should be one of :cel20_θ, :cel20_θφ or :fourier"))
#     end
# end

end

# import .Periodic: run_periodic
end