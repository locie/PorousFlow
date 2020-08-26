module Discretization
export build_upwind_operators, build_odefunc, build_1D_operators, prealloc
using LinearAlgebra, SparseArrays, FiniteDiff, DiffEqOperators

function prealloc(symbols, template)
    return Dict(symb => copy(template) for symb in symbols)
end

function build_upwind_operators_periodic(T, n₁, Δx)
    I₁ = sparse(I, n₁, n₁)
    Pₓₘ = sparse(zeros(T, n₁, n₁))
    Pₓₘ[1, n₁] = -1

    ∂ₓₘ = spdiagm(
            -1 => -ones(T, n₁ - 1),
            0 => ones(T, n₁)
        )
    Δₓₘ = (∂ₓₘ + Pₓₘ) / T(Δx)

    Pₓₚ = sparse(zeros(T, n₁, n₁))
    Pₓₚ[n₁, 1] = 1

    ∂ₓₚ = spdiagm(
            0 => -ones(T, n₁),
            1 => ones(T, n₁ - 1)
        )
    Δₓₚ = (∂ₓₚ + Pₓₚ) / T(Δx)
    return Δₓₘ, Δₓₚ
end

function build_1D_operators_periodic(T, n₁, Δx)
    I₁ = sparse(I, n₁, n₁)

    Pₓ = sparse(zeros(T, n₁, n₁))
    Pₓ[1, n₁] = T(-1 / 2)
    Pₓ[n₁, 1] = T(1 / 2)
    ∂ₓ = spdiagm(
            -1 => T(-1 / 2) .* ones(T, n₁ - 1),
            1 => T(1 / 2) .* ones(T, n₁ - 1)
        )
    Δₓ = (∂ₓ + Pₓ) / T(Δx)

    Pₓₓ = sparse(zeros(T, n₁, n₁))
    Pₓₓ[1, n₁] = T(1)
    Pₓₓ[n₁, 1] = T(1)
    ∂ₓₓ = spdiagm(
            -1 => T(1) .* ones(T, n₁ - 1),
            0 => T(-2) .* ones(T, n₁),
            1 => T(1) .* ones(T, n₁ - 1)
            )
    Δₓₓ = (∂ₓₓ + Pₓₓ) / T(Δx^2)

    Pₓₓₓ = sparse(zeros(T, n₁, n₁))
    Pₓₓₓ[1, n₁] = T(1)
    Pₓₓₓ[1, n₁ - 1] = T(-1 / 2)
    Pₓₓₓ[2, n₁] = T(-1 / 2)

    Pₓₓₓ[n₁, 1] = T(-1)
    Pₓₓₓ[n₁, 2] = T(1 / 2)
    Pₓₓₓ[n₁ - 1, 1] = T(1 / 2)
    ∂ₓₓₓ = spdiagm(
            -2 => T(-1 / 2) .* ones(T, n₁ - 2),
            -1 => T(1) .* ones(T, n₁ - 1),
            1 => T(-1) .* ones(T, n₁ - 1),
            2 => T(1 / 2) .* ones(T, n₁ - 2)
            )
    Δₓₓₓ = (∂ₓₓₓ + Pₓₓₓ) / T(Δx^3)

    return Δₓ, Δₓₓ, Δₓₓₓ
end

function build_upwind_operators(T, n₁, Δx; bc=:periodic)
    if bc == :periodic
        return build_upwind_operators_periodic(T, n₁, Δx)
    elseif bc == :noflux
        Q = Neumann0BC(Δx)
        Δₓₘ = sparse(UpwindDifference(1, 2, Δx, n₁, -1.0) * Q)[1]
        Δₓₚ = sparse(UpwindDifference(1, 2, Δx, n₁, 1.0) * Q)[1]
        return Δₓₘ, Δₓₚ
    end
    throw(ErrorException("Only periodic and noflux bc are implemented for now"))
end

function build_1D_operators(T, n₁, Δx; bc=:periodic)
    if bc == :periodic
        return build_1D_operators_periodic(T, n₁, Δx)
    elseif bc == :noflux
        Q = Neumann0BC(Δx)
        Δₓ = sparse(CenteredDifference(1, 2, Δx, n₁) * Q)[1]
        Δₓₓ = sparse(CenteredDifference(2, 2, Δx, n₁) * Q)[1]
        Δₓₓₓ = sparse(CenteredDifference(3, 2, Δx, n₁) * Q)[1]
        return Δₓ, Δₓₓ, Δₓₓₓ
    end
    throw(ErrorException("Only periodic and noflux bc are implemented for now"))
end

function num_sparsity(f!, U, args...)
    dU = similar(U)
    FiniteDiff.finite_difference_jacobian((du, u) -> f(du, u, args...), U)
end

end