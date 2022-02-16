module Models
export build_onesided_3eq, BoundaryCondition, PeriodicCondition, NoFluxCondition, Grid
using ..Utils: dict2ntuple, @preallocate
using UnPack, PreallocationTools

abstract type BoundaryCondition end
struct PeriodicCondition <: BoundaryCondition end
struct NoFluxCondition <: BoundaryCondition end

abstract type AbstractGrid end
struct Grid{BC<:BoundaryCondition} <: AbstractGrid
    δ::Float64
    N::Int64
    bc::BC
end

Grid(δ, N, bc::BC) where {BC<:BoundaryCondition} = Grid{BC}(δ, N, bc)

idx(i, grid::Grid{PeriodicCondition}) = mod1(i, grid.N)
idx(i, grid::Grid{NoFluxCondition}) = clamp(i, 1, grid.N)

dx(u, i, grid) = (0.5u[idx(i + 1, grid)] - 0.5u[idx(i - 1, grid)]) / grid.δ
dxx(u, i, grid) = (u[idx(i + 1, grid)] - 2u[i] + u[idx(i - 1, grid)]) / grid.δ^2
dxxx(u, i, grid) = (0.5u[idx(i + 2, grid)] - u[idx(i + 1, grid)] + u[idx(i - 1, grid)] - 0.5u[idx(i - 2, grid)]) / grid.δ^3

function kernel!(
    dh, dqₗ, dqₚ,
    h, qₗ, qₚ,
    p, interps,
    grid, i
)

    if isa(grid.bc, NoFluxCondition)
        if i == 1
            dh[i] = 0
            dqₗ[i] = 0
            dqₚ[i] = 0
            return
        # outlet advection : a tester, semble pas très concluant
        # elseif i >= grid.N - 1
        #     dh[i] = -(dx(qₗ, i, grid) + dx(qₚ, i, grid))
        #     lidx = i - 1
        #     ridx = idx(i + 1, grid)
        #     dqₗ[i] = -(
        #         (qₗ[ridx] + qₚ[ridx]) * qₗ[ridx] / h[ridx]
        #         -
        #         (qₗ[lidx] + qₚ[lidx]) * qₗ[lidx] / h[lidx]
        #     ) / 2grid.δ
        #     dqₚ[i] = -(
        #         (qₗ[ridx] + qₚ[ridx]) * qₚ[ridx] / h[ridx]
        #         -
        #         (qₗ[lidx] + qₚ[lidx]) * qₚ[lidx] / h[lidx]
        #     ) / 2grid.δ
        #     return
        # end
    end

    @unpack Re, We, Ct, Fr, ξ = p
    if isa(Ct, AbstractArray)
        Ct = Ct[i]
    end

    @unpack S, F, F1, G, I, J, J1, K, K1 = interps
    @unpack L, L1, M, M1, PS, PF, PF1, PG, PI = interps
    @unpack PJ, PJ1, PK, PK1, PL, PL1, PM, PM1 = interps
    @unpack Sp, Fp, F1p, Gp, Jp, J1p, Kp, K1p = interps
    @unpack Lp, L1p, Mp, M1p, PSp, PFp, PF1p = interps
    @unpack PGp, PJp, PJ1p, PKp, PK1p, PLp, PL1p = interps
    @unpack PMp, PM1p, Gpp, PGpp = interps

    dxqₗ = dx(qₗ, i, grid)
    dxxqₗ = dxx(qₗ, i, grid)

    dxqₚ = dx(qₚ, i, grid)
    dxxqₚ = dxx(qₚ, i, grid)

    dxh = dx(h, i, grid)
    dxxh = dxx(h, i, grid)
    dxxxh = dxxx(h, i, grid)

    b = (1 - Ct * dxh + We * Fr^2 * dxxxh)

    rhs1 = (
        -(F(h[i]) * qₗ[i] / h[i] + Fp(h[i]) * qₚ[i] / h[i]) * dxqₗ
        -
        (F1(h[i]) * qₗ[i] / h[i] + F1p(h[i]) * qₚ[i] / h[i]) * dxqₚ
        + (G(h[i]) * (qₗ[i] / h[i])^2 + Gp(h[i]) * (qₚ[i] / h[i])^2 + Gpp(h[i]) * qₗ[i] * qₚ[i] / h[i]^2) * dxh
        + 1 / Fr^2 * (I(h[i]) * b * h[i] - qₗ[i] / h[i]^2)
        + 1 / Re * (
            (J(h[i]) + ξ * J1(h[i])) * qₗ[i] * (dxh / h[i])^2
            +
            (Jp(h[i]) + ξ * J1p(h[i])) * qₚ[i] * (dxh / h[i])^2
            -
            (K(h[i]) + ξ * K1(h[i])) * dxqₗ * dxh / h[i]
            -
            (Kp(h[i]) + ξ * K1p(h[i])) * dxqₚ * dxh / h[i]
            -
            (L(h[i]) + ξ * L1(h[i])) * qₗ[i] / h[i] * dxxh
            -
            (Lp(h[i]) + ξ * L1p(h[i])) * qₚ[i] / h[i] * dxxh
            + (M(h[i]) + ξ * M1(h[i])) * dxxqₗ
            + (Mp(h[i]) + ξ * M1p(h[i])) * dxxqₚ
        )
    )

    rhs2 = (
        -(PF(h[i]) * qₗ[i] / h[i] + PFp(h[i]) * qₚ[i] / h[i]) * dxqₗ
        -
        (PF1(h[i]) * qₗ[i] / h[i] + PF1p(h[i]) * qₚ[i] / h[i]) * dxqₚ
        + (PG(h[i]) * (qₗ[i] / h[i])^2 + PGp(h[i]) * (qₚ[i] / h[i])^2 + PGpp(h[i]) * qₗ[i] * qₚ[i] / h[i]^2) * dxh
        + 1 / Fr^2 * (PI(h[i]) * b * h[i] - qₚ[i] / h[i]^2)
        + 1 / Re * (
            (PJ(h[i]) + ξ * PJ1(h[i])) * qₗ[i] * (dxh / h[i])^2
            +
            (PJp(h[i]) + ξ * PJ1p(h[i])) * qₚ[i] * (dxh / h[i])^2
            -
            (PK(h[i]) + ξ * PK1(h[i])) * dxqₗ * dxh / h[i]
            -
            (PKp(h[i]) + ξ * PK1p(h[i])) * dxqₚ * dxh / h[i]
            -
            (PL(h[i]) + ξ * PL1(h[i])) * qₗ[i] / h[i] * dxxh
            -
            (PLp(h[i]) + ξ * PL1p(h[i])) * qₚ[i] / h[i] * dxxh
            + (PM(h[i]) + ξ * PM1(h[i])) * dxxqₗ
            + (PMp(h[i]) + ξ * PM1p(h[i])) * dxxqₚ
        )
    )

    dh[i] = -(dxqₗ + dxqₚ)
    dqₗ[i] = (-PSp(h[i]) * rhs1 + Sp(h[i]) * rhs2) / (PS(h[i]) * Sp(h[i]) - PSp(h[i]) * S(h[i]))
    dqₚ[i] = (PS(h[i]) * rhs1 - S(h[i]) * rhs2) / (PS(h[i]) * Sp(h[i]) - PSp(h[i]) * S(h[i]))
    return
end

function update_template!(dU, U, p, t, grid, interps, (h, qₗ, qₚ))
    nu = 3

    h = get_tmp(h, U)
    qₗ = get_tmp(qₗ, U)
    qₚ = get_tmp(qₚ, U)

    h .= U[1:nu:end]
    qₗ .= U[2:nu:end]
    qₚ .= U[3:nu:end]

    dh = @view dU[1:nu:end]
    dqₗ = @view dU[2:nu:end]
    dqₚ = @view dU[3:nu:end]

    if isa(grid.bc, NoFluxCondition)
        @unpack signal = p
        h[1] = signal(t)
        qₗ[1] = interps.I(h[1]) * h[1]^3
        qₚ[1] = interps.PI(h[1]) * h[1]^3
    end

    for i in eachindex(h)
        kernel!(dh, dqₗ, dqₚ, h, qₗ, qₚ, p, interps, grid, i)
    end

    return
end


function build_onesided_3eq(nx, Δx, interps; bc::T) where {T<:BoundaryCondition}
    # see PreallocationTools: allow AutoDiff with preallocation
    @preallocate h_dc, qₗ_dc, qₚ_dc = dualcache(zeros(nx))
    grid = Grid{T}(Δx, nx, bc)

    return (dU, U, p, t) -> update_template!(dU, U, p, t, grid, interps, (h_dc, qₗ_dc, qₚ_dc))
end
end