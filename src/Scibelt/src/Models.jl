module Models
export build_two_sided, build_onesided_2eq, build_onesided_3eq
using ..ModelHelpers
using UnPack, PreallocationTools


function kernel_onesided_2eq!(
    dh, dq,
    h, q,
    p, interps,
    grid, i
)

    @unpack Re, We, Ct, Fr, ξ = p
    if isa(Ct, AbstractArray)
        Ct = Ct[i]
    end

    @unpack S, F, G, I, J, J1, K, K1, L, L1, M, M1 = interps

    @fdiff_kernel (h, q) grid i begin
        dh[i] = -dx(q)
        dq[i] = 1 / S(h) * (
            -F(h) * q / h * dx(q)
            + G(h) * q^2 / h^2 * dx(h)
            + 1 / Fr^2 * (I(h) * (1 - Ct * dx(h) + We * Fr^2 * dxxx(h)) * h - q / h^2)
            + 1 / Re * (
                (J(h) + ξ * J1(h)) * q / h^2 * dx(h)^2
                -
                (K(h) + ξ * K1(h)) * dx(q) * dx(h) / h
                -
                (L(h) + ξ * L1(h)) * q / h * dxx(h)
                +
                (M(h) + ξ * M1(h)) * dxx(q)
            )
        )
    end
    return
end

function update_onesided_2eq!(dU, U, p, t, cache, interps, grid)
    @prealloc_block cache dU U t (h, q)

    if isa(grid.bc, NoFluxCondition)
        @unpack signal = p
        h[1] = signal(t)
        q[1] = interps.I(h[1]) * h[1]^3
    end

    for i in eachindex(h)
        kernel_onesided_2eq!(dh, dq, h, q, p, interps, grid, i)
    end

    if isa(grid.bc, NoFluxCondition)
        dh[1] = 0
        dqₗ[1] = 0
        dqₚ[1] = 0
    end

    return
end

function build_onesided_2eq(nx, Δx, interps; bc::T) where {T<:BoundaryCondition}
    @preallocate_dual h, q = zeros(nx)
    cache = (h = h, q = q)
    grid = Grid{T}(Δx, nx, bc)
    return (dU, U, p, t) -> update_onesided_2eq!(dU, U, p, t, cache, interps, grid)
end


function kernel_onesided_3eq!(
    dh, dqₗ, dqₚ,
    h, qₗ, qₚ,
    p, interps,
    grid, i
)

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

    @fdiff_kernel (h, qₗ, qₚ) grid i begin
        b = (1 - Ct * dx(h) + We * Fr^2 * dxxx(h))

        rhs1 = (
            -(F(h) * qₗ / h + Fp(h) * qₚ / h) * dx(qₗ)
            -
            (F1(h) * qₗ / h + F1p(h) * qₚ / h) * dx(qₚ)
            + (G(h) * (qₗ / h)^2 + Gp(h) * (qₚ / h)^2 + Gpp(h) * qₗ * qₚ / h^2) * dx(h)
            + 1 / Fr^2 * (I(h) * b * h - qₗ / h^2)
            + 1 / Re * (
                (J(h) + ξ * J1(h)) * qₗ * (dx(h) / h)^2
                +
                (Jp(h) + ξ * J1p(h)) * qₚ * (dx(h) / h)^2
                -
                (K(h) + ξ * K1(h)) * dx(qₗ) * dx(h) / h
                -
                (Kp(h) + ξ * K1p(h)) * dx(qₚ) * dx(h) / h
                -
                (L(h) + ξ * L1(h)) * qₗ / h * dxx(h)
                -
                (Lp(h) + ξ * L1p(h)) * qₚ / h * dxx(h)
                + (M(h) + ξ * M1(h)) * dxx(qₗ)
                + (Mp(h) + ξ * M1p(h)) * dxx(qₚ)
            )
        )

        rhs2 = (
            -(PF(h) * qₗ / h + PFp(h) * qₚ / h) * dx(qₗ)
            -
            (PF1(h) * qₗ / h + PF1p(h) * qₚ / h) * dx(qₚ)
            + (PG(h) * (qₗ / h)^2 + PGp(h) * (qₚ / h)^2 + PGpp(h) * qₗ * qₚ / h^2) * dx(h)
            + 1 / Fr^2 * (PI(h) * b * h - qₚ / h^2)
            + 1 / Re * (
                (PJ(h) + ξ * PJ1(h)) * qₗ * (dx(h) / h)^2
                +
                (PJp(h) + ξ * PJ1p(h)) * qₚ * (dx(h) / h)^2
                -
                (PK(h) + ξ * PK1(h)) * dx(qₗ) * dx(h) / h
                -
                (PKp(h) + ξ * PK1p(h)) * dx(qₚ) * dx(h) / h
                -
                (PL(h) + ξ * PL1(h)) * qₗ / h * dxx(h)
                -
                (PLp(h) + ξ * PL1p(h)) * qₚ / h * dxx(h)
                + (PM(h) + ξ * PM1(h)) * dxx(qₗ)
                + (PMp(h) + ξ * PM1p(h)) * dxx(qₚ)
            )
        )

        dh[i] = -(dx(qₗ) + dx(qₚ))
        dqₗ[i] = (-PSp(h) * rhs1 + Sp(h) * rhs2) / (PS(h) * Sp(h) - PSp(h) * S(h))
        dqₚ[i] = (PS(h) * rhs1 - S(h) * rhs2) / (PS(h) * Sp(h) - PSp(h) * S(h))
    end
    return
end

function update_onesided_3eq!(dU, U, p, t, cache, interps, grid)
    @prealloc_block cache dU U t (h, qₗ, qₚ)

    if isa(grid.bc, NoFluxCondition)
        @unpack signal = p
        h[1] = signal(t)
        qₗ[1] = interps.I(h[1]) * h[1]^3
        qₚ[1] = interps.PI(h[1]) * h[1]^3
    end

    for i in eachindex(h)
        kernel_onesided_3eq!(dh, dqₗ, dqₚ, h, qₗ, qₚ, p, interps, grid, i)
    end

    if isa(grid.bc, NoFluxCondition)
        dh[1] = 0
        dqₗ[1] = 0
        dqₚ[1] = 0
    end

    return
end


function build_onesided_3eq(nx, Δx, interps; bc::T) where {T<:BoundaryCondition}
    @preallocate_dual h_dc, qₗ_dc, qₚ_dc = zeros(nx)
    cache = (h = h_dc, qₗ = qₗ_dc, qₚ = qₚ_dc)
    grid = Grid{T}(Δx, nx, bc)
    return (dU, U, p, t) -> update_onesided_3eq!(dU, U, p, t, cache, interps, grid)

end

function kernel_twosided!(
    dh₊, dh₋, dq₊, dq₋, dvb,
    h₊, h₋, q₊, q₋, vb,
    p, interps,
    grid, i
)

    @unpack Re, We, Ct, Fr, ξ, Da = p
    if isa(Ct, AbstractArray)
        Ct = Ct[i]
    end

    @unpack S, F, G, I, P = interps
    @unpack J, K, L, M, T = interps
    @unpack J1, K1, L1, M1, T1 = interps

    @fdiff_kernel (h₊, h₋, q₊, q₋, vb) grid i begin
        b₊ = 1 - Ct * dx(h₊) + We * Fr^2 * dxxx(h₊)
        b₋ = 1 - Ct * dx(h₋) + We * Fr^2 * dxxx(h₋)

        dh₊[i] = -dx(q₊) + vb
        dh₋[i] = -dx(q₋) - vb

        dq₊[i] = (
            1 / S(h₊) * (
                -F(h₊) * q₊ / h₊ * dx(q₊)
                + G(h₊) * (q₊ / h₊)^2 * dx(h₊)
                + 1 / Fr^2 * (I(h₊) * b₊ * h₊ - q₊ / h₊^2)
                + P(h₊) * q₊ / h₊^2 * vb
                + 1 / Re * (
                    (J(h₊) + ξ * J1(h₊)) * q₊ / h₊^2 * dx(h₊)^2
                    -
                    (K(h₊) + ξ * K1(h₊)) * dx(q₊) * dx(h₊) / h₊
                    -
                    (L(h₊) + ξ * L1(h₊)) * q₊ / h₊ * dxx(h₊)
                    +
                    (M(h₊) + ξ * M1(h₊)) * dxx(q₊)
                    -
                    (T(h₊) + ξ * T1(h₊)) * dx(vb) / h₊
                )
            )
        )

        dq₋[i] = (
            1 / S(h₋) * (
                -F(h₋) * q₋ / h₋ * dx(q₋)
                + G(h₋) * (q₋ / h₋)^2 * dx(h₋)
                + 1 / Fr^2 * (I(h₋) * b₋ * h₋ - q₋ / h₋^2)
                -
                P(h₋) * q₋ / h₋^2 * vb
                +
                1 / Re * (
                    (J(h₋) + ξ * J1(h₋)) * q₋ / h₋^2 * dx(h₋)^2
                    -
                    (K(h₋) + ξ * K1(h₋)) * dx(q₋) * dx(h₋) / h₋
                    -
                    (L(h₋) + ξ * L1(h₋)) * q₋ / h₋ * dxx(h₋)
                    + (M(h₋) + ξ * M1(h₋)) * dxx(q₋)
                    + (T(h₋) + ξ * T1(h₋)) * dx(vb) / h₋
                )
            )
        )

        dvb[i] = 1 / Re * (dxx(vb) + (Re * We * (dxx(h₊) - dxx(h₋)) - 2 * δ / Da * vb * ξ) / (2 * (δ / εₕ - δ) + h₊ + h₋))
    end
    return
end

function update_twosided!(dU, U, p, t, cache, interps, grid)
    @prealloc_block cache dU U t (h₊, h₋, q₊, q₋, vb)

    if isa(grid.bc, NoFluxCondition)
        error("No Flux Boundary not implemented for this model")
        @unpack signal = p
        h[1] = signal(t)
        qₗ[1] = interps.I(h[1]) * h[1]^3
        qₚ[1] = interps.PI(h[1]) * h[1]^3
    end

    for i in eachindex(h)
        kernel!(
            dh₊, dh₋, dq₊, dq₋, dvb,
            h₊, h₋, q₊, q₋, vb,
            p, interps,
            grid, i
        )
    end

    if isa(grid.bc, NoFluxCondition)
        dh₊[1] = 0
        dh₋[1] = 0
        dq₊[1] = 0
        dq₋[1] = 0
        dvb[1] = 0
    end

    return
end

function build_twosided(nx, Δx, interps; bc::BC) where {BC<:BoundaryCondition}
    nu = 5
    @preallocate_dual h₊, h₋, q₊, q₋, vb = zeros(nx)
    cache = (h₊ = h₊, h₋ = h₋, q₊ = q₊, q₋ = q₋, vb = vb)
    grid = Grid{T}(Δx, nx, bc)
    return (dU, U, p, t) -> update_twosided!(dU, U, p, t, cache, interps, grid)

end

end