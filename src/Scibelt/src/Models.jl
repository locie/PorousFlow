module Models
export build_onesided_2eq, build_onesided_3eq, build_twosided
include("./Discretization.jl")
include("./Utils.jl")
using .Discretization: build_1D_operators, build_upwind_operators, prealloc
using .Utils: dict2ntuple
using FiniteDiff: finite_difference_jacobian!, JacobianCache
using SparseDiffTools: matrix_colors
using Interpolations, LinearAlgebra, Parameters, DiffEqBase, SparseArrays, DataFrames


function make_model_builder(
    update_template::Function,
    nu::Integer,
    prealloc_symbols::Vector{Symbol},
    ints_symbols::Union{Vector{Symbol},Nothing}=nothing;
    upwind=false)
    function model_builder(nx, Δx::T, p₀, coeff_df=nothing; bc=:periodic, eval_sparsity=true) where {T}
        ops = Dict(zip([:Dx, :Dxx, :Dxxx], build_1D_operators(T, nx, Δx; bc=bc)))
        if upwind
            ops = merge(ops, Dict(zip([:Dxm, :Dxp], build_upwind_operators(T, nx, Δx; bc=bc))))
        end
        ops = dict2ntuple(ops)

        cache = dict2ntuple(prealloc(prealloc_symbols, zeros(T, nx)))
        if ~isnothing(coeff_df)
            build_interp(key) = LinearInterpolation(coeff_df[!, :H], coeff_df[!, key], extrapolation_bc=Flat())
            if isnothing(ints_symbols)
                ints_symbols = filter!(key -> key != :H, names(coeff_df))
            end
            ints = dict2ntuple(Dict((key, build_interp(key)) for key in ints_symbols))

        else
            ints = NamedTuple()
        end
        update! = (dU::Vector{T}, U::Vector{T}, p, t) -> update_template(
            dU, U, p, t, cache, ops, ints; openflow=ifelse(bc == :noflux, true, false)
            )
        if !eval_sparsity
            return ODEFunction(update!), ints
        end

        sparsity = zeros(T, (nu * nx, nu * nx))
        fake_U = rand(T, nu * nx)
        finite_difference_jacobian!(sparsity, (dU, U) -> update!(dU, U, p₀, 0.0), fake_U)
        sparsity = sparse(sparsity)
        colors = matrix_colors(sparsity);
        sparsecache = JacobianCache(fake_U, colorvec=colors, sparsity=sparsity)
        function jac!(J, U, p, t)
            f = (dU, U) -> update!(dU, U, p, t)
            finite_difference_jacobian!(J, f, U, sparsecache)
            return
        end

        return ODEFunction(update!; jac=jac!, jac_prototype=sparsity), ints
    end
    return model_builder
end

function model_onesided_2eq(dU, U, p, t, c::NamedTuple, ops::NamedTuple, ints::NamedTuple; openflow=false)
    nu = 2
    @unpack Dx, Dxx, Dxxx = ops
    @unpack h, q = c
    @unpack dxq, dxxq, dxh, dxxh, dxxxh = c
    @unpack Re, We, Ct, Fr, ξ, FSV = p
    @unpack S, F, G, I, J, J1, K, K1, L, L1, M, M1 = ints
    if openflow
        @unpack ampl, freq = p
    end

    @. h = U[1:nu:end]
    @. q = U[2:nu:end]

    dth = @view dU[1:nu:end]
    dtq = @view dU[2:nu:end]

    mul!(dxq, Dx, q)
    mul!(dxxq, Dxx, q)

    mul!(dxh, Dx, h)
    mul!(dxxh, Dxx, h)
    mul!(dxxxh, Dxxx, h)


    @. dth = -dxq
    @. dtq = 1 / S(h) * (
        - F(h) * q / h * dxq
        + G(h) * q^2 / h^2 * dxh
        + 1 / Fr^2 * (I(h) * (1 - Ct * dxh + We * Fr^2 * dxxxh) * h - q / h^2)
        + 1 / Re * (
            (J(h) + ξ * J1(h)) * q / h^2 * dxh^2
            - (K(h) + ξ * K1(h)) * dxq * dxh / h
            - (L(h) + ξ * L1(h)) * q / h * dxxh
            + (M(h) + ξ * M1(h)) * dxxq
        )
    )

    return
end


function model_onesided_3eq(dU, U, p, t, c::NamedTuple, ops::NamedTuple, ints::NamedTuple; openflow=false)
    nu = 3
    @unpack Dx, Dxx, Dxxx = ops
    @unpack h, qₗ, qₚ = c
    @unpack dxh, dxxh, dxxxh, dxqₗ, dxxqₗ, dxqₚ, dxxqₚ, rhs1, rhs2, b = c
    @unpack Re, We, Ct, Fr, ξ, FSV = p

    @unpack S, F, F1, G, I, J, J1, K, K1 = ints
    @unpack L, L1, M, M1, PS, PF, PF1, PG, PI = ints
    @unpack PJ, PJ1, PK, PK1, PL, PL1, PM, PM1 = ints
    @unpack Sp, Fp, F1p, Gp, Jp, J1p, Kp, K1p = ints
    @unpack Lp, L1p, Mp, M1p, PSp, PFp, PF1p = ints
    @unpack PGp, PJp, PJ1p, PKp, PK1p, PLp, PL1p = ints
    @unpack PMp, PM1p, Gpp, PGpp = ints

    @. h = U[1:nu:end]
    @. qₗ = U[2:nu:end]
    @. qₚ = U[3:nu:end]

    dth = @view dU[1:nu:end]
    dtqₗ = @view dU[2:nu:end]
    dtqₚ = @view dU[3:nu:end]

    mul!(dxh, Dx, h)
    mul!(dxxh, Dxx, h)
    mul!(dxxxh, Dxxx, h)

    mul!(dxqₗ, Dx, qₗ)
    mul!(dxxqₗ, Dxx, qₗ)

    mul!(dxqₚ, Dx, qₚ)
    mul!(dxxqₚ, Dxx, qₚ)

    @. b = (1 - Ct * dxh + We * Fr^2 * dxxxh)

    @. rhs1 = (
        - (F(h) * qₗ / h + Fp(h) * qₚ / h) * dxqₗ
        - (F1(h) * qₗ / h + F1p(h) * qₚ / h) * dxqₚ
        + (G(h) * (qₗ / h)^2 + Gp(h) * (qₚ / h)^2 + Gpp(h) * qₗ * qₚ / h^2) * dxh
        + 1 / Fr^2 * (I(h) * b * h - qₗ / h^2)
        + 1 / Re * (
            (J(h) + ξ * J1(h)) * qₗ * (dxh / h)^2
            + (Jp(h) + ξ * J1p(h)) * qₚ * (dxh / h)^2
            - (K(h) + ξ * K1(h)) * dxqₗ * dxh / h
            - (Kp(h) + ξ * K1p(h)) * dxqₚ * dxh / h
            - (L(h) + ξ * L1(h)) * qₗ / h * dxxh
            - (Lp(h) + ξ * L1p(h)) * qₚ / h * dxxh
            + (M(h) + ξ * M1(h)) * dxxqₗ
            + (Mp(h) + ξ * M1p(h)) * dxxqₚ
            )
        )

    @. rhs2 = (
        - (PF(h) * qₗ / h + PFp(h) * qₚ / h) * dxqₗ
        - (PF1(h) * qₗ / h + PF1p(h) * qₚ / h) * dxqₚ
        + (PG(h) * (qₗ / h)^2 + PGp(h) * (qₚ / h)^2 + PGpp(h) * qₗ * qₚ / h^2) * dxh
        + 1 / Fr^2 * (PI(h) * b * h - qₚ / h^2)
        + 1 / Re * (
            (PJ(h) + ξ * PJ1(h)) * qₗ * (dxh / h)^2
            + (PJp(h) + ξ * PJ1p(h)) * qₚ * (dxh / h)^2
            - (PK(h) + ξ * PK1(h)) * dxqₗ * dxh / h
            - (PKp(h) + ξ * PK1p(h)) * dxqₚ * dxh / h
            - (PL(h) + ξ * PL1(h)) * qₗ / h * dxxh
            - (PLp(h) + ξ * PL1p(h)) * qₚ / h * dxxh
            + (PM(h) + ξ * PM1(h)) * dxxqₗ
            + (PMp(h) + ξ * PM1p(h)) * dxxqₚ
            )
        )

    @. dth = -(dxqₗ + dxqₚ)
    @. dtqₗ = (-PSp(h) * rhs1 + Sp(h) * rhs2) / (PS(h) * Sp(h) - PSp(h) * S(h))
    @. dtqₚ = (PS(h) * rhs1 - S(h) * rhs2) / (PS(h) * Sp(h) - PSp(h) * S(h))

    return
end

function model_twosided(dU, U, p, t, c::NamedTuple, ops::NamedTuple, ints::NamedTuple; openflow=false)
    nu = 5
    @unpack Dx, Dxx, Dxxx = ops

    @unpack h₊, h₋, q₊, q₋, vb = c
    @unpack dxh₊, dxxh₊, dxxxh₊, dxh₋, dxxh₋, dxxxh₋ = c
    @unpack dxq₊, dxxq₊, dxq₋, dxxq₋ = c
    @unpack dxvb, dxxvb = c
    @unpack b₊, b₋ = c

    @unpack Re, We, Ct, Fr, ξ, δ, εₕ, Da = p

    @unpack S, F, G, I, P = ints
    @unpack J, K, L, M, T = ints
    @unpack J1, K1, L1, M1, T1 = ints

    @. h₊ = U[1:nu:end]
    @. h₋ = U[2:nu:end]
    @. q₊ = U[3:nu:end]
    @. q₋ = U[4:nu:end]
    @. vb = U[5:nu:end]

    dth₊ = @view dU[1:nu:end]
    dth₋ = @view dU[2:nu:end]
    dtq₊ = @view dU[3:nu:end]
    dtq₋ = @view dU[4:nu:end]
    dtvb = @view dU[5:nu:end]

    mul!(dxh₊, Dx, h₊)
    mul!(dxxh₊, Dxx, h₊)
    mul!(dxxxh₊, Dxxx, h₊)

    mul!(dxh₋, Dx, h₋)
    mul!(dxxh₋, Dxx, h₋)
    mul!(dxxxh₋, Dxxx, h₋)

    mul!(dxq₊, Dx, q₊)
    mul!(dxxq₊, Dxx, q₊)

    mul!(dxq₋, Dx, q₋)
    mul!(dxxq₋, Dxx, q₋)

    mul!(dxvb, Dx, vb)
    mul!(dxxvb, Dxx, vb)

    @. b₊ = 1 - Ct * dxh₊ + We * Fr^2 * dxxxh₊
    @. b₋ = 1 - Ct * dxh₋ + We * Fr^2 * dxxxh₋

    @. dth₊ = -dxq₊ + vb
    @. dth₋ = -dxq₋ - vb

    @. dtq₊ = (
        1 / S(h₊) * (
            - F(h₊) * q₊ / h₊ * dxq₊
            + G(h₊) * (q₊ / h₊)^2 * dxh₊
            + 1 / Fr^2 * (I(h₊) * b₊ * h₊ - q₊ / h₊^2)
            + P(h₊) * q₊ / h₊^2 * vb
            + 1 / Re * (
                  (J(h₊) + ξ * J1(h₊)) * q₊ / h₊^2 * dxh₊^2
                - (K(h₊) + ξ * K1(h₊)) * dxq₊ * dxh₊ / h₊
                - (L(h₊) + ξ * L1(h₊)) * q₊ / h₊ * dxxh₊
                + (M(h₊) + ξ * M1(h₊)) * dxxq₊
                - (T(h₊) + ξ * T1(h₊)) * dxvb / h₊
            )
        )
    )

    @. dtq₋ = (
        1 / S(h₋) * (
            - F(h₋) * q₋ / h₋ * dxq₋
            + G(h₋) * (q₋ / h₋)^2 * dxh₋
            + 1 / Fr^2 * (I(h₋) * b₋ * h₋ - q₋ / h₋^2)
            - P(h₋) * q₋ / h₋^2 * vb
            + 1 / Re * (
                  (J(h₋) + ξ * J1(h₋)) * q₋ / h₋^2 * dxh₋^2
                - (K(h₋) + ξ * K1(h₋)) * dxq₋ * dxh₋ / h₋
                - (L(h₋) + ξ * L1(h₋)) * q₋ / h₋ * dxxh₋
                + (M(h₋) + ξ * M1(h₋)) * dxxq₋
                + (T(h₋) + ξ * T1(h₋)) * dxvb / h₋
            )
        )
    )

    @. dtvb = 1 / Re * (dxxvb + (Re * We * (dxxh₊ - dxxh₋) - 2 * δ / Da * vb * ξ) / (2 * (δ / εₕ - δ) + h₊ + h₋))

    return
end

build_onesided_2eq = make_model_builder(
    model_onesided_2eq, 2,
    [:h, :q, :dxq, :dxxq, :dxh, :dxxh, :dxxxh],
    [:S, :F, :G, :I, :J, :J1, :K, :K1, :L, :L1, :M, :M1]
    )

build_onesided_3eq = make_model_builder(
    model_onesided_3eq, 3,
    [:h, :qₗ, :qₚ, :dxqₗ, :dxxqₗ, :dxqₚ, :dxxqₚ, :dxh, :dxxh, :dxxxh, :rhs1, :rhs2, :b],
    [
        :S, :F, :F1, :G, :I, :J, :J1, :K, :K1,
        :L,:L1, :M, :M1, :PS, :PF, :PF1, :PG,
        :PI, :PJ, :PJ1, :PK, :PK1, :PL, :PL1,
        :PM, :PM1, :Sp, :Fp, :F1p, :Gp, :Jp,
        :J1p, :Kp, :K1p, :Lp, :L1p, :Mp, :M1p,
        :PSp, :PFp, :PF1p, :PGp, :PJp, :PJ1p,
        :PKp, :PK1p, :PLp, :PL1p, :PMp, :PM1p, :Gpp, :PGpp
    ]
    )

    build_twosided = make_model_builder(
        model_twosided, 5,
        [:h₊, :h₋, :q₊, :q₋, :vb, :dxh₊, :dxxh₊, :dxxxh₊, :dxh₋, :dxxh₋, :dxxxh₋, :dxq₊, :dxxq₊, :dxq₋, :dxxq₋, :dxvb, :dxxvb, :b₊, :b₋],
        [:S, :F, :G, :I, :P, :J, :K, :L, :M, :T, :J1, :K1, :L1, :M1, :T1]
    )
end