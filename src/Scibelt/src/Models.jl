module Models
export build_hydro, build_cel20_θ, build_cel20_θφ, build_fourier
include("./Discretization.jl")
include("./Utils.jl")
using .Discretization: build_1D_operators, build_upwind_operators, prealloc
using .Utils: dict2ntuple
using FiniteDiff: finite_difference_jacobian!, JacobianCache
using SparseDiffTools: matrix_colors
using LinearAlgebra, Parameters, DiffEqBase, SparseArrays


function make_model_builder(
    update_template::Function,
    nu::Integer,
    prealloc_symbols::Vector{Symbol},
    ints_symbols::Union{Vector{Symbol},Nothing}=nothing;
    upwind=false)
    function model_builder(nx, Δx::T, p₀, coeff_file=nothing; bc=:periodic, eval_sparsity=true) where {T}
        ops = Dict(zip([:Dx, :Dxx, :Dxxx], build_1D_operators(T, nx, Δx; bc=bc)))
        if upwind
            ops = merge(ops, Dict(zip([:Dxm, :Dxp], build_upwind_operators(T, nx, Δx; bc=bc))))
        end
        ops = dict2ntuple(ops)

        cache = dict2ntuple(prealloc(prealloc_symbols, zeros(T, nx)))
        if ~isnothing(coeff_file)
            coeff_df = CSV.File(coeff_file; delim=' ', ignorerepeated=true) |> DataFrame;
            build_interp(key) = LinearInterpolation(coeff_df[!, :H], coeff_df[!, key], extrapolation_bc=Flat())
            if isnothing(ints_symbols)
                ints_symbols = filter!(key -> key != :H, names(coeff_df))
            end
            ints = dict2ntuple(Dict((key, build_interp(key)) for key in ints_symbols))
            update!(dU::Vector{T}, U::Vector{T}, p, t) = update_template(
                dU, U, p, t, cache, ops, ints; openflow=ifelse(bc == :noflux, true, false)
                )
        else
            update!(dU::Vector{T}, U::Vector{T}, p, t) = update_template(
                dU, U, p, t, cache, ops; openflow=ifelse(bc == :noflux, true, false)
                )
        end

        if ~eval_sparsity
            return ODEFunction(update!)
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

        return ODEFunction(update!; jac=jac!, jac_prototype=sparsity)
    end
    return model_builder
end

function model_onesided_2eq(dU, U, p, t, c::NamedTuple, ops::NamedTuple, ints::NamedTuple; openflow=false)
    nu = 2
    @unpack Dx, Dxx, Dxxx = ops
    @unpack h, q = c
    @unpack dxq, dxxq, dxh, dxxh, dxxxh = c
    @unpack Re, We, Ct = p
    @unpack S, F, G, I, J, J1, K, K1, L, L1, M, M1 = ints
    if openflow
        @unpack ampl, freq = p
    end

    @. h = U[1:nu:end]
    @. q = U[2:nu:end]

    # if openflow
    #     h[1] = cospi(2 * t * freq) * ampl + 1
    #     q[1] = h[1]^3 / 3
    # end

    dth = @view dU[1:nu:end]
    dtq = @view dU[2:nu:end]

    mul!(dxq, Dx, q)
    mul!(dxxq, Dxx, q)

    mul!(dxh, Dx, h)
    mul!(dxxh, Dxx, h)
    mul!(dxxxh, Dxxx, h)


    @. dth = -dxq
    @. dtq = (
    (9dxh * q^2) / 7h^2
    - dxq * (
        17q / 7h
        + 11dxh / (4h * Re)
        )
    + (
        (13dxxq) / 12.0
        + ((-10 + 31dxh^2) * q) / 12h^2
        - (11dxxh * q) / 8h
        - (5h * (-1 + Ct * dxh - dxxxh * We)) / 18.0
        ) / Re
    )

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

    # if openflow
    #     dth[1] = 0
    #     dtq[1] = 0
    # end

    return
end


build_onesided_2eq = make_model_builder(
    model_onesided_2eq, 2,
    [:h, :q, :dxq, :dxxq, :dxh, :dxxh, :dxxxh],
    [:S, :F, :G, :I, :J, :J1, :K, :K1, :L, :L1, :M, :M1],
    ;
    )

end
