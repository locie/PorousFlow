module Utils
export dict2ntuple
export unzip, vec_alternate, unvec_alternate
export tidify_results, compute_adim_number
export isoutlier_IQR, isoutlier_zscore
export filteroutlier_zscore!, filteroutlier_IQR!
export filteroutlier_zscore, filteroutlier_IQR
using DataFrames, UnPack, StatsBase

"""
    @dict vars...
Create a dictionary out of the given variables that has as keys the variable
names and as values their values.
Notice: `@dict a b` is the correct way to call the macro. `@dict a, b`
is incorrect. If you want to use commas you have to do `@dict(a, b)`.
## Examples
```jldoctest; setup = :(using DrWatson)
julia> ω = 5; χ = "test"; ζ = π/3;
julia> @dict ω χ ζ
Dict{Symbol,Any} with 3 entries:
  :ω => 5
  :χ => "test"
  :ζ => 1.0472
```
"""
macro dict(vars...)
    return esc_dict_expr_from_vars(vars)
end

"""
    esc_dict_expr_from_vars(vars)
Transform a `Tuple` of `Symbol` into a dictionary where each `Symbol` in `vars`
defines a key-value pair. The value is obtained by evaluating the `Symbol` in
the macro calling environment.
This should only be called when producing an expression intended to be returned by a macro.
"""
function esc_dict_expr_from_vars(vars)
    expr = Expr(:call, :Dict)
    for i in 1:length(vars)
        push!(expr.args, :($(QuoteNode(vars[i])) => $(esc(vars[i]))))
    end
    return expr
end

"""
    dict2ntuple(dict) -> ntuple
Convert a dictionary (with `Symbol` or `String` as key type) to
a `NamedTuple`.
"""
function dict2ntuple(dict::Dict{String,T}) where T
    NamedTuple{Tuple(Symbol.(keys(dict)))}(values(dict))
end
function dict2ntuple(dict::Dict{Symbol,T}) where T
    NamedTuple{Tuple(keys(dict))}(values(dict))
end

unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))

function vec_alternate(var...)
    vec(hcat(var...)') |> collect
end

function unvec_alternate(U::Vector{T}, nx::Integer; unpack_info...) where {T}
    rU = reshape(U, :, nx)' |> collect
    results = Dict{Symbol,Array{T}}()
    for (var, slice) in unpack_info
        results[var] = rU[slice...]
    end
    return NamedTuple{Tuple(Symbol.(keys(results)))}(values(results))
end

function unvec_alternate(U::Matrix{T}, nx::Integer; unpack_info...) where {T}
    rU = permutedims(reshape(U, size(U, 1), :, nx), (1, 3, 2)) |> collect
    results = Dict{Symbol,Array{T}}()
    for (var, slice) in unpack_info
        results[var] = rU[slice...]
    end
    return NamedTuple{Tuple(Symbol.(keys(results)))}(values(results))
end

function tidify_results(coords::NamedTuple, fields::NamedTuple)
    gridded_coords = unzip(collect(Iterators.product(values(coords)...))) .|> vec
    gridded_fields = values(fields) .|> vec
    return DataFrame(
        hcat(gridded_coords..., gridded_fields...),
        vcat(keys(coords)..., keys(fields)...)
        )
end

function compute_adim_number(d)
    @unpack ρ, ν, σ_inf, cp, β, Pr, Re, Bi, g = d
    lᵥ = (ν^2 / (g * sin(β)))^(1 / 3)
    tᵥ = (ν / (g * sin(β))^2)^(1 / 3)
    Ct = cot(β)
    Kaᵥ = σ_inf / (ρ * g^(1 / 3) * ν^(4 / 3))
    Ka = Kaᵥ / (sin(β)^(1 / 3))
    bhN = (3 * Re)^(1 / 3) * lᵥ
    Pe = Pr * Re
    We = Ka / ((3 * Re)^(2 / 3))
    B = Bi * (3 * Re)^(1 / 3)
    tf = 1 / ((tᵥ * lᵥ) / bhN)
    lf = 1 / bhN
    θ_flat = 1 / (1 + B)
    ϕ_flat = - B * θ_flat
    merge(d, @dict lᵥ tᵥ Ct Kaᵥ Ka Pe We B tf lf θ_flat ϕ_flat)
end

function isoutlier_zscore(y::AbstractArray; zscore_threshold=3.0)
    return abs.(zscore(y)) .> zscore_threshold
end

function filteroutlier_zscore!(y::AbstractArray; zscore_threshold=3.0, drop=false)
    y[isoutlier_zscore(y, zscore_threshold=zscore_threshold)] .= NaN
    if drop
        filter!((x)->~isnan(x), y)
    end
end

function filteroutlier_zscore(y::AbstractArray; zscore_threshold=3.0, drop=false)
    y = copy(y)
    filteroutlier_zscore!(y; zscore_threshold=zscore_threshold, drop=drop)
    return y
end

function isoutlier_IQR(y::AbstractArray; nIQR=1.5)
    Q1, Q3 = quantile(y, [0.25, 0.75])
    IQR = abs(Q3 - Q1)
    return @. (y < (Q1 - nIQR * IQR)) | (y > (Q3 + nIQR * IQR))
end

function filteroutlier_IQR!(y::AbstractArray; nIQR=1.5, drop=false)
    y[isoutlier_IQR(y; nIQR=nIQR)] .= NaN
    if drop
        filter!((x)->~isnan(x), y)
    end
end

function filteroutlier_IQR(y::AbstractArray; nIQR=1.5, drop=false)
    y = copy(y)
    filteroutlier_IQR!(y; nIQR=nIQR, drop=drop)
    return y
end
end