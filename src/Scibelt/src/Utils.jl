module Utils
export dict2ntuple, unzip, vec_alternate, unvec_alternate
export tidify_results, build_interps, @preallocate
using DataFrames, UnPack, Interpolations

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
function dict2ntuple(dict::Dict{String,T}) where {T}
    NamedTuple{Tuple(Symbol.(keys(dict)))}(values(dict))
end
function dict2ntuple(dict::Dict{Symbol,T}) where {T}
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

"""
    @preallocate caches... = template

build a preallocated array for each of the `caches` variable.

# Examples
```julia-repl
julia> @preallocate A, B = zeros(50, 50)
```
"""
macro preallocate(expr)
    expr.head != :(=) && error("Expression needs to be of form `a, b = c`")
    items, template = expr.args
    items = isa(items, Symbol) ? [items] : items.args
    kd = [:($key = $template) for key in items]
    kd_namedtuple = :(NamedTuple{Tuple($items)}(Tuple([$template for _ in $items])))
    kdblock = Expr(:block, kd...)
    expr = quote
        $kdblock
        $kd_namedtuple
    end
    return esc(expr)
end

function build_interps(coeff_df, ints_symbols = nothing)
    if ~isnothing(coeff_df)
        build_interp(key) = LinearInterpolation(coeff_df[!, :H], coeff_df[!, key], extrapolation_bc = Flat())
        if isnothing(ints_symbols)
            ints_symbols = filter!(key -> key != :H, names(coeff_df))
        end
        return dict2ntuple(Dict((key, build_interp(key)) for key in ints_symbols))
    end

    return ints
end

end