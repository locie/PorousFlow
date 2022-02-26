module ModelHelpers
export BoundaryCondition, PeriodicCondition, NoFluxCondition, Grid
export dx, dxx, dxxx, @fdiff_kernel
export @prealloc_block, @preallocate, @preallocate_dual

using MacroTools, PreallocationTools
using MacroTools: postwalk, prewalk

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

macro fdiff_kernel(vars, grid, i, expr)
    expr = postwalk(expr) do x
        if x ∈ vars.args
            return :($x[i])
        end
        if @capture(x, f_(u_[i])) & (f ∈ (:dx, :dxx, :dxxx))
            return :($f($u, i, grid))
        end
        return x
    end
    esc(expr)
end

macro prealloc_block(cache, dU, U, t, vars)
    prealloc_block = Expr(:block)
    for (i, var) in enumerate(vars.args)
        push!(prealloc_block.args, :($var = $PreallocationTools.get_tmp($cache[Symbol($(String(var)))], $U * $t)))
        push!(prealloc_block.args, :($var .= $U[$i:$(length(vars.args)):end]))
        push!(prealloc_block.args, :($(Symbol("d" * string(var))) = view($dU, $i:$(length(vars.args)):length($U))))
    end
    return esc(prealloc_block)
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
    kdblock = Expr(:block, [:($key = $template) for key in items]...)
    return esc(kdblock)
end

macro preallocate_dual(expr)
    expr.head != :(=) && error("Expression needs to be of form `a, b = c`")
    items, template = expr.args
    items = isa(items, Symbol) ? [items] : items.args
    kdblock = Expr(:block, [:($key = $PreallocationTools.dualcache($template)) for key in items]...)
    return esc(kdblock)
end

end