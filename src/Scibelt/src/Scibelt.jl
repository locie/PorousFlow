module Scibelt
    using Reexport

    include("./Discretization.jl")
    @reexport using .Discretization

    include("./Utils.jl")
    @reexport using .Utils

    include("./IO.jl")
    @reexport using .IO

    include("./Models.jl")
    @reexport using .Models
end
