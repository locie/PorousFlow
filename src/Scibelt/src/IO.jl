module IO
    export save_sim, load_sim, coerce_params
    using NetCDF, DataStructures

    """
    coerce_params(d::Dict)::Dict

    Coerce keys and values of an dictionnary to ensure NetCDF compatibility.
    Symbol key and values are converted to String. Number values are keept. Bool are converted to Int.
    Other types are dropped (the user being warned in that case).

    See also: [`load_sim`](@ref), [`coerce_params`](@ref)
    """
    function coerce_params(d::Dict)::Dict
        coerced_d = Dict{String, Any}()
        for (key, value) in d
            key = String(key)
            if typeof(value) <: Integer
                coerced_d[key] = Int(value)
            elseif typeof(value) <: AbstractFloat
                coerced_d[key] = value
            elseif typeof(value) <: Symbol
                coerced_d[key] = String(value)
            else
                @warn "Key $key (type $(typeof(key))) cannot be converted, it will be dropped."
            end
        end
        return coerced_d
    end

    """
    save_sim(filename::String, coords, fields; atts=Dict(), overwrite=true) -> filename::String

    Save a simulation returned by an experiment to a NetCDF file. It can also
    save global attributes from a dict to the file.
    The dictionnary will be processed by `coerce_params` to ensure NetCDF compat.

    # Examples
    ```julia-repl
    julia> coords, fields = run_periodic(d);
    julia> save_sim("/tmp/my_sim.nc", coords, fields; atts=d)
    "/tmp/my_sim.nc"
    ```

    See also: [`load_sim`](@ref), [`coerce_params`](@ref)
    """
    function save_sim(filename::String, coords, fields; atts=Dict(), overwrite=true)::String
        rm(filename; force=overwrite)
        coords_info = zip(String.(keys(coords)), values(coords)) |> collect
        for fname in keys(fields)
            data = fields[fname]
            n = ndims(data)
            nccreate(filename, String(fname), Iterators.flatten(coords_info[1:n])...)
            ncwrite(fields[fname], filename, String(fname))
        end
        ncputatt(filename, "global", coerce_params(atts))
        filename
    end

    """
    load_sim(filename::String)::NamedTuple{(:fields, :atts), NTuple{2, AbstractDict{Symbol, Any}}}

    Load a simulation written on disk with [`save_sim`](@ref) and stored via NetCDF format.

    # Examples
    ```julia-repl
    julia> coords, fields = run_periodic(d);
    julia> save_sim("/tmp/my_sim.nc", coords, fields; atts=d)
    julia> load_sim("/tmp/my_sim.nc")
    (fields=..., atts=...)
    ```

    See also: [`save_sim`](@ref)
    """
    function load_sim(filename::String)::NamedTuple{(:fields, :atts), NTuple{2, AbstractDict}}
        d, atts = NetCDF.open(filename) do nc
            d = OrderedDict{Symbol, Array}(Symbol(namevar) => data |> collect for (namevar, data) in nc.vars)
            atts = Dict{Symbol, Any}(Symbol(key) => value for (key, value) in nc.gatts)
            d, atts
        end
        return (fields=d, atts=atts)
    end
end