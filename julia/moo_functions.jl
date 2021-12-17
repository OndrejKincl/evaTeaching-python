module moo_functions

function quickavg(xs::Vector{Float64}; first::Int64 = 1, last::Int64 = length(xs))::Float64
    out = 0.0
    for k in first:last
        out += xs[k]
    end
    return out/(last - first + 1)
end

function zdt1(x::Vector{Float64})::Tuple{Float64, Float64}
    g  = 1.0 + 9.0*quickavg(x, first = 2)
    f1 = x[1]
    f2 = g * (1.0 - sqrt(f1/g))
    return (f1, f2)
end

function zdt2(x::Vector{Float64})::Tuple{Float64, Float64}
    g  = 1.0 + 9.0*quickavg(x, first = 2)
    f1 = x[1]
    f2 = g * (1.0 - (f1/g)^2)
    return (f1, f2)
end

function zdt3(x::Vector{Float64})::Tuple{Float64, Float64}
    g  = 1.0 + 9.0*quickavg(x, first = 2)
    f1 = x[1]
    f2 = g * (1.0 - sqrt(f1/g) - f1/g * sin(10*pi*f1))
    return (f1, f2)
end

function zdt4(x::Vector{Float64})::Tuple{Float64, Float64}
    g  = 1.0 + 10*(length(x)-1)
    for k in 2:length(x)
        g += x[k]^2 - 10*cos(4*pi*x[k])
    end
    f1 = x[1]
    f2 = g * (1.0 - sqrt(f1/g))
    return (f1, f2)
end

function zdt6(x::Vector{Float64})::Tuple{Float64, Float64}
    g  = 1.0 + 9.0*quickavg(x, first = 1)^0.25
    f1 = 1.0 - exp(-4.0*x[1])*sin(6*pi*x[1])^6
    f2 = g*(1.0 - (f1/g)^2)
    return (f1, f2)
end

function zdt()
    return Dict{Int8, Function}(
        1 => zdt1,
        2 => zdt2, 
        3 => zdt3, 
        4 => zdt4,
        6 => zdt6
    )
end

function opt_hvs()
    return Dict{Int8, Float64}(
        1 => 120.0 + 2/3,
        2 => 120.0 + 1/3,
        3 => 128.77811613069076060,
        4 => 120.0 + 2/3,
        6 => 117.51857519692037009,
    )
end

end