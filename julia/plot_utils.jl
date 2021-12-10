module plot_utils
using Statistics
using Plots

function save_log(x::AbstractArray, logs::Matrix{Float64}, path::String)
    (_, imax) = size(logs)
    q25 = zeros(imax)
    q50 = zeros(imax)
    q75 = zeros(imax)
    for i in 1:imax
        q25[i] = quantile(logs[:, i], 0.25)
        q50[i] = quantile(logs[:, i], 0.50)
        q75[i] = quantile(logs[:, i], 0.75)
    end
    open(path, "w") do file
        for i in 1:imax
            write(file, string(Float64(x[i])," ", q25[i], " ", q50[i], " ", q75[i], "\n"))
        end
    end
end 

function plot_log(paths::String...; xlims = (-Inf,Inf), ylims = (-Inf,Inf), symbols = 'A':'Z')
    p = plot(legend = :topright, 
                 #xlabel = "fitness evals",
                 xlabel = "time (s)",
                 ylabel = "objective value",
                 yaxis =:log,
                 ylims = (1e-16, 1e+4))
    colors = Plots.palette(:tab10)
    for i in 1:length(paths)
        path = paths[i]
        denom = length(paths) > 1 ? string(symbols[i],": ") : ""
        open(path, "r") do file
            q25 = Vector{Float64}()
            q50 = Vector{Float64}()
            q75 = Vector{Float64}()
            t = Vector{Float64}()
            while !eof(file)
                (_t, _q25, _q50, _q75) = parse.(Float64, split(readline(file)))
                push!(t, _t)
                push!(q25, _q25)
                push!(q50, _q50)
                push!(q75, _q75)
            end
            yticks = [10.0^(-2*n) for n in -3:7]
            plot!(t, q50, label = denom*"median", lw = 2.0,
                xlims = xlims, ylims = ylims, yticks = yticks, linecolor = colors[i])
            plot!(t, q25, fillrange = q75, alpha = 0.3, lw = 2.0,
                label = denom*"first-third quantile",
                xlims = xlims, ylims = ylims, yticks = yticks,
                linecolor = colors[i], fillcolor = colors[i])
        end
    end
    return p
end

export save_log
export plot_log
end