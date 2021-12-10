module diffevo

using BlackBoxOptimizationBenchmarking
using Plots
using Statistics
include("plot_utils.jl")

const CR = 0.941343846536435
const F = 0.5184306753627251
const NP = 100
const DIM = 10
const MAX_GEN = 800
const NEXPERIMENTS = 10
const EXPERIMENT_NAME = "diffevo_time";
const COMPARE_TO = "nothing";



const GENS_PER_FRAME = 100

mutable struct Agent
    pos::Vector{Float64}
    obj::Float64
    fit::Float64
    Agent() = begin
        pos = [randu(-4.0, 4.0) for _ in 1:DIM]
        return new(pos, Inf, -Inf)
    end
end

function randu(lo::Float64, hi::Float64)
    return (hi - lo)*rand() + lo
end

function copy!(dest::Agent, src::Agent)
    dest.pos .= src.pos
    dest.obj = src.obj
    dest.fit = src.fit
end

const Population = Vector{Agent}

function find_best(pop::Population)::Agent
    if length(pop) == 0
        @error("Empty population!")
    end
    winner = pop[1]
    for x in pop
        if x.fit > winner.fit
            winner = x
        end
    end
    return winner
end

function eval!(x::Agent, fun::BBOBFunction)
    val = fun(x.pos)
    x.fit = -val
    x.obj = val - fun.f_opt + 1e-15
end

function breed!(parents::Population, children::Population, fun::BBOBFunction)
    Threads.@threads for i in 1:NP
        x = parents[i]
        y = children[i]
        a = rand(parents)
        b = rand(parents)
        c = rand(parents)
        R = rand(1:DIM)
        for i in 1:DIM
            if R == i || rand() < CR
                y.pos[i] = a.pos[i] + F*(b.pos[i] - c.pos[i])
            else
                y.pos[i] = x.pos[i]
            end
        end
        eval!(y, fun)
    end
end

function exchange!(parents::Population, children::Population)
    for i in 1:NP
        if parents[i].fit <= children[i].fit
            temp = children[i]
            children[i] = parents[i]
            parents[i] = temp
        end
    end
end

function eva(fun::BBOBFunction)
    log_best = Vector{Float64}()
    #random initial population
    parents = [Agent() for _ in 1:NP]
    children = [Agent() for _ in 1:NP]
    best = parents[1]

    #eval them
    for x in parents
        eval!(x, fun)
    end
    #then iterate:
    for gen in 1:MAX_GEN
        breed!(parents, children, fun)
        exchange!(parents, children)
        best = find_best(parents)
        #save results
        if (gen % GENS_PER_FRAME == 0)
            println(gen, '\t', best.obj)
        end
        push!(log_best, best.obj)
    end
    return (best, log_best)
end


function main()
    bbob_funs = BlackBoxOptimizationBenchmarking.list_functions()
    for n in [2,6,8,10]
        log = zeros(NEXPERIMENTS, MAX_GEN)
        time_avg = 0.0
        for experiment in 1:NEXPERIMENTS
            println("experiment #", experiment)
            stats = @timed (result, log_best) = eva(bbob_funs[n])
            for gen in 1:MAX_GEN
                log[experiment, gen] = log_best[gen]
            end
            time_avg += stats.time/NEXPERIMENTS
            #uncomment this if you want to save the best solution
            #saveind(result, "julia-partition-results/"*EXPERIMENT_NAME*"-"*string(k)*".best")
        end
        t = LinRange(0.0, time_avg, MAX_GEN)
        new_path = "julia-cont_opt-results/"*EXPERIMENT_NAME*string(n)*".log"
        old_path = "julia-cont_opt-results/"*COMPARE_TO*string(n)*".log"        
        fig_path = "julia-cont_opt-results/"*EXPERIMENT_NAME*string(n)*".pdf" 
        plot_utils.save_log(t, log, new_path)
        if ispath(old_path)
            p = plot_utils.plot_log(old_path, new_path)
            savefig(p, fig_path)
        else
            p = plot_utils.plot_log(new_path)
            savefig(p, fig_path)
        end
    end
end

end