module diffevo

using BlackBoxOptimizationBenchmarking
using Plots
using Statistics
include("plot_utils.jl")

const NP = 100
const DIM = 10
const MAX_GEN = 2000
const TOL = 1e-12
const NEXPERIMENTS = 5
#const EXPERIMENT_NAME = "diffevo";
#const COMPARE_TO = "adapt";

mutable struct DE_parameters
    CR::Float64
    F::Float64
    loss::Float64
end

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

function breed!(parents::Population, children::Population, fun::BBOBFunction, par::DE_parameters)
    Threads.@threads for i in 1:NP
        x = parents[i]
        y = children[i]
        a = rand(parents)
        b = rand(parents)
        c = rand(parents)
        R = rand(1:DIM)
        for i in 1:DIM
            if R == i || rand() < par.CR
                y.pos[i] = a.pos[i] + par.F*(b.pos[i] - c.pos[i])
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

function eva(fun::BBOBFunction, par::DE_parameters)
    #log_best = Vector{Float64}()
    #random initial population
    parents = [Agent() for _ in 1:NP]
    children = [Agent() for _ in 1:NP]
    #best = parents[1]

    #eval them
    for x in parents
        eval!(x, fun)
    end
    #then iterate:
    gen = 1
    while gen < MAX_GEN
        breed!(parents, children, fun, par)
        exchange!(parents, children)
        best = find_best(parents)
        if best.obj < TOL
            break
        end
        gen += 1
        #save results
        #if (gen % GENS_PER_FRAME == 0)
        #    println(gen, '\t', best.obj)
        #end
        #push!(log_best, best.obj)
    end
    return gen
end

function copy!(dest::DE_parameters, src::DE_parameters)
    dest.F .= src.F
    dest.CR = src.CR
end

function tournament_selection(pop::Vector{DE_parameters}, n = 2)::DE_parameters
    winner = rand(pop)
    for _ in 2:n
        contestant = rand(pop)
        if (contestant.loss < winner.loss)
            winner = contestant
        end
    end
    return winner
end

function main()
    META_NGENS = 10
    META_NP = 20
    META_NELITES = 1
    bbob_funs = [BlackBoxOptimizationBenchmarking.list_functions()[i] for i in [2,6,8,10]]
    parents = [DE_parameters(rand(), rand(), Inf) for _ in 1:META_NP]
    children = [DE_parameters(rand(), rand(), Inf) for _ in 1:META_NP]
    for gen in 1:META_NGENS
        println("gen = ", gen)
        for ind in parents
            ind.loss = 0.0
            for fun in bbob_funs, ex in 1:NEXPERIMENTS
                ind.loss += eva(fun, ind)/(NEXPERIMENTS*length(bbob_funs))
                print("x")
            end
            println()
        end
        sort!(parents, by = (ind -> ind.loss))
        println("best = ", parents[1].loss, " F = ", parents[1].F, " CR = ", parents[1].CR)
        for k in 1:div(META_NP,2)
            p1 = tournament_selection(parents)
            p2 = tournament_selection(parents)
            ch1 = children[k]
            ch2 = children[k + div(META_NP,2)]
            ch1.F = 0.5*(p1.F + p2.F) + 0.05*randn()
            ch2.CR = 0.5*(p1.CR + p2.CR) + 0.05*randn()
        end
        for n in META_NELITES+1:META_NP
            temp = parents[n]
            parents[n] = children[n]
            children[n] = temp
        end
    end
end

end