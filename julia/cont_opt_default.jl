module cont_opt

using Random
using Plots
using Statistics
using BlackBoxOptimizationBenchmarking
using LinearAlgebra
using ForwardDiff
include("plot_utils.jl")

const MAX_GEN = 2000
const DIM = 10
const POP_SIZE = 100
const NELITES = 0
const GENS_PER_FRAME = 100
const X_LIM = 5.0

const MUT_PROB = 0.2
const CROSS_PROB = 0.8
const TOURNAMENT_LUCK_FACTOR = 0.0
const CLIMBER_SWAP = 0.0

#mutation variance
const MUT_VAR = 0.5

const NEXPERIMENTS = 20;
const EXPERIMENT_NAME = "default";

mutable struct Individial
    fitness::Float64
    obj_val::Float64
    mut_var::Float64
    x::Vector{Float64}
    climber::Bool
end

const Population = Vector{Individial}

function randu(lo::Float64, hi::Float64)
    return (hi - lo)*rand() + lo
end

function randu(lo::Float64, hi::Float64, dim::Integer)
    return [randu(lo, hi) for _ in 1:dim]
end

function spawn()::Individial
    return Individial(-Inf, Inf, MUT_VAR, randu(-X_LIM, X_LIM, DIM), false)
end

function eval!(ind::Individial, fun::BBOBFunction)
    val = fun(ind.x)
    ind.fitness = -val
    ind.obj_val = val - fun.f_opt + 1e-15
end

function mutate!(ind::Individial)
    ind.climber = (rand() < CLIMBER_SWAP ? !ind.climber : ind.climber)
    ind.mut_var = clamp(ind.mut_var*exp(MUT_VAR*randn()), 0.0, MUT_VAR)
    if rand() < MUT_PROB && !ind.climber
        for i in 1:DIM
            ind.x[i] = clamp(ind.x[i] + ind.mut_var*randn(), -X_LIM, X_LIM)
        end
    end
end

function climb!(ind::Individial, Dfun::Any)
    if ind.climber
        way = -Dfun(ind.x)
        way *= ind.mut_var/norm(way)
        ind.x += way
    end
end

function breed!(parents::Population, childs::Population)
    npairs = div(POP_SIZE, 2)
    Threads.@threads for n in 1:npairs
        parent1 = tournament_selection(parents, 2)
        parent2 = tournament_selection(parents, 2)
        child1 = childs[n]
        child2 = childs[n + npairs]
        copy!(child1, parent1)
        copy!(child2, parent2)
        cross!(parent1, parent2, child1, child2)
    end
end

function copy!(dest::Individial, src::Individial)
    dest.x .= src.x
    dest.climber = src.climber
    #dest.mut_var = src.mut_var
end

function cross!(
        parent1::Individial, parent2::Individial,
        child1::Individial, child2::Individial
        )
    if rand() < CROSS_PROB && !parent1.climber && !parent2.climber
        cross_point = rand(1:(DIM-1))
        for i in 1:cross_point
            child1.x[i] = parent2.x[i]
            child2.x[i] = parent1.x[i]
        end
    end
end

function tournament_selection(pop::Population, n = 2)::Individial
    winner = rand(pop)
    for _ in 2:n
        contestant = rand(pop)
        if (contestant.fitness > winner.fitness) && (rand() > TOURNAMENT_LUCK_FACTOR)
            winner = contestant
        end
    end
    return winner
end
        

function saveind(ind::Individial, path::String)
    open(path, "w") do file
        for i in 1:DIM
            write(file, string(ind.x[i])*"\n")
        end
    end
end

function eva(fun::BBOBFunction)
    Dfun = x -> ForwardDiff.gradient(fun, x)
    log_best = Vector{Float64}()
    #random initial population
    pop = [spawn() for _ in 1:POP_SIZE]
    #buffer for next population
    childs = [spawn() for _ in 1:POP_SIZE]
    #eval them
    for ind in pop
        eval!(ind, fun)
    end
    #then iterate:
    for gen in 1:MAX_GEN
        #generate offspring
        breed!(pop, childs)
        #mutate and calculate fitness
        Threads.@threads for ind in childs
            mutate!(ind)
            climb!(ind, Dfun)
            eval!(ind, fun)
        end

        #differential_mut!(childs)
        #share!(childs)

        #kill the parents, except elites
        Threads.@threads for n in NELITES+1:POP_SIZE
            temp = pop[n]
            pop[n] = childs[n]
            childs[n] = temp
        end
        #order by fitness
        sort!(pop, by = (ind -> ind.obj_val))
        #save results
        if (gen % GENS_PER_FRAME == 0)
            println(pop[1].obj_val, '\t', pop[1].fitness)
        end
        push!(log_best, pop[1].obj_val)
    end
    return (pop[1], log_best)
end

function main()
    bbob_funs = BlackBoxOptimizationBenchmarking.list_functions()
    for n in [2, 6, 8, 10]
        log = zeros(NEXPERIMENTS, MAX_GEN)
        for experiment in 1:NEXPERIMENTS
            println("experiment #", experiment)
            @time (result, log_best) = eva(bbob_funs[n])
            for gen in 1:length(log_best)
                log[experiment, gen] = log_best[gen]
            end
            #uncomment this if you want to save the best solution
            #saveind(result, "julia-partition-results/"*EXPERIMENT_NAME*"-"*string(k)*".best")
        end
        log_path = "julia-cont_opt-results/"*EXPERIMENT_NAME*string(n)*".log"
        fig_path = "julia-cont_opt-results/"*EXPERIMENT_NAME*string(n)*".pdf"
        plot_utils.save_log(POP_SIZE*(1:MAX_GEN), log, log_path)
        p = plot_utils.plot_log(log_path)
        savefig(p, fig_path)
    end
end

end