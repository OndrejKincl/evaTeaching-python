module lamarque

using Random
using Plots
using Statistics
using BlackBoxOptimizationBenchmarking
using LinearAlgebra
using ForwardDiff
using Parameters

include("plot_utils.jl")

const MAX_GEN = 200
const DIM = 10
const POP_SIZE = 100
const NELITES = 20
const GENS_PER_FRAME = 10
const LUCK_FACTOR = 0.1
const X_LIM = 5.0

const MUT_PROB = 0.2
const CROSS_PROB = 0.8

#mutation variance
const MUT_VAR = 0.1

const NEXPERIMENTS = 10
const EXPERIMENT_NAME = "lamarck_small_pop"
const COMPARE_TO = "diffevo_time"

@with_kw mutable struct Individial
    fitness::Float64 = -Inf
    obj_val::Float64 = +Inf
    mut_var::Float64 = MUT_VAR
    opt_step::Float64 = 0.8
    x::Vector{Float64} = randu(-X_LIM, X_LIM, DIM)
end

const Population = Vector{Individial}

function randu(lo::Float64, hi::Float64)
    return (hi - lo)*rand() + lo
end

function randu(lo::Float64, hi::Float64, dim::Integer)
    return [randu(lo, hi) for _ in 1:dim]
end

function eval!(ind::Individial, fun::BBOBFunction)
    val = fun(ind.x)
    ind.fitness = -val
    ind.obj_val = val - fun.f_opt + 1e-15
end

function mutate!(ind::Individial)
    ind.mut_var *= exp(MUT_VAR*randn())
    ind.mut_var = min(ind.mut_var, MUT_VAR)
   
    ind.opt_step *= exp(MUT_VAR*randn())
    ind.opt_step = min(ind.opt_step, 1.0)

    if rand() < MUT_PROB
        for i in 1:DIM
            ind.x[i] += ind.mut_var*randn()
            ind.x[i] = clamp(ind.x[i], -X_LIM, X_LIM)
        end
    end
end

function optimize!(ind::Individial, Dfun::Any, Hfun::Any)
    try
        b = Dfun(ind.x)
        A = Hfun(ind.x)
        ind.x -= ind.opt_step*A\b
    catch
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
    #dest.mut_var = src.mut_var
end

function cross!(
        parent1::Individial, parent2::Individial,
        child1::Individial, child2::Individial
        )
    if rand() < CROSS_PROB
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
        if (contestant.fitness > winner.fitness) && rand() > LUCK_FACTOR
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
    Hfun = x -> ForwardDiff.hessian(fun, x)
    log_best = Vector{Float64}()
    #random initial population
    pop = [Individial() for _ in 1:POP_SIZE]
    #buffer for next population
    childs = [Individial() for _ in 1:POP_SIZE]
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
            optimize!(ind, Dfun, Hfun)
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
    for n in [10]
        println()
        println("BBOB function #", n)
        println("============================================")
        log = zeros(NEXPERIMENTS, MAX_GEN)
        time_avg = 0.0
        for experiment in 1:NEXPERIMENTS
            println("experiment #", experiment)
            stats = @timed (result, log_best) = eva(bbob_funs[n])
            for gen in 1:length(log_best)
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
            p = plot_utils.plot_log(old_path, new_path, symbols = ["DE","LAM"])
            savefig(p, fig_path)
        else
            p = plot_utils.plot_log(new_path)
            savefig(p, fig_path)
        end      
    end
end

end