using Random
using Plots
using Statistics

const MAX_GEN = 1000
const DNA_LEN = 25
const POP_SIZE = 20
const MUTATION_RATE = 0.001
const EXPERIMENTS = 20;
const EXPERIMENT_NAME = "pop20mut0001";

mutable struct Individial
    fitness::Int64
    dna::Vector{Bool}
    Individial() = begin
        ind = new(0, rand(Bool, DNA_LEN))
        eval!(ind)
        #eval2!(ind)
        return ind
    end
end

const Population = Vector{Individial}

function eval!(ind::Individial)
    ind.fitness = sum(ind.dna)
end

#version that seek 10101010...
function eval2!(ind::Individial)
    ind.fitness = 0;
    for k in 1:DNA_LEN
        ind.fitness += (k % 2 == 1) ? ind.dna[k] : 1-ind.dna[k]
    end
end

function mutate!(ind::Individial)
    for k in 1:DNA_LEN
        ind.dna[k] = rand() > MUTATION_RATE ? ind.dna[k] : !ind.dna[k]
    end
end

function breed!(parents::Population, childs::Population)
    tot_fitness = sum(ind -> ind.fitness, parents)
    npairs = div(POP_SIZE, 2)
    Threads.@threads for n in 1:npairs
        parent1 = roulette_wheel(parents, rand(1:tot_fitness))
        parent2 = roulette_wheel(parents, rand(1:tot_fitness))
        child1 = childs[n]
        child2 = childs[n + npairs]
        cross!(parent1, parent2, child1, child2)
    end
end

function cross!(
        parent1::Individial, parent2::Individial,
        child1::Individial, child2::Individial
        )
    cross_point = rand(0:DNA_LEN)
    for k in 1:DNA_LEN
        child1.dna[k] = (k <= cross_point ? parent1.dna[k] : parent2.dna[k])
        child2.dna[k] = (k <= cross_point ? parent2.dna[k] : parent1.dna[k])
    end
end

function roulette_wheel(pop::Population, x::Int64)::Individial
    n = 0
    while x > 0 && n+1 < length(pop)
        n += 1
        x -= pop[n].fitness
    end
    return pop[n]
end

function eva()
    log_best = Vector{Int64}()
    log_avg = Vector{Float64}()
    log_q25 = Vector{Float64}()
    log_q75 = Vector{Float64}()
    best = 1
    #random initial population
    pop = [Individial() for k in 1:POP_SIZE]
    #buffer for next population
    childs = [Individial() for k in 1:POP_SIZE]
    #then iterate:
    for gen in 1:MAX_GEN
        #generate offspring
        breed!(pop, childs)
        #mutate and calculate fitness
        Threads.@threads for ind in childs
            mutate!(ind)
            eval!(ind)
            #eval2!(ind)
        end
        #kill the parents
        begin
            temp = pop
            pop = childs
            childs = temp
        end
        #find the best guy
        best = 1
        for k in 1:POP_SIZE
            best = (pop[k].fitness > pop[best].fitness) ? k : best
        end
        #save results
        push!(log_best, pop[best].fitness)
        push!(log_avg , sum(ind -> ind.fitness/length(pop), pop))
        push!(log_q25, quantile!([p.fitness for p in pop], 0.25))
        push!(log_q75, quantile!([p.fitness for p in pop], 0.75))
    end
    return (pop[best], log_best, log_avg, log_q25, log_q75)
end

function main()
    log_avg = zeros(MAX_GEN)
    log_best = zeros(MAX_GEN)
    log_q25 = zeros(MAX_GEN)
    log_q75 = zeros(MAX_GEN)
    for k in 1:EXPERIMENTS
        println("experiment #", k)
        @time (result, _log_best, _log_avg, _log_q25, _log_q75) = eva()
        println(result);
        log_avg += _log_avg/EXPERIMENTS
        log_best += _log_best/EXPERIMENTS
        log_q25 += _log_q25/EXPERIMENTS
        log_q75 += _log_q75/EXPERIMENTS
    end
    p = plot(ylims = (12.5,DNA_LEN), legend = :bottomright)
    plot!(log_avg, label = "average")
    plot!(log_best, label = "best")
    plot!(log_q25, fillrange = log_q75, alpha = 0.3, label = "first-third quantile")
    savefig(p, "sga-"*EXPERIMENT_NAME*".png")
end

main()