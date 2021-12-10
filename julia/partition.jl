module partition

using Random
using Plots
using Statistics

const MAX_GEN = 500
const NBANDITS = 10
const POP_SIZE = 500
const NELITES = 50
const GENS_PER_FRAME = 50

#mutation probabilities
const SWAP_PROB = 0.5
const BALANCE_PROB = 0.5
const CROSS_PROB = 0.1     

const NEXPERIMENTS = 1;
const EXPERIMENT_NAME = "easy";

mutable struct Individial
    fitness::Float64
    obj_val::Int64
    item_owner::Vector{Int64}
    bandit_wealth::Vector{Int64}
    Individial(nitems::Int64) = begin
        fitness = -Inf
        obj_val = typemax(Int64)
        bandit_wealth = zeros(Int64, NBANDITS)
        item_owner = rand(1:NBANDITS, nitems)
        return new(fitness, obj_val, item_owner, bandit_wealth)
    end
end

const Population = Vector{Individial}

function eval!(ind::Individial, item_price::Vector{Int64})
    nitems = length(ind.item_owner)
    #reset all bandits
    for b in 1:NBANDITS
        ind.bandit_wealth[b] = 0
    end
    #calculate their wealth
    for i in 1:nitems
        ind.bandit_wealth[ind.item_owner[i]] += item_price[i]
    end
    #find fitness and objective value
    ind.obj_val = maximum(ind.bandit_wealth) - minimum(ind.bandit_wealth)
    
    #calc fitness
    avg = sum(ind.bandit_wealth)/NBANDITS
    ind.fitness = 0
    for b in 1:NBANDITS
        ind.fitness += abs(ind.bandit_wealth[b] - avg)/NBANDITS
    end
    ind.fitness = 1/(1.0 + 50*ind.fitness)
end

function mutate!(ind::Individial)
    #swam randomly items of two bandits
    nitems = length(ind.item_owner)
    if rand() < SWAP_PROB
        i1 = rand(1:nitems)
        i2 = rand(1:nitems)
        b = ind.item_owner[i1]
        ind.item_owner[i1] = ind.item_owner[i2]
        ind.item_owner[i2] = b
    end
    #take a random item from a rich bandit and give it to a poor
    if rand() < BALANCE_PROB
        b_poor = argmin(ind.bandit_wealth)
        b_rich = argmax(ind.bandit_wealth)
        i0 = rand(1:nitems)
        for i in 1:nitems
            j = 1 + (i + i0)%nitems
            if ind.item_owner[j] == b_rich
                ind.item_owner[j] = b_poor
                break
            end
        end
    end
end

function breed!(parents::Population, childs::Population)
    tot_fitness = sum(ind -> ind.fitness, parents)
    npairs = div(POP_SIZE, 2)
    Threads.@threads for n in 1:npairs
        parent1 = roulette_wheel(parents, rand()*tot_fitness)
        parent2 = roulette_wheel(parents, rand()*tot_fitness)
        child1 = childs[n]
        child2 = childs[n + npairs]
        cross!(parent1, parent2, child1, child2)
    end
end

function cross!(
        parent1::Individial, parent2::Individial,
        child1::Individial, child2::Individial
        )
    child1.item_owner .= parent1.item_owner
    child2.item_owner .= parent2.item_owner
    #recombination
    if rand() < CROSS_PROB
        nitems = length(parent1.item_owner)
        b1_rich = argmax(parent1.bandit_wealth)
        b2_rich = argmax(parent2.bandit_wealth)
        i0 = rand(1:nitems)
        for i in 1:nitems
            j = 1 + (i + i0)%nitems
            if parent1.item_owner[j] == b1_rich
                child1.item_owner[j] = parent2.item_owner[j]
                b1_rich = parent2.item_owner[j]
            end
            if parent2.item_owner[j] == b2_rich
                child2.item_owner[j] = parent1.item_owner[j]
                b2_rich = parent1.item_owner[j]
            end
        end 
    end
end

function roulette_wheel(pop::Population, x::Float64)::Individial
    n = 0
    while x > 0.0 && n+1 < length(pop)
        n += 1
        x -= pop[n].fitness
    end
    return pop[n]
end

function read_items(path::String)::Vector{Int64}
    item_price = Vector{Int64}()
    for line in readlines(path)
        push!(item_price, parse(Int64, line))
    end
    return item_price
end

function saveind(ind::Individial, item_price::Vector{Int64}, path::String)
    open(path, "w") do file
        for i in 1:length(ind.item_owner)
            write(file, string(item_price[i])*" "*string(ind.item_owner[i])*"\n")
        end
    end
end

function eva(item_price::Vector{Int64})
    log_best = Vector{Int64}()
    log_avg = Vector{Float64}()
    log_q25 = Vector{Float64}()
    log_q75 = Vector{Float64}()
    nitems = length(item_price)
    println("number of items: ", nitems)
    #random initial population
    pop = [Individial(nitems) for k in 1:POP_SIZE]
    #buffer for next population
    childs = [Individial(nitems) for k in 1:POP_SIZE]
    #eval them
    for k in 1:POP_SIZE
        eval!(pop[k], item_price)
    end
    #then iterate:
    for gen in 1:MAX_GEN
        #generate offspring
        breed!(pop, childs)
        #mutate and calculate fitness
        Threads.@threads for ind in childs
            mutate!(ind)
            eval!(ind, item_price)
        end
        #kill the parents, except elites
        Threads.@threads for n in NELITES+1:POP_SIZE
            temp = pop[n]
            pop[n] = childs[n]
            childs[n] = temp
        end
        #order by fitness
        sort!(pop, by = (ind -> -ind.fitness))
        #save results
        if (gen % GENS_PER_FRAME == 0)
            println(pop[1].obj_val, '\t', pop[1].fitness)
        end
        push!(log_best, pop[1].obj_val)
        push!(log_avg , sum(ind -> ind.obj_val/length(pop), pop))
        push!(log_q25, quantile!([p.obj_val for p in pop], 0.25))
        push!(log_q75, quantile!([p.obj_val for p in pop], 0.75))
    end
    return (pop[1], log_best, log_avg, log_q25, log_q75)
end

function main()
    item_price = read_items("evaTeaching-python/inputs/partition-easy.txt")
    log_avg = zeros(MAX_GEN)
    log_best = zeros(MAX_GEN)
    log_q25 = zeros(MAX_GEN)
    log_q75 = zeros(MAX_GEN)
    conv_prob = 0.0;
    avg_conv_time = 0.0;
    for k in 1:NEXPERIMENTS
        println("experiment #", k)
        @time (result, _log_best, _log_avg, _log_q25, _log_q75) = eva(item_price)
        log_avg += _log_avg/NEXPERIMENTS
        log_best += _log_best/NEXPERIMENTS
        log_q25 += _log_q25/NEXPERIMENTS
        log_q75 += _log_q75/NEXPERIMENTS
        for k in 1:MAX_GEN
            if _log_best[k] == 0
                conv_prob += 1/NEXPERIMENTS
                avg_conv_time += k/NEXPERIMENTS
                break
            end
        end 
        #uncomment this if you want to save the best solution
        #saveind(result, item_price, "julia-partition-results/"*EXPERIMENT_NAME*"-"*string(k)*".best")
    end
    avg_conv_time = (conv_prob == 0 ? NaN : avg_conv_time/conv_prob)
    println("convergence probability = ", conv_prob)
    println("avg conv time = ", avg_conv_time) 
    p = plot(legend = :topright, xlabel = "fitness evals", ylabel = "objective value", ylims = (0.0, 10000.0))
    x = POP_SIZE*(1:MAX_GEN)
    plot!(x, log_avg, label = "average")
    plot!(x, log_best, label = "best")
    plot!(x, log_q25, fillrange = log_q75, alpha = 0.3, label = "first-third quantile")
    savefig(p, "julia-partition-results/"*EXPERIMENT_NAME*".png")
end

function verify_solution(path::String)
    bandit_wealth = zeros(Int64, NBANDITS)
    file = readlines(path)
    for row in file
        line = split(row)
        p = parse(Int64, line[1])
        b = parse(Int64, line[2])
        bandit_wealth[b] += p
    end
    for b in 1:NBANDITS
        print(bandit_wealth[b], " | ")
    end
    println()
end

end