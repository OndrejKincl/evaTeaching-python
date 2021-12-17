module multi
include("moo_functions.jl")
include("plot_utils.jl")
using Plots

const POP_SIZE = 100
const DIM = 10
const MAX_GEN = 50
const CROSS_PROB = 0.2
const MUT_PROB = 0.8
const MUT_STEP = 0.05
const REPEATS = 10
const NELITES = 0
const HYP_REF = (11.,11.)

const EXPERIMENT_NAME = "multi"
const COMPARE_TO = "none"
const GENS_PER_FRAME = 5
const LOG_SIZE = div(MAX_GEN, GENS_PER_FRAME)

mutable struct Individual
    x::Vector{Float64}
    f::Tuple{Float64, Float64}
    front::Int64
    ssc::Float64 #secondary sorting criterion
    Individual() = begin
        x = rand(DIM)
        f = (0,0)
        front = 0
        ssc = 0.0
        return new(x,f,front,ssc)
    end
end

Base.isless(ind1::Individual, ind2::Individual) = begin
    return (ind1.front, ind1.f) < (ind2.front, ind2.f)
end

const Population = Vector{Individual}

function isdominated(ind1::Individual, ind2::Individual)::Bool
    return ind1.f[1] >= ind2.f[1] && ind1.f[2] >= ind2.f[2] && ind1.f != ind2.f
end

function isdominated(ind::Individual, pop::Population; first = 1)::Bool
    for k in first:POP_SIZE
        if isdominated(ind, pop[k]) 
            return true
        end
    end
    return false
end

#find fronts and reorder pop accordingly
function findfronts!(pop::Population)
    for k in 1:POP_SIZE
        pop[k].front = 0
    end
    front = 0
    removed = 0
    while removed < POP_SIZE
        sorted = 0
        for l in (removed+1):POP_SIZE
            if !isdominated(pop[l], pop, first = (removed+1))
                pop[l].front = front
                sorted += 1
                #swap
                k = removed + sorted
                temp = pop[k] 
                pop[k] = pop[l]
                pop[l] = temp
            end
        end
        front += 1
        removed += sorted
    end
end

#find ssc using crowding dist
function findssc!(pop::Population)
    for k in 1:POP_SIZE
        ind = pop[k]
        if k == 1 || k == POP_SIZE
            ind.ssc = Inf
            continue
        end
        ind0 = pop[k-1]
        ind1 = pop[k+1]
        if ind0.front < ind.front || ind.front < ind1.front
            ind.ssc = Inf
            continue
        end
        ind.ssc = ind1.f[1] - ind0.f[1] + ind0.f[2] - ind1.f[2]
    end
end

function hypervolume(pop::Population)::Float64
    vol = (HYP_REF[1] - pop[1].f[1])*(HYP_REF[2] - pop[1].f[2])
    for k in 2:POP_SIZE
        if pop[k].front == 0
            break
        end
        vol += (HYP_REF[1] - pop[k].f[1])*(pop[k-1].f[2] - pop[k].f[2])
    end
    return vol        
end

function eval!(ind::Individual, fun::Function)
    ind.f = fun(ind.x)
end

function breed!(parents::Population, childs::Population)
    npairs = div(POP_SIZE, 2)
    Threads.@threads for n in 1:npairs
        parent1 = tournament_selection(parents)
        parent2 = tournament_selection(parents)
        child1 = childs[n]
        child2 = childs[n + npairs]
        copy!(child1, parent1)
        copy!(child2, parent2)
        cross!(parent1, parent2, child1, child2)
    end
end

function tournament_selection(pop::Population)::Individual
    ind1 = rand(pop)
    ind2 = rand(pop)
    if ind1.front < ind2.front
        return ind1
    elseif ind1.front == ind2.front && ind1.ssc > ind2.ssc
        return ind1
    end
    return ind2
end

function copy!(dest::Individual, src::Individual)
    dest.x .= src.x
    #dest.mut_var = src.mut_var
end

function cross!(
        parent1::Individual, parent2::Individual,
        child1::Individual, child2::Individual
        )
    if rand() < CROSS_PROB
        cross_point = rand(1:(DIM-1))
        for i in 1:cross_point
            child1.x[i] = parent2.x[i]
            child2.x[i] = parent1.x[i]
        end
    end
end

function mutate!(ind::Individual)
    if rand() < MUT_PROB
        for k in 1:DIM
            ind.x[k] += MUT_STEP*randn()
            ind.x[k] = clamp(ind.x[k], 0., 1.)
        end
    end
end

function eva(fun::Function, opt_hypvol::Float64)
    pop = [Individual() for _ in 1:POP_SIZE]
    buffer = [Individual() for _ in 1:POP_SIZE]
    log_hypvol = Float64[]
    for k in 1:POP_SIZE
        eval!(pop[k], fun)
        eval!(buffer[k], fun)
    end
    findfronts!(pop)
    sort!(pop)
    findssc!(pop)
    #then iterate:
    for gen in 1:MAX_GEN
        #generate offspring
        breed!(pop, buffer)
        #kill the parents, except elites
        Threads.@threads for n in NELITES+1:POP_SIZE
            temp = pop[n]
            pop[n] = buffer[n]
            buffer[n] = temp
        end
        #mutate and calculate fitness
        Threads.@threads for ind in buffer
            mutate!(ind)
            eval!(ind, fun)
        end
        findfronts!(pop)
        sort!(pop)
        findssc!(pop)
        #save results
        if (gen % GENS_PER_FRAME == 0)
            @show(gen)
            obj = opt_hypvol - hypervolume(pop) 
            @show(obj)
            push!(log_hypvol, obj)
        end
    end
    #extract nondom front
    po_front = filter(ind -> ind.front == 0, pop)
    return (po_front, log_hypvol)
end

function main()
    funs = moo_functions.zdt()
    opt_hypvols = moo_functions.opt_hvs()
    for n in [1]
        println()
        println("function #", n)
        println("============================================")
        log = zeros(REPEATS, LOG_SIZE)
        for experiment in 1:REPEATS
            println("experiment #", experiment)
            (result, log_hypvol) = eva(funs[n], opt_hypvols[n])
            for gen in 1:LOG_SIZE
                log[experiment, gen] = log_hypvol[gen]
            end
            #save result as a plot
            if experiment == 1
                f1 = [ind.f[1] for ind in result]
                f2 = [ind.f[2] for ind in result]
                fig = scatter(f1, f2, xlabel = "f1", ylabel = "f2")
                savefig(fig, "julia-multi-results/"*EXPERIMENT_NAME*string(n)*"_graph.pdf")
            end
        end
        t = LinRange(0, MAX_GEN*POP_SIZE, LOG_SIZE)
        new_path = "julia-multi-results/"*EXPERIMENT_NAME*string(n)*".log"
        old_path = "julia-multi-results/"*COMPARE_TO*string(n)*".log"        
        fig_path = "julia-multi-results/"*EXPERIMENT_NAME*string(n)*".pdf" 
        plot_utils.save_log(t, log, new_path)
        if ispath(old_path)
            p = plot_utils.plot_log(old_path, new_path, symbols = ["old","new"])
            savefig(p, fig_path)
        else
            p = plot_utils.plot_log(new_path)
            savefig(p, fig_path)
        end      
    end
end

end

