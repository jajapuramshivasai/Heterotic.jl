module COBYLA

export COBYLAOptimizer, optimize

mutable struct COBYLAOptimizer
    maxeval::Int        # Maximum function evaluations
    rhobeg::Float64     # Initial trust region radius
    rhoend::Float64     # Final trust region radius
    verbose::Bool       # Print progress information
end

# Constructor with default values
function COBYLAOptimizer(; 
    maxeval = 1000, 
    rhobeg = 1.0, 
    rhoend = 1e-6, 
    verbose = false)
    
    return COBYLAOptimizer(maxeval, rhobeg, rhoend, verbose)
end

"""
    optimize(f, x0, constraints=[], optimizer=COBYLAOptimizer())

Minimize objective function `f` starting from `x0` subject to given constraints.
Each constraint function in `constraints` should return ≤ 0 for feasible points.

Returns tuple (best_point, best_value)
"""
function optimize(f, x0, constraints=[]; optimizer=COBYLAOptimizer())
    n = length(x0)
    m = length(constraints)
    
    # Number of interpolation points
    npt = n + 1
    
    # Initialize
    x = copy(x0)
    rho = optimizer.rhobeg
    nfeval = 0
    
    # Create initial simplex
    sim = zeros(npt, n)
    sim[1, :] = x
    for i in 1:n
        sim[i+1, :] = x
        sim[i+1, i] += rho
    end
    
    # Function and constraint values
    f_values = zeros(npt)
    c_values = zeros(m, npt)
    
    # Evaluate function and constraints at initial points
    for i in 1:npt
        f_values[i] = f(sim[i, :])
        for j in 1:m
            c_values[j, i] = constraints[j](sim[i, :])
        end
        nfeval += 1
    end
    
    # Main iteration loop
    while nfeval < optimizer.maxeval && rho > optimizer.rhoend
        # Find best feasible point or least infeasible point
        best_idx = 1
        for i in 2:npt
            if all(c_values[:, i] .<= 0) && (!all(c_values[:, best_idx] .<= 0) || f_values[i] < f_values[best_idx])
                best_idx = i
            elseif !all(c_values[:, i] .<= 0) && !all(c_values[:, best_idx] .<= 0)
                max_violation_i = maximum(c_values[:, i])
                max_violation_best = maximum(c_values[:, best_idx])
                if max_violation_i < max_violation_best
                    best_idx = i
                end
            end
        end
        
        # Find worst point
        worst_idx = argmax([f_values[i] + 100*sum(max.(0, c_values[:, i])) for i in 1:npt])
        
        # Calculate centroid excluding worst point
        centroid = zeros(n)
        for i in 1:npt
            if i != worst_idx
                centroid .+= sim[i, :]
            end
        end
        centroid ./= (npt - 1)
        
        # Reflect worst point through centroid
        x_new = 2 * centroid - sim[worst_idx, :]
        
        # Evaluate at new point
        f_new = f(x_new)
        c_new = [con(x_new) for con in constraints]
        nfeval += 1
        
        # Check if new point is better than worst point
        new_metric = f_new + 100*sum(max.(0, c_new))
        worst_metric = f_values[worst_idx] + 100*sum(max.(0, c_values[:, worst_idx]))
        
        if new_metric < worst_metric
            # Replace worst point with new point
            sim[worst_idx, :] = x_new
            f_values[worst_idx] = f_new
            for j in 1:m
                c_values[j, worst_idx] = c_new[j]
            end
        else
            # Contract simplex toward best point
            for i in 1:npt
                if i != best_idx
                    sim[i, :] = 0.5 * (sim[i, :] + sim[best_idx, :])
                    f_values[i] = f(sim[i, :])
                    for j in 1:m
                        c_values[j, i] = constraints[j](sim[i, :])
                    end
                    nfeval += 1
                end
            end
        end
        
        # Check convergence
        max_dist = maximum([norm(sim[i, :] - sim[best_idx, :]) for i in 1:npt if i != best_idx])
        if max_dist < rho
            rho *= 0.5
            if optimizer.verbose
                println("Reducing trust region radius to $rho")
            end
        end
        
        if optimizer.verbose && (nfeval % 20 == 0)
            println("Iter: $nfeval, Best value: $(f_values[best_idx]), rho: $rho")
        end
    end
    
    best_idx = argmin(f_values[findall(all(c_values .<= 0, dims=1)[:])])
    
    return sim[best_idx, :], f_values[best_idx]
end

end # module