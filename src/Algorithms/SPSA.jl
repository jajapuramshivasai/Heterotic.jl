module SPSA

export spsa_gradient

"""
    spsa_gradient(params, cost_function; perturbation_size=0.01, num_iterations=1)

Estimate the gradient of `cost_function` at `params` using the Simultaneous Perturbation
Stochastic Approximation (SPSA) method.

# Arguments
- `params`: Vector of parameters
- `cost_function`: Function that takes a parameter vector and returns a scalar cost
- `perturbation_size`: Size of the perturbation for gradient estimation
- `num_iterations`: Number of gradient estimates to average

# Returns
- Estimated gradient vector
"""
function spsa_gradient(params, cost_function; perturbation_size=0.01, num_iterations=1)
    n = length(params)
    gradient = zeros(n)
    
    for _ in 1:num_iterations
        # Generate random perturbation vector (Bernoulli ±1)
        delta = 2.0 * (rand(n) .> 0.5) .- 1.0
        
        # Evaluate cost function at two nearby points
        f_plus = cost_function(params + perturbation_size * delta)
        f_minus = cost_function(params - perturbation_size * delta)
        
        # Compute gradient approximation
        gradient_estimate = (f_plus - f_minus) / (2 * perturbation_size) .* (1 ./ delta)
        
        # Accumulate gradient estimates
        gradient += gradient_estimate
    end
    
    # Average if doing multiple iterations
    return gradient / num_iterations
end

end # module