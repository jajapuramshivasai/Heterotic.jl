module Grovers
using Heterotic.QSim
using LinearAlgebra

export grover ,oracle, diffusion

"""
    statevector(n::Int, init::Int)
"""

"""
    oracle(s::Vector{ComplexF64}, winner::Int)

Applies the oracle to the state vector `s`.
The oracle flips the phase of the `winner` state.
"""
function oracle(s::Vector{ComplexF64}, winner::Int)
    n = nb(s)
    if !(0 <= winner < 1 << n)
        error("Winner state is out of bounds for the number of qubits.")
    end
    
    # The oracle U_w is a diagonal matrix with -1 at the winner index
    # and 1 everywhere else. Applying it is equivalent to flipping the
    # phase of the corresponding amplitude in the state vector.
    s[winner + 1] *= -1
    return s
end

"""
    diffusion(s::Vector{ComplexF64})

Applies the Grover diffusion operator (inversion about the mean) to the state `s`.
"""
function diffusion(s::Vector{ComplexF64})
    n = nb(s)

    # Apply H to all qubits
    for i in 1:n
        h!(s, i)
    end

    # Apply reflection about |0...0> state (U_s = 2|0><0| - I)
    # This can be implemented as a multi-controlled Z gate with X-gates
    # on all qubits before and after.
    
    # Apply X to all qubits
    for i in 1:n
        x!(s, i)
    end

    # Multi-controlled Z gate (on all qubits, targetting the last one)
    # A multi-controlled Z gate flips the phase of the |1...1> state.
    # We can implement this with a controlled-Z gate on the last qubit,
    # controlled by all other qubits.
    # For simplicity in this simulator, we can directly flip the phase of the |1...1> state
    # which is the last element of the state vector after the X gates.
    if n > 1
        # A simple way to implement multi-controlled Z is to use a single
        # controlled Z gate recursively, but for a statevector simulation,
        # a direct phase flip is more efficient.
        # Here we use a controlled Z from qubit n-1 to n, with controls on all others.
        # For a full implementation, one would decompose the multi-controlled gate.
        # A simpler approach that is equivalent to 2|s_uniform><s_uniform| - I
        # is to reflect around the |0...0> state.
        
        # The operator is I - 2|0...0><0...0|.
        # It flips the phase of every state except |0...0>.
        # After the X-gates, the |0...0> state is now the |1...1> state.
        # The operator becomes a reflection around |1...1>.
        # Let's implement the reflection around |0...0> directly.
        
        # Store the amplitude of the |0...0> state
        s0 = s[1]
        # Flip the phase of all states
        s .*= -1
        # Add back 2 * s0 to the |0...0> state's amplitude
        s[1] += 2 * s0
    else # For a single qubit
        z!(s, 1)
    end


    # Apply X to all qubits again
    for i in 1:n
        x!(s, i)
    end

    # Apply H to all qubits again
    for i in 1:n
        h!(s, i)
    end
    
    return s
end


"""
    grover(n::Int, winner::Int)

Performs Grover's search algorithm for `n` qubits to find the `winner` state.

# Arguments
- `n::Int`: The number of qubits.
- `winner::Int`: The integer representation of the state to find (from 0 to 2^n - 1).

# Returns
- `Vector{ComplexF64}`: The final state vector after the search.
"""
function grover(n::Int, winner::Int)
    # 1. Initialization
    s = statevector(n, 0) # Start in |0...0>

    # Apply Hadamard to all qubits to create superposition
    for i in 1:n
        h!(s, i)
    end

    # 2. Iterations
    # Optimal number of iterations
    k = round(Int, (π / 4) * sqrt((1 << n)))
    
    println("Running Grover's algorithm for $n qubits to find state |$winner⟩.")
    println("Optimal number of iterations: $k")

    for i in 1:k
        # Apply Oracle
        oracle(s, winner)
        
        # Apply Diffusion operator
        diffusion(s)
    end

    # 3. Measurement (implied, we return the final state)
    println("\nFinal state probabilities:")
    prstate(s)
    
    return s
end

end # module Grovers

# Example Usage:
# using .Grovers
# n_qubits = 3
# winning_state = 5 # We want to find the state |101>
# final_state = Grovers.grover(n_qubits, winning_state)