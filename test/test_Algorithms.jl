
using Test
using Heterotic.QSim
using Heterotic.Grovers
using Heterotic.QFT

@testset "QFT & iQFT" begin
    @testset "dft_matrix ↔ idft_matrix" begin
        N = 8
        W = dft_matrix(N)
        W⁻¹ = idft_matrix(N)
        I = id(3)
        @test isapprox(W * W⁻¹, I; atol=1e-10)
        @test isapprox(W⁻¹ * W, I; atol=1e-10)
    end
end

@testset "Grover primitives" begin
    @testset "full algorithm demo" begin
        function oracle_custom(s)
            x!(s, 2)  # marks 10 state
            h!(s, 2)  
            cnot!(s, 1, 2)
            h!(s, 2)
            x!(s, 2)
            return s
        end
        s = statevector(2,0)
        h!(s, 1)  # Apply Hadamard to create uniform superposition
        h!(s, 2)  # Apply Hadamard to create uniform superposition
        for i in 1:1
            s = oracle_custom(s)  # Apply the oracle
            s = diffusion(s)  # Apply the diffusion operator
            # println(s) #Flag

        end

        # Find and print the index with maximum probability
        max_prob_index = argmax(abs2.(s))
        println("Index with maximum absolute squared value: $max_prob_index")

        s = statevector(2, 0)  # Reset state
        h!(s, 1)  # Apply Hadamard to create uniform superposition
        h!(s, 2)  # Apply Hadamard to create uniform superposition
        winner = 2
        s2 = oracle(copy(s), winner)
        @test isapprox(s2, ComplexF64[1/2, 1/2, -1/2, 1/2]; atol=1e-4)
    
    end

end