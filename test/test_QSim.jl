using Test
include("../src/QSim/QSim.jl")
using .QSim

# Test statevector creation
@testset "Statevector Tests" begin
    sv = statevector(2, 0)
    @test length(sv) == 4
    @test sv[1] == 1 + 0im
    @test all(sv[2:end] .== 0 + 0im)
end

# Test density matrix creation
@testset "Density Matrix Tests" begin
    dm = density_matrix(2, 0)
    @test size(dm) == (4, 4)
    @test dm[1, 1] == 1 + 0im
    @test all(dm[2:end, :] .== 0 + 0im)
    @test all(dm[:, 2:end] .== 0 + 0im)
end

# Test single-qubit gates
@testset "Single-Qubit Gate Tests" begin
    sv = statevector(1, 0)
    h!(sv, 1)
    @test isapprox(sv,[0.7071067811865475 + 0.0im, 0.7071067811865475 + 0im])

    x!(sv, 1)
    @test isapprox(abs2.(sv), [0.5, 0.5])
    @test isapprox(sv[1], 1 / √2 + 0im)
    @test isapprox(sv[2], 1 / √2 + 0im)

    z!(sv, 1)
    @test isapprox(sv[2], -1 / √2 + 0im)
end

# Test controlled gates
@testset "Controlled Gate Tests" begin
    sv = statevector(2, 0)
    cnot!(sv, 1, 2)
    @test isapprox(abs2.(sv), [1.0, 0.0, 0.0, 0.0])

    sv = statevector(2, 2)
    cnot!(sv, 1, 2)
    @test isapprox(abs2.(sv), [0.0, 0.0, 1.0, 0.0])
end

# Test measurement probabilities
@testset "Measurement Tests" begin
    sv = statevector(2, 0)
    p0, p1 = measure_z(sv, 1)
    @test isapprox(p0, 1.0)
    @test isapprox(p1, 0.0)

    h!(sv, 1)
    p0, p1 = measure_z(sv, 1)
    @test isapprox(p0, 0.5)
    @test isapprox(p1, 0.5)
end

# Test rotation gates
@testset "Rotation Gate Tests" begin
    sv = statevector(1, 0)
    rx!(sv, 1, π / 2)
    @test isapprox(abs2.(sv), [0.5, 0.5])

    # rz!(sv, 1, π / 2)
    # @test isapprox(abs2.(sv), [0.1464466094, 0.8535533906], atol=1e-6)

    # ry!(sv, 1, π / 4)
    # @test isapprox(abs2.(sv), [0.1464466094, 0.8535533906], atol=1e-6)
end

# Test swap gate
@testset "Swap Gate Tests" begin
    sv = statevector(2, 2)
    swap!(sv, 1, 2)
    @test isapprox(abs2.(sv), [0.0, 1.0, 0.0, 0.0])
end