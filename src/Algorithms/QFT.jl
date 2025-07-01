module QFT
using Heterotic.QSim

export  qft, iqft, dft_matrix, idft_matrix

"""
    qft(n::Int) -> Circuit
"""

"""
    dft_matrix(N::Int) -> Matrix{ComplexF64}

Generate the N×N Discrete Fourier Transform matrix:
  W[j,k] = exp(2π·im*(j-1)*(k-1)/N) / √N
"""
function dft_matrix(N::Int)
    ω = exp(2π * im / N)
    M = Matrix{ComplexF64}(undef, N, N)
    for j in 1:N, k in 1:N
        M[j, k] = ω^((j-1)*(k-1)) / sqrt(N)
    end
    return M
end

"""
    idft_matrix(N::Int) -> Matrix{ComplexF64}

Generate the inverse DFT matrix (iDFT):
  W⁻¹ = (W)†
"""
function idft_matrix(N::Int)
    conj.(transpose(dft_matrix(N)))
end
# Example usage:
# n_qubits = 3
# qft_circuit = qft(n_qubits)
# iqft_circuit = iqft(n_qubits)

# To verify, applying QFT and then iQFT should result in the identity.
# combined_circuit = qft_circuit + iqft_circuit 
# This should be equivalent to an empty circuit or identity operation.
end

"""
    qft(n::Int) -> Circuit
    """