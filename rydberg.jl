using LinearAlgebra
using SparseArrays
using Arpack
using Dates

function σ(N::Int)
    d = 2^N
    σ_all = [SparseMatrixCSC{Float64, Int}[] for i in 1:3]

    n = spdiagm([0,1])
    x = sparse([0 1; 1 0])
    z = spdiagm([1,-1])

    σ0 = [n,x,z]
    Id = spdiagm([1,1])

    for i in 1:N
        for k in 1:3
            σk = spdiagm([1])
            for j in 1:i-1
                σk = kron(σk, Id)
            end
            σk = kron(σk, σ0[k])
            for j in i+1:N
                σk = kron(σk, Id)
            end
            push!(σ_all[k], σk)
        end
    end
    return σ_all
end

function ham(N::Int, σ_all::Vector{Vector{SparseMatrixCSC{Float64, Int}}},
        Ω::Float64, Δ::Float64)
    d = 2^N
    H = spzeros(Float64, d, d)
    for i in 1:N
        H += -Δ * σ_all[1][i] + Ω/2 * σ_all[2][i]
        for r in 1:3
            if i+r <= N
                H += 1/r^6 * σ_all[1][i] * σ_all[1][i+r]
            end
        end
    end
    return H 
end

function main(N::Int)
    d = 2^N
    σ_all = σ(N)

    ψ0 = zeros(d)
    ψ0[1] = 1
    
    n_all = zeros(N)
    for j in 1:N
        n_all[j] = real(ψ0'*σ_all[1][j]*ψ0)
    end
    println(n_all)
    
    h = 20.0
    H1 = ham(N, σ_all, h, h)
    E1, U1 = eigen(Hermitian(Matrix(H1)))
   
    t = π / h / √2
    ψ0 = U1 * diagm(exp.(-1im*E1*t)) * U1' * ψ0
    x_all = zeros(N)
    for j in 1:N
        x_all[j] = real(ψ0'*σ_all[2][j]*ψ0)
    end
    println(x_all)

    H2 = ham(N, σ_all, 0.0, 0.0)
    E2, U2 = eigen(Hermitian(Matrix(H2)))
    
    t = π
    ψ0 = U2 * diagm(exp.(-1im*E2*t)) * U2' * ψ0

    println(ψ0' * σ_all[2][1] * σ_all[3][2] * ψ0)
    println(ψ0' * σ_all[3][1] * σ_all[2][2] * ψ0)

    # measure Z
    z_meas = zeros(N)
    z_meas[1] = real(ψ0' * σ_all[3][1] * σ_all[3][2] * ψ0)
    z_meas[N] = real(ψ0' * σ_all[3][N-1] * σ_all[3][N] * ψ0)
    for i in 2:N-1
        z_meas[i] = real(ψ0' * σ_all[3][i-1] * σ_all[3][i] * σ_all[3][i+1] * ψ0)
    end

    # measure X
    x_meas = zeros(N+1)
    h = 20.0
    H3 = ham(N, σ_all, h, h)
    E3, U3 = eigen(Hermitian(Matrix(H3)))
    t = π / h / √2
    ψ = U3 * diagm(exp.(-1im*E3*t)) * U3' * ψ0

    ψ1 = σ_all[3][1] * ψ
    for i in 3:2:N
        ψ1 = σ_all[3][i] * ψ1
    end
    x_meas[N+1] = real(ψ'ψ1)
    
    x_meas[1] = real(ψ' * σ_all[3][1] * σ_all[3][2] * ψ)
    x_meas[N] = real(ψ' * σ_all[3][N-1] * σ_all[3][N] * ψ)
    for i in 2:N-1
        x_meas[i] = real(ψ' * σ_all[3][i-1] * σ_all[3][i] * σ_all[3][i+1] * ψ)
    end

    # measure X+Z
    xzp_meas = zeros(N)
    h = 20.0
    
    H4 = ham(N, σ_all, 0.0, h)
    E4, U4 = eigen(Hermitian(Matrix(H4)))
    t = π / h / 2
    ψ = U4 * diagm(exp.(-1im*E4*t)) * U4' * ψ0
    
    H5 = ham(N, σ_all, h, 0.0)
    E5, U5 = eigen(Hermitian(Matrix(H5)))
    t = π / h / 4
    ψ = U5 * diagm(exp.(-1im*E5*t)) * U5' * ψ

    xzp_meas[1] = real(ψ' * σ_all[3][1] * σ_all[3][2] * ψ)
    xzp_meas[N] = real(ψ' * σ_all[3][N-1] * σ_all[3][N] * ψ)
    for i in 2:N-1
        xzp_meas[i] = real(ψ' * σ_all[3][i-1] * σ_all[3][i] * σ_all[3][i+1] * ψ)
    end

    # measure X-Z
    xzm_meas = zeros(N)
    h = 20.0
    
    H6 = ham(N, σ_all, 0.0, -h)
    E6, U6 = eigen(Hermitian(Matrix(H6)))
    t = π / h / 2
    ψ = U6 * diagm(exp.(-1im*E6*t)) * U6' * ψ0
    
    H7 = ham(N, σ_all, h, 0.0)
    E7, U7 = eigen(Hermitian(Matrix(H7)))
    t = π / h / 4
    ψ = U7 * diagm(exp.(-1im*E7*t)) * U7' * ψ

    xzm_meas[1] = real(ψ' * σ_all[3][1] * σ_all[3][2] * ψ)
    xzm_meas[N] = real(ψ' * σ_all[3][N-1] * σ_all[3][N] * ψ)
    for i in 2:N-1
        xzm_meas[i] = real(ψ' * σ_all[3][i-1] * σ_all[3][i] * σ_all[3][i+1] * ψ)
    end

    return z_meas, x_meas, xzp_meas, xzm_meas
end

N = 9
z_meas, x_meas, xzp_meas, xzm_meas = main(N)

open("./data/ryd_z_meas_L_$(N).dat", "w") do io
    write(io, z_meas)
end
open("./data/ryd_x_meas_L_$(N).dat", "w") do io
    write(io, x_meas)
end
open("./data/ryd_xzp_meas_L_$(N).dat", "w") do io
    write(io, xzp_meas)
end
open("./data/ryd_xzm_meas_L_$(N).dat", "w") do io
    write(io, xzm_meas)
end
