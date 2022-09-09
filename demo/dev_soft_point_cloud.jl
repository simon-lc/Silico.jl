using LinearAlgebra
using BenchmarkTools

include("../cvxnet/loss.jl")
# include("../cvxnet/point_cloud.jl")
include("../cvxnet/softmax.jl")


function soft_intersection(e, v, A, b, δ)
    n = length(b)
    α = zeros(n)
    d = zeros(n)
    noskip = trues(n)

    for i = 1:n
        denum = (A[i,:]' * v)
        noskip[i] = abs(denum) > 1e-3
        α[i] = (b[i] - A[i,:]' * e) / denum
        noskip[i] = noskip[i] && (α[i] >= 0)
        d[i] = maximum(A * (e + α[i] * v) - b)
    end
    # @show α
    # @show d
    αsoft = softmax_mean(α[noskip], (- 1α)[noskip], δ)
    dsoft = softmax_mean(d[noskip], (- 1α)[noskip], δ)
    # @show αsoft
    return αsoft, dsoft
end



Xlims = [-2, 2]
Ylims = [-2, 2]
S = 100

X = range(Xlims..., length=S)
Y = range(Ylims..., length=S)
V = zeros(S,S)

A0 = [
    +1.0 -0.4;
    +0.0 +1.0;
    -1.0 -0.4;
    +0.0 -1.0;
    ]
for i = 1:4
    A0[i,:] ./= norm(A0[i,:])
end
b0 = 0.5*[
    +1.5,
    +1,
    +1.5,
    +0,
    ];
o0 = [0, 0.0]

Af = [0 1.0]
bf = [0.0]
of = [0, 0.0]

for i = 1:S
    for j = 1:S
        p = [X[i], Y[j]]
        V[j,i] = maximum(A0 * p - b0)
    end
end

plt = heatmap(
    X, Y, V,
    aspectratio=1.0,
    xlims=Xlims,
    ylims=Ylims,
    legend=false,
    xlabel="x", ylabel="y",
    )
plt = plot_polytope(A0, b0, 1e2, xlims=Xlims, ylims=Ylims)

e = [0.0, 2.0]
v = [0.4, -1.0]
δ = 1e+1
α, d = soft_intersection(e, v, A0, b0, δ)

for θ in range(-0.1π, -0.9π, length=100)
    v = [cos(θ), sin(θ)]
    α0, d0 = soft_intersection(e, v, A0, b0, δ)
    αf, df = soft_intersection(e, v, Af, bf, δ)
    pc0 = e + α0 * v
    pcf = e + αf * v
    α = softmax_mean([α0, αf], [-α0 - 100d0, -αf - 100df], δ)
    pc = e + α * v
    scatter!(plt, pc0[1:1], pc0[2:2], color=:black, legend=false)
    scatter!(plt, pcf[1:1], pcf[2:2], color=:green, legend=false)
    scatter!(plt, pc[1:1],  pc[2:2], color=:yellow, legend=false, markersize=8)

end
display(plt)

range(-0.2π, -0.8π, step=20)


# function mysolve!(A, b, e, v)
#     α = 0.0
#     x = zeros(2)
#
#     ρ + v' * (v * α + e - x)
#     (x - v * α - e) + A' * γ
