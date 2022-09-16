using Plots

include("../halfspace.jl")
include("../softmax.jl")
include("../visuals.jl")


################################################################################
# polytope params
################################################################################
Ap0 = [
    -sqrt(2)/2 sqrt(2)/2;
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.00ones(5,2)
bp0 = 0.4*[
    +1,
    +1,
    +1,
    +1,
     1,
    ]
op0 = [0.0, +0.0]

Ap1 = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.20ones(4,2)
bp1 = 0.30*[
    +1,
    +1,
    +1,
     1,
    ]
op1 = [0.25, 0.25]

Ap = [Ap0, Ap1,]
bp = [bp0, bp1,]
op = [op0, op1,]
θ_p, polytope_dimensions_p = pack_halfspaces(Ap, bp, op)



################################################################################
# figure params
################################################################################
figsize = 300
S = 200
lims = (-1.0, +1.0)
δ = 100.0

################################################################################
# figures
################################################################################
plt = plot_polytope(Ap0, bp0 + Ap0 * op0, δ, size=(figsize, figsize), S=S,
    plot_halfspaces=false, plot_heatmap=false, countour_color=:black)
plt = plot_polytope(Ap1, bp1 + Ap1 * op1, δ, size=(figsize, figsize), S=S,
    plot_halfspaces=false, plot_heatmap=false, countour_color=:black, plt=plt)
savefig(plt, joinpath("/home/simon/Downloads", "plot0.png"))


plt = plot_polytope(Asol[1], bsol[1] + Asol[1] * osol[1], δ, size=(figsize, figsize), S=S,
    plot_halfspaces=false, plot_heatmap=true, countour_color=:black,)
Asol, bsol, osol
Asol[1]

figsize=500
δ_sigmoid = 0.05
plt = plot_heatmap(size=(figsize, figsize), S=S,
    plot_halfspaces=false, countour_color=:white,
    xlims=(-1,1), ylims=(-0.5,1.5),
    # f=x -> 1 * sum((0 .- sigmoid.(-1/δ_sigmoid .* sdfV(x, θsol0, polytope_dimensions, δ))).^2),
    f=x -> sigmoid(-1/δ_sigmoid * sdfV(x, θsol0, polytope_dimensions, δ))^2,
    )


plt = plot_heatmap(size=(figsize, figsize), S=S,
    plot_halfspaces=false, countour_color=:white,
    xlims=(-1,1), ylims=(-0.5,1.5),
    # f=x -> 1 * sum((0 .- sigmoid.(-1/δ_sigmoid .* sdfV(x, θsol0, polytope_dimensions, δ))).^2),
    # f=x -> sigmoid(-1/δ_sigmoid * vectorized_sdf(x, θsol0, polytope_dimensions, δ))^2,
    f=f,
    )

δ_sigmoid = 0.01
Asol, bsol, osol = unpack_halfspaces(θinit, polytope_dimensions)
bosol = [bi + Asol[i] * osol[i] for (i,bi) in enumerate(bsol)]
f = x -> vectorized_sdf([1.0, ], [1; 0;;], [x[1] - 1; x[2];;], Asol, bosol, 0.05)[1]
f = x -> sigmoid.(-1/δ_sigmoid .* vectorized_sdf([1.0, ], [1; 0;;], [x[1] - 1; x[2];;], Asol, bosol, 15)[1])
f([1,2])

[0.0,
[1; 0;;]

# ϕ = vectorized_sdf(α .+ sample, v, e, A, b, δ_sdf)
#
# l += 5 * sum((0 .- sigmoid.(-1/δ_sigmoid .* ϕ)).^2)



plt = plot_polytope(Ap0, bp0 + Ap0 * op0, δ, size=(figsize, figsize), S=S,
    plot_halfspaces=false, countour_color=:white)
savefig(plt, joinpath("/home/simon/Downloads", "plot1.png"))

plt = plot_polytope(Ap1, bp1 + Ap1 * op1, δ, size=(figsize, figsize), S=S,
    plot_halfspaces=false, countour_color=:white)
savefig(plt, joinpath("/home/simon/Downloads", "plot2.png"))



plt = plot_heatmap(size=(figsize, figsize), S=S,
    plot_halfspaces=false, countour_color=:white,
    f=x->sdfV(x, θ_p, polytope_dimensions_p, δ),
    )
savefig(plt, joinpath("/home/simon/Downloads", "plot3.png"))

plt = plot_heatmap(size=(figsize, figsize), S=S,
    plot_halfspaces=false, countour_color=:white, levels=[0.5],
    f=x->sigmoid(-30 * sdfV(x, θ_p, polytope_dimensions_p, δ)),
    )
savefig(plt, joinpath("/home/simon/Downloads", "plot4.png"))


function local_f(x)
    v = 0.0
    np = length(polytope_dimensions_p)
    for i = 1:np
        v += sigmoid(-100 * sdfV(x, θ_p, polytope_dimensions_p, i, δ))
    end
    return v
end

plt = plot_heatmap(size=(figsize, figsize), S=S,
    plot_halfspaces=false, countour_color=:blue, levels=[0.99, 1.99],
    f=local_f,
    )
savefig(plt, joinpath("/home/simon/Downloads", "plot5.png"))
