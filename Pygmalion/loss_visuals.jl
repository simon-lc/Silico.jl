using Plots

include("halfspace.jl")
include("softmax.jl")
include("visuals.jl")


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
