include(joinpath(module_dir(), "Pygmalion/Pygmalion.jl"))

vis = Visualizer()
open(vis)
# render(vis)
set_background!(vis)
set_light!(vis, direction="Negative")
set_floor!(vis, color=RGBA(0.4,0.4,0.4,0.4))
iterate_color = RGBA(1,1,0,0.6);


################################################################################
# polytope parameters
################################################################################
Ap0 = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.00ones(4,2)
bp0 = 0.5*[
    +1,
    +1,
    +1,
     1,
    ]
op0 = [0.0, +0.5]

Ap1 = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.20ones(4,2)
bp1 = 0.25*[
    +1,
    +1,
    +1,
     1,
    ]
op1 = [0.5, 0.2]

Ap2 = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.20ones(4,2)
bp2 = 0.25*[
    +1,
    +1,
    +1,
     1,
    ]
op2 = [-0.5, 0.7]

Af = [0.0  1.0]
bf = [0.0]
of = [0.0, 0.0]

Ap = [Ap0, Ap1, Ap2, Af]
bp = [bp0, bp1, bp2, bf]
op = [op0, op1, op2, of]
θ_p, polytope_dimensions_p = pack_halfspaces(Ap, bp, op)

build_2d_polytope!(vis[:polytope], Ap0, bp0 + Ap0 * op0, name=:poly0, color=RGBA(1,1,1,1.0))
build_2d_polytope!(vis[:polytope], Ap1, bp1 + Ap1 * op1, name=:poly1, color=RGBA(1,1,1,1.0))
build_2d_polytope!(vis[:polytope], Ap2, bp2 + Ap2 * op2, name=:poly2, color=RGBA(1,1,1,1.0))


################################################################################
# camera parameters
################################################################################
ne = 10
nβ = 10
ρ0 = 1e-4
e0 = [[ex, 2.0] for ex in range(-1.70, 1.70, length=ne)]
β0 = [-π + atan(e[2], e[1]) .+ Vector(range(+0.25π, -0.25π, length=nβ)) for e in e0]

d0 = [trans_point_cloud(e0[i], β0[i], ρ0, θ_p, polytope_dimensions_p) for i = 1:ne]
for i = 1:ne
	c = i > 5 ? RGBA(0.1,0.1,0.1,1) : RGBA(0.9,0.1,0.1,1)
	build_point_cloud!(vis[:point_cloud], nβ; color=c, name=Symbol(i))
end
set_2d_point_cloud!(vis, e0, d0; name=:point_cloud)


################################################################################
# Initialization
################################################################################
nh = 8
polytope_dimensions = [nh,nh,nh,nh,nh,nh]
np = length(polytope_dimensions)

d_object = filter_point_cloud(d0; altitude_threshold=0.01)
θinit, kmres = parameter_initialization(d_object, polytope_dimensions)
Ainit, binit, oinit = unpack_halfspaces(deepcopy(θinit), polytope_dimensions)
visualize_kmeans!(vis, θinit, polytope_dimensions, d_object, kmres)
polytope_dimensions
setvisible!(vis[:cluster], true)
setvisible!(vis[:initial], true)

setvisible!(vis[:cluster], false)
setvisible!(vis[:initial], false)


################################################################################
# optimization
################################################################################
# projection
local_projection(θ) = projection(θ, polytope_dimensions,
	Alims=[-1.00, +1.00],
	blims=[+0.02, +0.60],
	olims=[-3.00, +3.00],
	)

# step_projection
local_step_projection(Δθ) = step_projection(Δθ, polytope_dimensions;
	Alims=[-0.40, +0.40],
	blims=[-0.05, +0.05],
	olims=[-0.05, +0.05],
	)
# regularization
θdiag = zeros(0)
for i = 1:np
	θi = [1e-2 * ones(2nh); 1e0 * ones(nh); 1e0 * ones(2)]
    A, b, o = unpack_halfspaces(θi)
    push!(θdiag, pack_halfspaces(A, b, o)...)
end
θdiag

parameters = Dict(
	:thickness => 0.2,
	:δ_sdf => 15.0,
	:δ_sigmoid => 0.1,
	:δ_softabs => 0.5,
	:altitude_threshold => 0.01,
	:rendering => 5.0,
	:sdf_matching => 20.0,
	:overlap => 0.0*2.0,
	:individual => 0.0*1.0,
	:side_regularization => 0.5,
	:shape_regularization => 0.0*0.5,
	:inside => 1.0,
	:outside => 0.1,
	:floor => 0.0*0.1,
)
max_iterations = 20

# loss and gradients
local_loss(θ) = shape_loss(θ, polytope_dimensions, e0, β0, ρ0, d0; parameters...)
local_grad(θ) = shape_grad(θ, polytope_dimensions, e0, β0, ρ0, d0; parameters...)
local_hess(θ) = Diagonal(1e-6*ones(length(θ)))

local_loss(θsol3)
local_loss(θinit)
local_grad(θinit)
local_hess(θinit)

ΔA_scale = 3e0
Δb_scale = 1e0
Δo_scale = 1e0
Δθ_scale = []
for nh in polytope_dimensions
	push!(Δθ_scale, [ΔA_scale * ones(2nh); Δb_scale * ones(nh); Δo_scale * ones(2)]...)
end
Δθ_scale
Δinit = 1e-1


################################################################################
# solve
################################################################################
local_loss(θ) = shape_loss(θ, polytope_dimensions, e0[1:5], β0[1:5], ρ0, d0[1:5]; parameters...)
local_grad(θ) = shape_grad(θ, polytope_dimensions, e0[1:5], β0[1:5], ρ0, d0[1:5]; parameters...)
adam_opt = Adam(θinit, local_loss, local_grad)
adam_opt.eps = 1e-8
adam_opt.a = 1e-2
θsol0, θiter0 = adam_solve!(adam_opt, max_iterations=2max_iterations)
visualize_iterates!(vis, θiter0, polytope_dimensions, e0, β0, ρ0, max_iterations=2max_iterations+1, color=iterate_color)

local_loss(θ) = shape_loss(θ, polytope_dimensions, e0[6:10], β0[6:10], ρ0, d0[6:10]; parameters...)
local_grad(θ) = shape_grad(θ, polytope_dimensions, e0[6:10], β0[6:10], ρ0, d0[6:10]; parameters...)
θsol1, θiter1 = adam_solve!(adam_opt, max_iterations=2max_iterations)
visualize_iterates!(vis, θiter1, polytope_dimensions, e0, β0, ρ0, max_iterations=2max_iterations+1, color=iterate_color)


local_loss(θ) = shape_loss(θ, polytope_dimensions, e0, β0, ρ0, d0; parameters...)
local_grad(θ) = shape_grad(θ, polytope_dimensions, e0, β0, ρ0, d0; parameters...)
θsol2, θiter2 = adam_solve!(adam_opt, max_iterations=2max_iterations)
visualize_iterates!(vis, θiter2, polytope_dimensions, e0, β0, ρ0, max_iterations=2max_iterations+1, color=iterate_color)



visualize_iterates!(vis, θiter0, polytope_dimensions, e0, β0, ρ0, max_iterations=2max_iterations+1, color=iterate_color)
visualize_iterates!(vis, θiter1, polytope_dimensions, e0, β0, ρ0, max_iterations=2max_iterations+1, color=iterate_color)
visualize_iterates!(vis, θiter2, polytope_dimensions, e0, β0, ρ0, max_iterations=2max_iterations+1, color=iterate_color)
visualize_iterates!(vis, [θiter0; θiter1; θiter2], polytope_dimensions, e0, β0, ρ0, max_iterations=6max_iterations+3, color=iterate_color)
