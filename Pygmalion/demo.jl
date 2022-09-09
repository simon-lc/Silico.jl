using GeometryBasics
using Plots
using RobotVisualizer
using MeshCat
using Polyhedra
using Quaternions
using Optim
using StaticArrays
using ForwardDiff
using Clustering
using LinearAlgebra

include("../src/DojoLight.jl")
include("../system_identification/sr1.jl")
include("halfspace.jl")
include("transparency_point_cloud.jl")
include("visuals.jl")
include("softmax.jl")
include("utils.jl")

vis = Visualizer()
render(vis)
open(vis)
set_background!(vis)
set_light!(vis)
set_floor!(vis, color=RGBA(0.4,0.4,0.4,0.4))
iterate_color = RGBA(1,1,0,0.6)

# parameters
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

e0 = [0.0, +2.0]
ρ0 = 0.0002
nβ = 25
β0 = -π + atan(e0[2], e0[1]) .+ Vector(range(+0.3π, -0.3π, length=nβ))

build_2d_polytope!(vis[:polytope], Ap0, bp0 + Ap0 * op0, name=:poly0, color=RGBA(0,0,0,0.3))
build_2d_polytope!(vis[:polytope], Ap1, bp1 + Ap1 * op1, name=:poly1, color=RGBA(0,0,0,0.3))
build_2d_polytope!(vis[:polytope], Ap2, bp2 + Ap2 * op2, name=:poly2, color=RGBA(0,0,0,0.3))

d0 = trans_point_cloud(e0, β0, ρ0/100, θ_p, polytope_dimensions_p)
build_point_cloud!(vis[:point_cloud], nβ; color=RGBA(0.1,0.1,0.1,1), name=Symbol(1))
set_2d_point_cloud!(vis, [e0], [d0]; name=:point_cloud)

nh = 8
polytope_dimensions = [nh,nh,nh,nh,nh,nh]
np = length(polytope_dimensions)
θinit, d_object, kmres = parameter_initialization(d0, polytope_dimensions; altitude_threshold=0.3)
Ainit, binit, oinit = unpack_halfspaces(deepcopy(θinit), polytope_dimensions)
visualize_kmeans!(vis, θinit, polytope_dimensions, d_object, kmres)
polytope_dimensions
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
	:overlap => 2.0,
	:individual => 1.0,
	:side_regularization => 0.5,
	:shape_regularization => 0.5,
	:inside => 1.0,
	:outside => 0.1,
	:floor => 0.1,
)
max_iterations = 20

# loss and gradients
local_loss(θ) = shape_loss(θ, polytope_dimensions, [e0], [β0], ρ0, [d0]; parameters...)
local_grad(θ) = shape_grad(θ, polytope_dimensions, [e0], [β0], ρ0, [d0]; parameters...)
local_hess(θ) = Diagonal(1e-6*ones(length(θ)))


# local_loss(θsol3)
local_loss(θinit)
local_grad(θinit)
local_hess(θinit)
# parameters = Dict(
# 	:thickness => 0.2,
# 	:δ_sdf => 15.0,
# 	:δ_sigmoid => 0.1,
# 	:δ_softabs => 0.5,
# 	:altitude_threshold => 0.01,
# 	:rendering => 0.0 * 5.0,
# 	:sdf_matching => 0.0 * 20.0,
# 	:overlap => 0.0 * 2.0,
# 	:individual => 0.0 * 1.0,
# 	:side_regularization => 0.0 * 0.5,
# 	:shape_regularization => 1.0 * 0.05,
# 	:inside => 0.0 * 1.0,
# 	:outside => 0.0 * 0.1,
# 	:floor => 0.0 * 0.1,
# )
local_loss(θsol3)

ΔA_scale = 3e0#+0.60
Δb_scale = 1e0#+0.05
Δo_scale = 1e0# +0.05
Δθ_scale = []
for nh in polytope_dimensions
	push!(Δθ_scale, [ΔA_scale * ones(2nh); Δb_scale * ones(nh); Δo_scale * ones(2)]...)
end
Δθ_scale
Δinit = 1e-1

################################################################################
# solve
################################################################################
θsol0, θiter0 = sr1_solver!(θinit, local_loss, local_grad, local_projection, x->x;
        max_iterations=max_iterations,
        reg_min=1e-4,
        reg_max=1e2,
        reg_step=2.0,
        line_search_iterations=20,
        line_search_schedule=0.70,
        loss_tolerance=1e-4,
        grad_tolerance=1e-4,
		Δx_scale=Δθ_scale,
		Δinit=Δinit,
		Binit=Matrix(Diagonal(θdiag)))
visualize_iterates!(vis, θiter0, polytope_dimensions, e0, β0, ρ0, max_iterations=max_iterations+1, color=iterate_color)


# bfgs = BFGS(;
# 	alphaguess = Optim.LineSearches.InitialStatic(),
# 	linesearch = Optim.LineSearches.StrongWolfe(),
# 	initial_invH = initial_invH,
# 	initial_stepnorm = nothing,
# 	manifold = Flat(),
# 	)
#
# lbfgs = LBFGS(; m = 10,
# 	alphaguess = Optim.LineSearches.InitialStatic(),
# 	linesearch = Optim.LineSearches.Static(),
# 	P = nothing,
# 	# precondprep = (P, x) -> nothing,
# 	# manifold = Flat(),
# 	# scaleinvH0 = true && (typeof(P) <: Nothing)
# 	)
#
# res = Optim.optimize(
# 	local_loss,
# 	local_grad,
# 	θinit,
# 	bfgs,
# 	Optim.Options(
# 		iterations=40,
# 		allow_f_increases=true,
# 		# extended_trace = true,
# 		store_trace = true,
# 		show_trace = true,
# 		# show_trace = false,
# 		);
# 	inplace=false)
# res1 = res.trace[1]
# θsol0 = res.minimizer
# θiter0 = [θsol0 for i=1:31]
# visualize_iterates!(vis, θiter0, polytope_dimensions, e0, β0, ρ0, max_iterations=max_iterations+1)


e1 = [-2.25, +2.0]
β1 = -π + atan(e1[2], e1[1]) .+ Vector(range(+0.20π, -0.20π, length=nβ))
d1 = trans_point_cloud(e1, β1, ρ0/100, θ_p, polytope_dimensions_p)
build_point_cloud!(vis[:point_cloud_1], nβ; color=RGBA(0.1,0.1,0.8,1), name=Symbol(1))
set_2d_point_cloud!(vis, [e1], [d1]; name=:point_cloud_1)

# loss and gradients
local_loss(θ) = shape_loss(θ, polytope_dimensions, [e1], [β1], ρ0, [d1]; parameters...)
local_grad(θ) = shape_grad(θ, polytope_dimensions, [e1], [β1], ρ0, [d1]; parameters...)

θsol1, θiter1 = sr1_solver!(θsol0, local_loss, local_grad, local_projection, x->x;
        max_iterations=max_iterations,
        reg_min=1e-4,
        reg_max=1e2,
        reg_step=2.0,
        line_search_iterations=20,
        line_search_schedule=0.70,
        loss_tolerance=1e-4,
        grad_tolerance=1e-4,
		Δx_scale=Δθ_scale,
		Δinit=Δinit,
		Binit=Matrix(Diagonal(θdiag)))
visualize_iterates!(vis, θiter1, polytope_dimensions, e1, β1, ρ0, max_iterations=max_iterations+1, color=iterate_color)






e2 = [+1.25, +2.0]
β2 = -π + atan(e2[2], e2[1]) .+ Vector(range(+0.3π, -0.3π, length=nβ))
d2 = trans_point_cloud(e2, β2, ρ0/100, θ_p, polytope_dimensions_p)
build_point_cloud!(vis[:point_cloud_2], nβ; color=RGBA(0.1,0.8,0.1,1), name=Symbol(1))
set_2d_point_cloud!(vis, [e2], [d2]; name=:point_cloud_2)

# loss and gradients
local_loss(θ) = shape_loss(θ, polytope_dimensions, [e2], [β2], ρ0, [d2]; parameters...)
local_grad(θ) = shape_grad(θ, polytope_dimensions, [e2], [β2], ρ0, [d2]; parameters...)

θsol2, θiter2 = sr1_solver!(θsol1, local_loss, local_grad, local_projection, x->x;
        max_iterations=max_iterations,
        reg_min=1e-4,
        reg_max=1e2,
        reg_step=2.0,
        line_search_iterations=20,
        line_search_schedule=0.70,
        loss_tolerance=1e-4,
        grad_tolerance=1e-4,
		Δx_scale=Δθ_scale,
		Δinit=Δinit,
		Binit=Matrix(Diagonal(θdiag)))
visualize_iterates!(vis, θiter2, polytope_dimensions, e1, β1, ρ0, max_iterations=max_iterations+1, color=iterate_color)




# loss and gradients
local_loss(θ) = shape_loss(θ, polytope_dimensions, [e0, e1, e2], [β0, β1, β2], ρ0, [d0, d1, d2]; parameters...)
local_grad(θ) = shape_grad(θ, polytope_dimensions, [e0, e1, e2], [β0, β1, β2], ρ0, [d0, d1, d2]; parameters...)

θsol3, θiter3 = sr1_solver!(θsol2, local_loss, local_grad, local_projection, x->x;
        max_iterations=max_iterations,
        reg_min=1e-4,
        reg_max=1e2,
        reg_step=2.0,
        line_search_iterations=20,
        line_search_schedule=0.70,
        loss_tolerance=1e-4,
        grad_tolerance=1e-4,
		Δx_scale=Δθ_scale,
		Δinit=Δinit,
		Binit=Matrix(Diagonal(θdiag)))
visualize_iterates!(vis, θiter3, polytope_dimensions, e1, β1, ρ0, max_iterations=max_iterations+1, color=iterate_color)
