include(joinpath(module_dir(), "Pygmalion/Pygmalion.jl"))
include("shape_loss.jl")

vis = Visualizer()
open(vis)
render(vis)
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
nβ = 30
eye_positions = [[ex, 2.0] for ex in range(-1.70, 1.70, length=ne)]
β0 = [-π + atan(e[2], e[1]) .+ Vector(range(+0.15π, -0.15π, length=nβ)) for e in eye_positions]

for i = 1:ne
	c = i > 5 ? RGBA(0.1,0.1,0.1,1) : RGBA(0.9,0.1,0.1,1)
	build_point_cloud!(vis[:point_cloud], nβ; color=c, name=Symbol(i))
end

α1, α_hit1, v1, v_hit1, e1, e_hit1 = vectorized_ray(eye_positions[1:5], β0[1:5], Ap, bp, op;
		altitude_threshold=0.01,
		max_length=100.0,
		)
α, α_hit, v, v_hit, e, e_hit = vectorized_ray(eye_positions, β0, Ap, bp, op;
		altitude_threshold=0.01,
		max_length=100.0,
		)

d0s = [e[:,(i-1)*nβ .+ (1:nβ)] + α[(i-1)*nβ .+ (1:nβ)]' .* v[:,(i-1)*nβ .+ (1:nβ)] for i = 1:ne]
d0_hit = e_hit .+ α_hit' .* v_hit
set_2d_point_cloud!(vis, eye_positions, d0s; name=:point_cloud)

################################################################################
# Initialization
################################################################################
nh = 5
polytope_dimensions = [nh,nh,nh,nh,nh,nh]
np = length(polytope_dimensions)

θinit, kmres = parameter_initialization(d0_hit, polytope_dimensions)
Ainit, binit, oinit = unpack_halfspaces(deepcopy(θinit), polytope_dimensions)
visualize_kmeans!(vis, θinit, polytope_dimensions, d0_hit, kmres)
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
	blims=[+0.05, +0.60],
	olims=[-3.00, +3.00],
	)

parameters = Dict(
	:thickness => 0.2,
	:δ_sdf => 0.025,
	:δ_sigmoid => 0.01,
	:δ_softabs => 0.5,
	:altitude_threshold => 0.01,
	:rendering => 1.0 * 5.0,
	:sdf_matching => 1.0 * 10.0,
	:side_regularization => 1.0 * 0.5,
	:shape_regularization => 0.0 * 0.5,
	:inside => 1.0 * 0.4,
	:outside => 1.0 * 0.1,
	:floor => 1.0 * 0.1,
	)

# loss and gradients
function local_loss(θ)
	A, b, bo = preprocess_halfspaces(θ, polytope_dimensions)
	l = shape_loss(
		α, α_hit, v, v_hit, e, e_hit,
		A, b, bo; parameters...
		)
	return l
end

shape_loss_gradient(
	α, α_hit, v, v_hit, e, e_hit,
	A, b, bo, parameters,
	) = gradient(keyword_shape_loss,
		α, α_hit, v, v_hit, e, e_hit,
		A, b, bo, parameters,
		)[7:9]

function local_grad(θ)
	A, b, bo = preprocess_halfspaces(θ, polytope_dimensions)
	grads = shape_loss_gradient(
		α, α_hit, v, v_hit, e, e_hit,
		A, b, bo, parameters,
		)
	dldAb = [vcat(vec.(grads[1])...); vcat(grads[2]...); vcat(grads[3]...)]
	dAbdθ = ForwardDiff.jacobian(θ -> preprocess_halfspaces(θ, polytope_dimensions, vectorize=true), θ)
	return  dAbdθ' * dldAb
end


local_loss(θinit)
local_loss(θsol0)
local_grad(θinit)

################################################################################
# solve
################################################################################
adam_opt = Adam(θinit, local_loss, local_grad)
adam_opt.eps = 1e-8
adam_opt.a = 8e-3
max_iterations = 20
@elapsed θsol0, θiter0 = adam_solve!(adam_opt, projection=local_projection, max_iterations=10max_iterations)

visualize_iterates!(vis, θiter0[1:10:end], polytope_dimensions, eye_positions[1],
 	β0, 1e-4, max_iterations=max_iterations+1, color=iterate_color)
Asol, bsol, osol = unpack_halfspaces(local_projection(θsol0), polytope_dimensions)







function local_loss(θ)
	A, b, bo = preprocess_halfspaces(θ, polytope_dimensions)
	l = shape_loss(
		α1, α_hit1, v1, v_hit1, e1, e_hit1,
		A, b, bo; parameters...
		)
	return l
end
function local_grad(θ)
	A, b, bo = preprocess_halfspaces(θ, polytope_dimensions)
	grads = shape_loss_gradient(
		α1, α_hit1, v1, v_hit1, e1, e_hit1,
		A, b, bo, parameters,
		)
	dldAb = [vcat(vec.(grads[1])...); vcat(grads[2]...); vcat(grads[3]...)]
	dAbdθ = ForwardDiff.jacobian(θ -> preprocess_halfspaces(θ, polytope_dimensions, vectorize=true), θ)
	return  dAbdθ' * dldAb
end

adam_opt = Adam(deepcopy(θsol0), local_loss, local_grad)
adam_opt.eps = 1e-8
adam_opt.a = 8e-3
max_iterations = 20
@elapsed θsol1, θiter1 = adam_solve!(adam_opt, projection=local_projection, max_iterations=10max_iterations)

visualize_iterates!(vis, θiter1[1:10:end], polytope_dimensions, eye_positions[1],
 	β0, 1e-4, max_iterations=max_iterations+1, color=iterate_color)
Asol, bsol, osol = unpack_halfspaces(local_projection(θsol0), polytope_dimensions)





#
# α, α_hit, v, v_hit, e, e_hit = vectorized_ray(eye_positions, β0, Ap, bp, op;
# 		altitude_threshold=0.01,
# 		max_length=100.0,
# 		)

# aa = [0.0 1.0]
# α = 10*[1,2,3,4,5,6,7,8,9,10]
# v = 10ones(2, 10)
# o = [1,2]
# aa * (α' .* v .- o) .- [1]
