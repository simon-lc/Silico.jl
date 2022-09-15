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
eye_positions = [[ex, 2.0] for ex in range(-1.70, 1.70, length=ne)]
β0 = [-π + atan(e[2], e[1]) .+ Vector(range(+0.15π, -0.15π, length=nβ)) for e in eye_positions]


for i = 1:ne
	c = i > 5 ? RGBA(0.1,0.1,0.1,1) : RGBA(0.9,0.1,0.1,1)
	build_point_cloud!(vis[:point_cloud], nβ; color=c, name=Symbol(i))
end

α, α_hit, v, v_hit, e, e_hit = vectorized_ray(eye_positions, β0, Ap, bp, op;
		altitude_threshold=0.1,
		max_length=100.0,
		)

d0s = [e[:,(i-1)*nβ .+ (1:nβ)] + α[(i-1)*nβ .+ (1:nβ)]' .* v[:,(i-1)*nβ .+ (1:nβ)] for i = 1:ne]
d0 = hcat(d0s...)
d0_hit = e_hit .+ α_hit' .* v_hit
set_2d_point_cloud!(vis, eye_positions, d0s; name=:point_cloud)

################################################################################
# Initialization
################################################################################
nh = 8
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

# step_projection
local_step_projection(Δθ) = step_projection(Δθ, polytope_dimensions;######################################
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
	:rendering => 1.0 * 5.0,
	:sdf_matching => 01.0 * 20.0,############################################
	:overlap => 0.0 * 2.0,
	:individual => 0.0 * 1.0,
	:side_regularization => 1.0 * 0.5,
	:shape_regularization => 1.0 * 0.5,
	:inside => 1.0 * 1.0,
	:outside => 00.0 * 0.1, ################################
	:floor => 00.0 * 0.1, ##############################
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
	dAbdθ = ForwardDiff.jacobian(θ -> vec_preprocess_halfspaces(θ, polytope_dimensions), θ)
	return  dAbdθ' * dldAb
end

function vec_preprocess_halfspaces(θ::Vector{T}, polytope_dimensions) where T
	θ_floor, polytope_dimensions_floor = add_floor(θ, polytope_dimensions)
	np = length(polytope_dimensions_floor)
	A, b, o = unpack_halfspaces(θ_floor, polytope_dimensions_floor)
	bo = [b[i] + (A[i] * o[i]) for i = 1:np]
	# return A, b
	return [vcat(vec.(A)...); vcat(b...); vcat(bo...)]
end


vec_preprocess_halfspaces(θinit, polytope_dimensions)
local_loss(θinit)
local_loss(θsol0)
local_grad(θinit)

################################################################################
# solve
################################################################################
# local_loss(θ) = shape_loss(θ, polytope_dimensions, e0[1:5], β0[1:5], ρ0, d0[1:5]; parameters...)
# local_grad(θ) = shape_grad(θ, polytope_dimensions, e0[1:5], β0[1:5], ρ0, d0[1:5]; parameters...)
adam_opt = Adam(θinit, local_loss, local_grad)
adam_opt.eps = 1e-8
adam_opt.a = 6e-3
max_iterations = 20
@elapsed θsol0, θiter0 = adam_solve!(adam_opt, projection=local_projection, max_iterations=10max_iterations)

visualize_iterates!(vis, θiter0[1:10:end], polytope_dimensions, e0, β0, ρ0, max_iterations=max_iterations+1, color=iterate_color)
Asol, bsol, osol = unpack_halfspaces(θsol0, polytope_dimensions)

for i = 1:np
	plt = plot_polytope(Asol[i], bsol[i], 100.0, xlims=(-3,3), ylims=(-3,3))
	display(plt)
	sleep(0.5)
end



# local_loss(θ) = shape_loss(θ, polytope_dimensions, e0[6:10], β0[6:10], ρ0, d0[6:10]; parameters...)
# local_grad(θ) = shape_grad(θ, polytope_dimensions, e0[6:10], β0[6:10], ρ0, d0[6:10]; parameters...)
# θsol1, θiter1 = adam_solve!(adam_opt, max_iterations=2max_iterations)
# visualize_iterates!(vis, θiter1, polytope_dimensions, e0, β0, ρ0, max_iterations=2max_iterations+1, color=iterate_color)
#
#
# local_loss(θ) = shape_loss(θ, polytope_dimensions, e0, β0, ρ0, d0; parameters...)
# local_grad(θ) = shape_grad(θ, polytope_dimensions, e0, β0, ρ0, d0; parameters...)
# θsol2, θiter2 = adam_solve!(adam_opt, max_iterations=2max_iterations)
# visualize_iterates!(vis, θiter2, polytope_dimensions, e0, β0, ρ0, max_iterations=2max_iterations+1, color=iterate_color)
#
#
#
# visualize_iterates!(vis, θiter0, polytope_dimensions, e0, β0, ρ0, max_iterations=2max_iterations+1, color=iterate_color)
# visualize_iterates!(vis, θiter1, polytope_dimensions, e0, β0, ρ0, max_iterations=2max_iterations+1, color=iterate_color)
# visualize_iterates!(vis, θiter2, polytope_dimensions, e0, β0, ρ0, max_iterations=2max_iterations+1, color=iterate_color)
# visualize_iterates!(vis, [θiter0; θiter1; θiter2], polytope_dimensions, e0, β0, ρ0, max_iterations=6max_iterations+3, color=iterate_color)
