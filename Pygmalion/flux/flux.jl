using Flux
using BenchmarkTools
include("../Pygmalion.jl")
include("shape_loss.jl")


vis = Visualizer()
# render(vis)
open(vis)
set_background!(vis)
set_light!(vis, direction="Negative")
set_floor!(vis, color=RGBA(0.4,0.4,0.4,0.4))


nβ = 1000
ρ0 = 1e-4

Af = [0.0  1.0]
bf = [0.0]
of = [0.0, 0.0]
Af1 = [-1.0 +0.0;
	   +1.0 +0.0;
	   +0.0 -1.0;
       +0.0 +1.0;
       ]
bf1 = 0.5 * [1,1,1,1]
of1 = [0.2, 0.5]
Ap = [Af1, Af]
bp = [bf1, bf]
op = [of1, of]
θ_p, polytope_dimensions_p = pack_halfspaces([Af1], [bf1], [of1])
θ_f, polytope_dimensions_f = pack_halfspaces(Ap, bp, op)
np = length(polytope_dimensions_f)
build_2d_polytope!(vis, Af1, bf1 + Af1 * of1, name=:f1)



e0 = [0, 2.0]
β0 = -π + atan(e0[2], e0[1]) .+ Vector(range(-0.25π, +0.25π, length=nβ))
build_point_cloud!(vis[:point_cloud], nβ; color=RGBA(0.9,0.1,0.1,1), name=Symbol(1))
At0, bt0 = preprocess_halfspaces(θ_p, polytope_dimensions_p)



α, α_hit, v, v_hit, e, e_hit = vectorized_ray([e0], [β0], Ap, bp, op;
		altitude_threshold=0.01,
		max_length=5.0,
		)

build_point_cloud!(vis[:point_cloud], nβ; color=RGBA(0.9,0.1,0.1,1), name=Symbol(1))
d0 = e + α' .* v
set_2d_point_cloud!(vis, [e0], [d0]; name=:point_cloud)

@elapsed shape_loss(
	α, α_hit, v, v_hit, e, e_hit,
	At0, bt0,
	)

shape_loss_gradient(
	α, α_hit, v, v_hit, e, e_hit,
	A, b,
	) = gradient(shape_loss,
		α, α_hit, v, v_hit, e, e_hit,
		A, b,
		)[7:8]

@elapsed shape_loss_gradient(
	α, α_hit, v, v_hit, e, e_hit,
	At0, bt0,
	)




#
#
#
#
# @elapsed αβ0 = rendering_loss(
#     α0,
#     v0,
#     AA,
#     bb,
#     )
#
# scatter(α0, ylims=(0, 5))
# scatter!(αβ0, linewidth=6.0)
#
# dAb_loss(α, v, AA, bb) = gradient(rendering_loss,
#     α,
#     v,
#     AA,
#     bb)[3:4]
#
# @elapsed dAb_loss(α0, v0, AA, bb)
#
#
# @elapsed l100 = inside_loss(
#     α0,
#     v0,
# 	e0,
#     AA,
#     bb,
#     )
#
# dAb_loss(α, v, e, AA, bb) = gradient(inside_loss,
#     α,
#     v,
# 	e,
#     AA,
#     bb)[3:4]
#
# @elapsed dAb_loss(α0, v0, e0, AA, bb)
#
#
#
# @elapsed l100 = outside_loss(
#     α0,
#     v0,
# 	e0,
#     AA,
#     bb,
#     )
#
# dAb_loss(α, v, e, AA, bb) = gradient(outside_loss,
#     α,
#     v,
# 	e,
#     AA,
#     bb)[3:4]
#
# @elapsed dAb_loss(α0, v0, e0, AA, bb)
#
# @elapsed l100 = floor_loss(
#     α0,
#     v0,
#     AA,
#     bb,
#     )
#
# dAb_loss(α, v, e, AA, bb) = gradient(floor_loss,
#     α,
#     v,
#     AA,
#     bb)[3:4]
#
# @elapsed dAb_loss(α0, v0, e0, AA, bb)
#
#
# @elapsed l100 = sdf_matching_loss(
#     α0,
#     v0,
#     AA,
#     bb,
#     )
#
# dAb_loss(α, v, e, AA, bb) = gradient(sdf_matching_loss,
#     α,
#     v,
#     AA,
#     bb)[3:4]
#
# @elapsed dAb_loss(α0, v0, e0, AA, bb)
