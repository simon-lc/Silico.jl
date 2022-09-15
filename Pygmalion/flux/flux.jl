using Flux
using CUDA
using BenchmarkTools

CUDA.functional()

vis = Visualizer()
# render(vis)
open(vis)
set_background!(vis)
set_light!(vis, direction="Negative")
set_floor!(vis, color=RGBA(0.4,0.4,0.4,0.4))


Tf = Float32
nβ = 1000
# nβ = 1
e0 = [0.0, 3.0]
ρ0 = 1e-4


Af = [0.5  0.9]
bf = [0.1]
of = [0.2, 0.1]

Af1 = [0 1;
       # +sqrt(2)/2 +sqrt(2)/2;
       +sqrt(2)/2 -sqrt(2)/2;
       # -sqrt(2)/2 -sqrt(2)/2;
       -1.0 0.0;
       ]
bf1 = [0.7, -0.5, 2.0]
of1 = [0.2, 0.1]
θinit, polytope_dimensions = pack_halfspaces([Af, Af1], [bf, bf1], [of, of1])
np = length(polytope_dimensions)

build_2d_polytope!(vis, Af1, bf1 + Af1 * of1, name=:f1)

Ap = [Af1, Af]
bp = [bf1, bf]
op = [of1, of]
θinit, polytope_dimensions = pack_halfspaces(Ap, bp, op)
# for i = 1:3
    # build_2d_polytope!(vis, Ap[i], bp[i] + Ap[i] * op[i], name=Symbol(i))
# end

e0 = [0, 2.0]
# β0 = -π + atan(e0[2], e0[1]) .+ Vector(range(-0.0π, -0.0π, length=nβ))
β0 = -π + atan(e0[2], e0[1]) .+ Vector(range(-0.35π, +0.35π, length=nβ))
# β0 = -π + atan(e0[2], e0[1]) .+ Vector(range(+0.10π, -0.10π, length=nβ))
# β0 = -π + atan(e0[2], e0[1]) .+ Vector(range(+0.00π, -0.00π, length=nβ))
# β0 = -π + atan(e0[2], e0[1]) .+ Vector(range(-0.05π, -0.05π, length=nβ))
d0 = trans_point_cloud(e0, β0, ρ0, θinit, polytope_dimensions)
plot(d0[1,:], d0[2,:])
build_point_cloud!(vis[:point_cloud], nβ; color=RGBA(0.9,0.1,0.1,1), name=Symbol(1))
set_2d_point_cloud!(vis, [e0], [d0]; name=:point_cloud)


v0 = hcat([[cos(β0[i]), sin(β0[i])] for i=1:nβ]...)
α0 = [d0[:,i]'*v0[:,i] - e0'*v0[:,i] for i=1:nβ]
AA = [Matrix(reshape(unpack_halfspaces(θinit, polytope_dimensions, i)[1], (polytope_dimensions[i],2))) for i = 1:np]
bb = [Vector(unpack_halfspaces(θinit, polytope_dimensions, i)[2]) for i = 1:np]
oo = [Vector(unpack_halfspaces(θinit, polytope_dimensions, i)[3]) for i = 1:np]
bb = [bb[i] + AA[i] * oo[i] - AA[i] * e0 for i=1:np]

α0 = convert.(Tf, α0)
v0 = convert.(Tf, v0)
AA = [convert.(Tf, Ai) for Ai in AA]
bb = [convert.(Tf, bi) for bi in bb]

function rendering_loss(α_prev::AbstractVector, α_ref::AbstractVector, v::AbstractMatrix,
    A::AbstractMatrix, b::AbstractVector; max_length=5.0)

    # sdf nβ
    # sdfv nh x nβ
    # Av nh x nβ
    # αβ nβ
    # αhβ nh x nβ
    # α_ref nβ
    # v 2
    # A nh x 2
    # b nh

    nβ = length(α_ref)
    nh = length(b)

    αβ = α_prev
    Av = A * v
    αhβ = max.(0, Av .\ b) # we apply max to avoid intersection with object behind the camera.

    for i = 1:nh
        # for a given half-space i compute the value of the all half-spaces for all rays j
        sdfv = αhβ[i,:]' .* Av .- b
        # compute sdf for all rays j hitting the half-space i as the maximum over all half-spaces
        sdf = vec(findmax(sdfv, dims=1)[1])
        # compute condition for updating α
        cnd = (sdf .< 1e-5) .&& (αhβ[i,:] .< αβ)
        # update value of α if condition is met
        αβ = cnd .* αhβ[i,:] + (1 .- cnd) .* αβ
    end
    # we obtain the length of each ray after intersection with polytope A, b
    return αβ
end

function rendering_loss(α_ref::AbstractVector, v::AbstractMatrix{T},
    A::AbstractVector, b::AbstractVector; max_length=5.0) where T

    nβ = length(α_ref)
    np = length(b)

    αβ = max_length * ones(T,nβ)
    for i = 1:np
        αβ = rendering_loss(αβ, α_ref, v, A[i], b[i], max_length=max_length)
    end
    return sum((α_ref .- αβ).^2) / nβ
    # return αβ
end

@elapsed αβ0 = rendering_loss(
    α0,
    v0,
    AA,
    bb,
    )

scatter(α0, ylims=(0, 5))
scatter!(αβ0, linewidth=6.0)

dAb_loss(α, v, AA, bb) = gradient(rendering_loss,
    α,
    v,
    AA,
    bb)[3:4]

@elapsed dAb_loss(α0, v0, AA, bb)

aaa = 10
aaa = 10
aaa = 10
aaa = 10
aaa = 10
aaa = 10

# @benchmark dAb_loss(α0, v0, AA[1], bb[1])


# @benchmark ForwardDiff.gradient(b -> rendering_loss(
#     α0,
#     v0,
#     AA[1],
#     b,
#     ),
#     bb[1])
# @benchmark ForwardDiff.gradient(A -> rendering_loss(
#     α0,
#     v0,
#     A,
#     bb[1],
#     ),
#     AA[1])

# @benchmark rendering_loss(
#     cnd,
#     sdf0,
#     sdfv,
#     Av,
#     αβ,
#     αhβ,
#     α0,
#     v0,
#     AA[1],
#     bb[1],
#     )

# @benchmark rendering_loss(
#     cu_cnd,
#     cu_sdf0,
#     cu_sdfv,
#     cu_Av,
#     cu_αβ,
#     cu_αhβ,
#     cu_α0,
#     cu_v0,
#     cu_AA[1],
#     cu_bb[1],
#     )
#

# Main.@profiler [rendering_loss(
#     cu_cnd,
#     cu_sdf0,
#     cu_sdfv,
#     cu_Av,
#     cu_αβ,
#     cu_αhβ,
#     cu_α0,
#     cu_v0,
#     cu_AA[1],
#     cu_bb[1],
#     ) for i = 1:1000]
#

#
# function rendering_loss(cond::AbstractVector, sdf::AbstractVector, sdfv::AbstractMatrix,
#     Av::AbstractMatrix, αβ::AbstractVector, αhβ::AbstractMatrix,
#     α_ref::AbstractVector{T}, v::AbstractMatrix{T},
#     A::AbstractMatrix{T}, b::AbstractVector{T}) where T
#
#     # sdf nβ
#     # sdfv nh x nβ
#     # Av nh x nβ
#     # αβ nβ
#     # αhβ nh x nβ
#     # α_ref nβ
#     # v 2
#     # A nh x 2
#     # b nh
#
#     αβ .= +Inf
#     Av .= A * v
#     # αhβ .= Av
#     # for i = 1:nh
#     #     αhβ[i,:] ./= b[i]
#     # end
#     αhβ .= Av ./ b
#     for i = 1:nh
#         # for j = 1:nh
#         #     sdfv[j,:] .= αhβ[i,:] .* Av[j,:] .- b[i]
#         # end
#         @views sdfv .=  αhβ[i,:]' .* Av .- b[i]
#
#         sdf .= maximum(sdfv, dims=1)[1,:]
#         @views cond .= (sdf .< 1e-5) .&& (αhβ[i,:] .< αβ)
#         @views αβ .= αβ .* .!cond + αhβ[i,:] .* cond
#     end
#     return sum((α_ref .- αβ).^2) / nβ#, αβ
# end


# cu_α0 = cu(α0)
# cu_v0 = cu(v0)
# cu_AA = cu.(AA)
# cu_bb = cu.(bb)
# cu_cnd = cu(cnd)
# cu_sdf0 = cu(sdf0)
# cu_sdfv = cu(sdfv)
# cu_Av = cu(Av)
# cu_αβ = cu(αβ)
# cu_αhβ = cu(αhβ)

# cnd = zeros(Bool, nβ)
# sdf0 = zeros(Tf, nβ)
# sdfv = zeros(Tf, nh, nβ)
# Av = zeros(Tf, nh, nβ)
# αβ = zeros(Tf, nβ)
# αhβ = zeros(Tf, nh, nβ)

# @elapsed rendering_loss(
#     cnd,
#     sdf0,
#     sdfv,
#     Av,
#     αβ,
#     αhβ,
#     α0,
#     v0,
#     AA[1],
#     bb[1],
#     )
#
# @elapsed rendering_loss(
#     cu_cnd,
#     cu_sdf0,
#     cu_sdfv,
#     cu_Av,
#     cu_αβ,
#     cu_αhβ,
#     cu_α0,
#     cu_v0,
#     cu_AA[1],
#     cu_bb[1],
#     )



#
# function rendering_loss(α_prev::AbstractVector, α_ref::AbstractVector, v::AbstractMatrix,
#     A::AbstractMatrix, b::AbstractVector)
#
#     # sdf nβ
#     # sdfv nh x nβ
#     # Av nh x nβ
#     # αβ nβ
#     # αhβ nh x nβ
#     # α_ref nβ
#     # v 2
#     # A nh x 2
#     # b nh
#
#     nβ = length(α_ref)
#     nh = length(b)
#
#     αβ = α_prev
#     Av = A * v
#     αhβ = max.(0, Av .\ b) # we apply max to avoid intersection with object behind the camera.
#
#     sdfv = zeros(nh, nβ)
#     sdf = zeros(nβ)
#     for i = 1:nh
#         for j = 1:nβ
#             sdfv[:,j] = αhβ[i,j] .* Av[:,j] .- b
#         end
#         for j = 1:nβ
#             sdf[j] = maximum(sdfv[:,j])
#         end
#         for j = 1:nβ
#             cnd = (sdf[j] < 1e-5) && (αhβ[i,j] < αβ[j])
#             if cnd
#                 αβ[j] = αhβ[i,j]
#             end
#         end
#     end
#     return αβ
# end
#
