function trans_intersection(e::AbstractVector, v::AbstractVector, ρ,
		A::AbstractVector, b::AbstractVector, o::AbstractVector)
	d = 2
	nh = length(b)
	αmin = +Inf
    αmax = -Inf

	c = 0
    for i = 1:nh
		# @views ai = A[(i-1)*d .+ (1:d)]
		@views ai = A[i:nh:end]
        denum = ai' * v
        (abs(denum) < 1e-5) && continue
        α = (b[i] - ai' * e + ai' * o) / denum
        # x = e .- o .+ α .* v
		# s = maximum(A * x .- b)
        s = -Inf
		for j = 1:nh
			# @views aj = A[(j-1)*d .+ (1:d)]
			@views aj = A[j:nh:end]
			prod = aj'*e - aj'*o + α*aj'*v - b[j]
			s = max(s, prod)
		end
		if s <= 1e-10
			c += 1
			αmin = min(αmin, α)
			αmax = max(αmax, α)
		end
    end
	(c == 1) && (αmax = +Inf)
    return αmin, αmax
end

function trans_intersection!(α::AbstractMatrix, e::AbstractVector, v::AbstractVector, ρ, θ::AbstractVector,
		polytope_dimensions::Vector{Int}, perm=zeros(Int,length(polytope_dimensions)))

	np = length(polytope_dimensions)
	α .= +Inf

	off = 0
    for i = 1:np
		Ai, bi, oi = unpack_halfspaces(θ, polytope_dimensions, i)
		αmin, αmax = trans_intersection(e, v, ρ, Ai, bi, oi)
		α[1,i] = αmin
		α[2,i] = αmax
    end

	@views first_col = α[1, :]
	sortperm!(perm, first_col)
	α .= α[:, perm]
	return nothing
end

function transparency(α_sorted::Matrix, ρ)
	np = size(α_sorted, 2)

	α_trans = 0.0
	cum_e = 1.0
	for i = 1:np
		αmin = α_sorted[1,i]
		αmax = α_sorted[2,i]
		(αmin <= 0) && continue
		(αmin == Inf) && continue
		δ = αmax - αmin
		ex = exp(-δ/ρ)
		α_trans += αmin * (1 - ex) * cum_e
		cum_e *= ex
    end
	return α_trans
end

function trans_point_loss(α, α_hat)
	Δα = (α - α_hat)
	return 0.5 * Δα^2 + softabs(abs(Δα), 0.001)
end

function trans_point_loss(e::AbstractVector, β, ρ, θ::AbstractVector{D}, polytope_dimensions::Vector{Int}, d̂::Matrix) where D
	np = length(polytope_dimensions)
	nβ = length(β)
	α = zeros(D, 2, np)
	perm = zeros(Int,np)
	di = zeros(D, 2)

	l = 0.0
	for i = 1:nβ
		@views d̂i = d̂[:,i]
		v = SVector(cos(β[i]), sin(β[i]))
		α_hat = d̂i' * v - e' * v
		trans_intersection!(α, e, v, ρ, θ, polytope_dimensions, perm)
		α_trans = transparency(α, ρ)
		l += trans_point_loss(α_trans, α_hat)
	end
    return l / nβ
end

function point_cloud_smoothing(vis::Visualizer, e, β, θ, polytope_dimensions)
	set_floor!(vis)
	set_background!(vis)
	set_light!(vis)

	anim = MeshCat.Animation(20)
	for (i, ρ) in enumerate(range(0, 3.0, length=100))
		atframe(anim, i) do
			d = trans_point_cloud(e, β, exp(-log(10)*ρ), θ, polytope_dimensions)
			num_points = size(d, 2)
			set_2d_point_cloud!(vis, [e], [d]; name=:point_cloud)
		end
	end
	setanimation!(vis, anim)
	return vis, anim
end

function trans_point_cloud(e::AbstractVector, β, ρ, θ, polytope_dimensions) where T
	np = length(polytope_dimensions)
	nβ = length(β)
	d = zeros(2,nβ)
	α = zeros(2,np)

	off = 0
	for i = 1:nβ
		v = [cos(β[i]), sin(β[i])]
		trans_intersection!(α, e, v, ρ, θ, polytope_dimensions)
		d[:,i] = e + transparency(α, ρ) * v
	end
	return d
end
#
#
# polytope_dimensions = [4, 4, 4]
# θ = rand(3 * (3*4+2))
# αp1 = rand(2,10)
#
# e1 = [1, 3.0]
# v1 = [1, 0.0]
# # trans_point_cloud(e1, v1, ρ0, αp1)
# # @benchmark trans_point_cloud($e1, $v1, $ρ0, $αp1)
# perm = [1,2,3]
# α1 = zeros(2,3)
# trans_intersection!(α1, e1, v1, ρ0, θ, polytope_dimensions, perm)
# # @benchmark trans_intersection!($α1, $e1, $v1, $ρ0, $θ, $polytope_dimensions)
# α1
#
# d_ref = rand(2,100)
# β1 = range(-0.4π, -0.6π, length=100)
# trans_point_loss(e1, β1, ρ0, θ, polytope_dimensions, d_ref)
# # @benchmark trans_point_loss($e1, $β1, $ρ0, $θ, $polytope_dimensions, $d_ref)
#
#
#
#
# Ap = [Ap0, Ap1, Ap2, Af]
# bp = [bp0, bp1, bp2, bf]
# op = [op0, op1, op2, of]
#
# Ap = [Ap0, Af]
# bp = [bp0, bf]
# op = [op0, of]
# θ_p, polytope_dimensions_p = pack_halfspaces(Ap, bp, op)
#
# d0 = trans_point_cloud(e0, β0, ρ0*1, θ_p, polytope_dimensions_p)
# build_point_cloud!(vis[:point_cloud], nβ; color=RGBA(0.1,0.1,0.1,1), name=Symbol(1))
# set_2d_point_cloud!(vis, [e0], [d0]; name=:point_cloud)
#
# α = zeros(2,length(polytope_dimensions_p))
# β0 = range(-π/2-0.01, -π/2+0.01, length=100)
# β0 = range(-π/2-1.01, -π/2+1.01, length=100)
# e0 = [0, 2.0]
# v0 = [0.01, -1.0]
# v0 /= norm(v0)
# trans_intersection!(α, e0, v0, ρ0, θ_p, polytope_dimensions_p)
# α
# transparency(α, ρ0)
#
#
# Ai, bi, oi = unpack_halfspaces(θ_p, polytope_dimensions_p, 1)
# Ai, bi, oi = unpack_halfspaces(θ_p, polytope_dimensions_p, 2)
# Ai, bi, oi = unpack_halfspaces(θ_p, polytope_dimensions_p, 3)
# Ai, bi, oi = unpack_halfspaces(θ_p, polytope_dimensions_p, 4)
#
# polytope_dimensions_p[4]
