function unpack_halfspaces(θ::Vector{T}, polytope_dimensions::Vector{Int}) where T
    nθ = 2 .+ 3 .* polytope_dimensions
    # nθ = 2 .+ 2 .* polytope_dimensions
    m = length(polytope_dimensions)

    A = [zeros(T, i, 2) for i in polytope_dimensions]
    b = [zeros(T, i) for i in polytope_dimensions]
    o = [zeros(T, i) for i in polytope_dimensions]

    off = 0
    for i = 1:m
        A[i], b[i], o[i] = unpack_halfspaces(θ[off .+ (1:nθ[i])])
        off += nθ[i]
    end
    return A, b, o
end

function unpack_halfspaces(θ::AbstractVector, polytope_dimensions::AbstractVector, idx::Int)
    d = 2
	np = length(polytope_dimensions)
	nh = polytope_dimensions[idx]

	idx_start = 1 - (3polytope_dimensions[idx] + 2)
	idx_end = 0
	for i = 1:idx
		idx_start += 3polytope_dimensions[i] + 2
		idx_end += 3polytope_dimensions[i] + 2
	end
	@views θi = θ[idx_start:idx_end]
	@views Ai = θi[1:2*nh]
	@views bi = θi[2*nh .+ (1:nh)]
	@views oi = θi[3*nh .+ (1:2)]
    return Ai, bi, oi
end

function pack_halfspaces(A::Vector{Matrix{T}}, b::Vector{Vector{T}}, o::Vector{Vector{T}}) where T
    n = length(b)
    polytope_dimensions = length.(b)
    nθ = 2 .+ 3 .* polytope_dimensions
    # nθ = 2 .+ 2 .* polytope_dimensions

    θ = zeros(T,sum(nθ))

    off = 0
    for i = 1:n
        θ[off .+ (1:nθ[i])] .= pack_halfspaces(A[i], b[i], o[i])
        off += nθ[i]
    end
    return θ, polytope_dimensions
end

function unpack_halfspaces(θ::Vector{T}) where T
    nθ = length(θ)
    n = Int(floor((nθ - 2)/3))
    # n = Int(floor((nθ - 2)/2))

    A = zeros(T, n, 2)
    b = zeros(T, n)
    o = zeros(T, 2)

    for i = 1:n
        A[i,:] .= θ[2*(i-1) .+ (1:2)]
        # A[i,:] .= [cos(θ[i]), sin(θ[i])]
    end
    b .= θ[2n .+ (1:n)]
    # b .= θ[1n .+ (1:n)]
    o .= θ[3n .+ (1:2)]
    # o .= θ[2n .+ (1:2)]
    return A, b, o
end

function pack_halfspaces(A::Matrix{T}, b::Vector{T}, o::Vector{T}) where T
    n = length(b)
    θ = zeros(T,2+3n)
    # θ = zeros(T,2+2n)

    for i = 1:n
        θ[2*(i-1) .+ (1:2)] = A[i,1:2]
        # θ[i] = atan(A[i,2], A[i,1])
    end
    θ[2n .+ (1:n)] .= b
    # θ[1n .+ (1:n)] .= b
    θ[3n .+ (1:2)] .= o
    # θ[2n .+ (1:2)] .= o
    return θ
end

function add_floor(θ, polytope_dimensions)
    A, b, o = unpack_halfspaces(θ, polytope_dimensions)
    Af = [0 1.0]
    bf = [0.0]
    of = [0, 0.0]
    return pack_halfspaces([A..., Af], [b..., bf], [o..., of])
end

function add_floor(A, b, o)
    Af = [0.0 1.0]
    bf = [0.0]
    of = [0.0, 0.0]
    return [A..., Af], [b..., bf], [o..., of]
end

function halfspace_transformation(pose::Vector{T}, A::Matrix, b::Vector, o::Vector) where T
    position = pose[1:2]
    orientation = pose[3:3]

    wRb = x_2d_rotation(orientation) # rotation from body frame to world frame
    bRw = wRb' # rotation from world frame to body frame
    Ā = A * bRw
    b̄ = b + A * bRw * position
    ō = wRb * o
    return Ā, b̄, ō
end

function halfspace_transformation(pose::Vector{T}, A::Vector{<:Matrix{T2}},
        b::Vector{<:Vector{T2}}, o::Vector{<:Vector{T2}}) where {T,T2}
    np = length(b)
    Ā = Vector{Matrix{promote_type(T, T2)}}()
    b̄ = Vector{Vector{promote_type(T, T2)}}()
    ō = Vector{Vector{promote_type(T, T2)}}()
    for i = 1:np
        Āi, b̄i, ōi = halfspace_transformation(pose, A[i], b[i], o[i])
        push!(Ā, Āi)
        push!(b̄, b̄i)
        push!(ō, ōi)
    end
    return Ā, b̄, ō
end

function transform(A::Matrix, b::Vector, o::Vector, x::Vector)
	# transform a polytope encoded by A, b, o into a translated and rotated polytope
	# encoded by At, bt, ot
	# xw = p + wRb * xb
	# A * (xb - o) <= b
	# At * (xw - ot) <= bt

	p = x[1:2] # 2D position
	θ = x[3] # angle
	bRw = [cos(θ) sin(θ); -sin(θ) cos(θ)]
	wRb = bRw'
	At = deepcopy(A * bRw)
	bt = deepcopy(b)
	ot = deepcopy(wRb * o + p)
	return At, bt, ot
end

function transform(A::Vector{Matrix{T}}, b::Vector{Vector{T}}, o::Vector{Vector{T}}, x::Vector) where T
	np = length(A)
	At = Vector{Matrix{T}}()
	bt = Vector{Vector{T}}()
	ot = Vector{Vector{T}}()
	for i = 1:np
		Ati, bti, oti = transform(A[i], b[i], o[i], x)
		push!(At, Ati)
		push!(bt, bti)
		push!(ot, oti)
	end
	return At, bt, ot
end

function normalize_A!(A::Matrix; ϵ=1e-6)
	nh = size(A,2)
	for i = 1:nh
		A[:,i] ./= norm(A[:,i]) + ϵ
	end
	return nothing
end

# polytope_dimensions0 = [1,2,3,4,22]
# A0 = [rand(i,2) for i in polytope_dimensions0]
# b0 = [rand(i) for i in polytope_dimensions0]
# o0 = [rand(2) for i in polytope_dimensions0]
# for i = 1:5
#     for j = 1:polytope_dimensions0[i]
#         A0[i][j,:] ./= norm(A0[i][j,:])
#     end
# end
# θ1, polytope_dimensions1 = pack_halfspaces(A0, b0, o0)
# A1, b1, o1 = unpack_halfspaces(θ1, polytope_dimensions1)
# A1 .- A0
# b1 .- b0
# o1 .- o0
