################################################################################
# contact
################################################################################
struct PolyHalfSpace{T,D,NP} <: Node{T}
    name::Symbol
    parent_name::Symbol
    index::NodeIndices
    friction_coefficient::Vector{T}
    A_parent_collider::Matrix{T} #polytope
    b_parent_collider::Vector{T} #polytope
    A_child_collider::Matrix{T} #polytope
    b_child_collider::Vector{T} #polytope
end

function PolyHalfSpace(parent_body::AbstractBody{T}, Ac::AbstractMatrix, bc::AbstractVector;
        parent_shape_id::Int=1,
        name::Symbol=:halfspace,
        friction_coefficient=0.2) where {T}

    parent_name = parent_body.name
    Ap = copy(parent_body.shapes[parent_shape_id].A)
    bp = copy(parent_body.shapes[parent_shape_id].b)

    return PolyHalfSpace(parent_name, friction_coefficient, Ap, bp, Ac, bc;
        name=name)
end

function PolyHalfSpace(
        parent_name::Symbol,
        friction_coefficient,
        Ap::Matrix{T},
        bp::Vector{T},
        Ac::Matrix{T},
        bc::Vector{T};
        name::Symbol=:halfspace) where {T}

    d = size(Ap, 2)
    np = size(Ap, 1)
    index = NodeIndices()
    return PolyHalfSpace{T,d,np}(
        name,
        parent_name,
        index,
        [friction_coefficient],
        Ap,
        bp,
        Ac,
        bc,
    )
end

primal_dimension(contact::PolyHalfSpace{T,D}) where {T,D} = D + 1 # x, ϕ
cone_dimension(contact::PolyHalfSpace{T,D,NP}) where {T,D,NP} = 1 + 1 + 2 + NP + 1 # γ ψ β λp, λc

function parameter_dimension(contact::PolyHalfSpace{T,D}) where {T,D}
    nAp = length(contact.A_parent_collider)
    nbp = length(contact.b_parent_collider)
    nAc = length(contact.A_child_collider)
    nbc = length(contact.b_child_collider)
    nθ = 1 + nAp + nbp + nAc + nbc
    return nθ
end

function unpack_variables(x::Vector, contact::PolyHalfSpace{T,D,NP}) where {T,D,NP}
    NC = 1
    off = 0
    c = x[off .+ (1:2)]; off += 2
    ϕ = x[off .+ (1:1)]; off += 1

    γ = x[off .+ (1:1)]; off += 1
    ψ = x[off .+ (1:1)]; off += 1
    β = x[off .+ (1:2)]; off += 2
    λp = x[off .+ (1:NP)]; off += NP
    λc = x[off .+ (1:NC)]; off += NC

    sγ = x[off .+ (1:1)]; off += 1
    sψ = x[off .+ (1:1)]; off += 1
    sβ = x[off .+ (1:2)]; off += 2
    sp = x[off .+ (1:NP)]; off += NP
    sc = x[off .+ (1:NC)]; off += NC
    return c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc
end

function get_parameters(contact::PolyHalfSpace{T,D}) where {T,D}
    θ = [
        contact.friction_coefficient;
        vec(contact.A_parent_collider); contact.b_parent_collider;
        vec(contact.A_child_collider); contact.b_child_collider;
        ]
    return θ
end

function set_parameters!(contact::PolyHalfSpace{T,D,NP}, θ) where {T,D,NP}
    friction_coefficient, A_parent_collider, b_parent_collider, A_child_collider, b_child_collider =
        unpack_parameters(θ, contact)
    contact.friction_coefficient .= friction_coefficient
    contact.A_parent_collider .= A_parent_collider
    contact.b_parent_collider .= b_parent_collider
    contact.A_child_collider .= A_child_collider
    contact.b_child_collider .= b_child_collider
    return nothing
end

function unpack_parameters(θ::Vector, contact::PolyHalfSpace{T,D,NP}) where {T,D,NP}
    @assert D == 2
    NC = 1
    off = 0
    friction_coefficient = θ[off .+ (1:1)]; off += 1
    A_parent_collider = reshape(θ[off .+ (1:NP*D)], (NP,D)); off += NP*D
    b_parent_collider = θ[off .+ (1:NP)]; off += NP
    A_child_collider = reshape(θ[off .+ (1:NC*D)], (NC,D)); off += NC*D
    b_child_collider = θ[off .+ (1:NC)]; off += NC
    return friction_coefficient, A_parent_collider, b_parent_collider, A_child_collider, b_child_collider
end

function residual!(e, x, θ, contact::PolyHalfSpace{T,D,NP},
        pbody::AbstractBody) where {T,D,NP}
    NC = 1
    # unpack parameters
    friction_coefficient, Ap, bp, Ac, bc = unpack_parameters(θ[contact.index.parameters], contact)
    # pp2, vp15, up2, timestep_p, gravity_p, mass_p, inertia_p = unpack_parameters(θ[pbody.index.parameters], pbody)
    pp2, timestep_p = unpack_pose_timestep(θ[pbody.index.parameters], pbody)

    # unpack variables
    c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc = unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = zeros(3)

    # contact position in the world frame
    contact_w = c + pp3[1:2]
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])
    # contact_c is expressed in cbody's frame
    contact_c = x_2d_rotation(pc3[3:3])' * (contact_w - pc3[1:2])

    # contact normal and tangent in the world frame
    normal_pw = -x_2d_rotation(pp3[3:3]) * Ap' * λp
    # normal_pw = Ac[1,:]
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw

    # rotation matrix from contact frame to world frame
    wRp = [tangent_pw normal_pw] # n points towards the parent body, [t,n,z] forms an oriented vector basis

    # force at the contact point in the contact frame
    f = [β[1] - β[2]; γ]
    # force at the contact point in the world frame
    f_pw = +wRp * f # parent
    # torques at the centers of masses in world frame
    τ_pw = (skew([contact_w - pp3[1:2]; 0]) * [f_pw; 0])[3:3]
    # overall wrench on both bodies in world frame
    # mapping the contact force into the generalized coordinates (at the centers of masses and in the world frame)
    wrench_p = [f_pw; τ_pw]

    # tangential velocities at the contact point
    tanvel_p = vp25[1:2] + (skew([pp3[1:2] - contact_w; 0]) * [zeros(2); vp25[3]])[1:2]
    tanvel_p = tanvel_p' * tangent_pw
    tanvel = tanvel_p

    # contact equality
    optimality = [
        x_2d_rotation(pp3[3:3]) * Ap' * λp + x_2d_rotation(pc3[3:3]) * Ac' * λc;
        1 - sum(λp) - sum(λc);
    ]
    slackness = [
        sγ - ϕ;
        sψ - (friction_coefficient[1] * γ - [sum(β)]);
        sβ - ([+tanvel; -tanvel] + ψ[1]*ones(2));
        sp - (- Ap * contact_p + bp + ϕ .* ones(NP));
        sc - (- Ac * contact_c + bc + ϕ .* ones(NC));
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.index.optimality] .+= optimality
    e[contact.index.slackness] .+= slackness
    e[pbody.index.optimality] .-= wrench_p
    return nothing
end

function residual!(e, x, θ, contact::PolyHalfSpace, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    residual!(e, x, θ, contact, pbody)
    return nothing
end
