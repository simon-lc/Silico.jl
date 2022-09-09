################################################################################
# contact
################################################################################
struct SphereHalfSpace1170{T,D} <: Node{T}
    name::Symbol
    parent_name::Symbol
    index::NodeIndices1170
    friction_coefficient::Vector{T}
    parent_radius::Vector{T} #sphere
    A_child_collider::Matrix{T} #polytope
    b_child_collider::Vector{T} #polytope
end

function SphereHalfSpace1170(parent_body::Body{T}, Ac::AbstractMatrix, bc::AbstractVector;
        parent_collider_id::Int=1,
        name::Symbol=:halfspace,
        friction_coefficient=0.2) where {T}

    parent_name = parent_body.name
    parent_radius = parent_body.shapes[parent_collider_id].radius[1]

    return SphereHalfSpace1170(parent_name, friction_coefficient, parent_radius, Ac, bc;
        name=name)
end

function SphereHalfSpace1170(
        parent_name::Symbol,
        friction_coefficient,
        parent_radius,
        Ac::Matrix{T},
        bc::Vector{T};
        name::Symbol=:halfspace) where {T}

    d = size(Ac, 2)
    index = NodeIndices1170()
    return SphereHalfSpace1170{T,d}(
        name,
        parent_name,
        index,
        [friction_coefficient],
        [parent_radius],
        Ac,
        bc,
    )
end

primal_dimension(contact::SphereHalfSpace1170{T,D}) where {T,D} = 0
cone_dimension(contact::SphereHalfSpace1170{T,D}) where {T,D} = 1 + 1 + 2 # γ ψ β


function parameter_dimension(contact::SphereHalfSpace1170{T,D}) where {T,D}
    nAc = length(contact.A_child_collider)
    nbc = length(contact.b_child_collider)
    nθ = 1 + 1 + nAc + nbc
    return nθ
end

function unpack_variables(x::Vector, contact::SphereHalfSpace1170{T,D}) where {T,D}
    num_cone = cone_dimension(contact)
    NC = 1
    off = 0

    γ = x[off .+ (1:1)]; off += 1
    ψ = x[off .+ (1:1)]; off += 1
    β = x[off .+ (1:2)]; off += 2

    sγ = x[off .+ (1:1)]; off += 1
    sψ = x[off .+ (1:1)]; off += 1
    sβ = x[off .+ (1:2)]; off += 2
    return γ, ψ, β, sγ, sψ, sβ
end

function get_parameters(contact::SphereHalfSpace1170{T,D}) where {T,D}
    θ = [
        contact.friction_coefficient;
        contact.parent_radius;
        vec(contact.A_child_collider); contact.b_child_collider;
        ]
    return θ
end

function set_parameters!(contact::SphereHalfSpace1170{T,D}, θ) where {T,D}
    friction_coefficient, parent_radius, A_child_collider, b_child_collider =
        unpack_parameters(θ, contact)
    contact.friction_coefficient .= friction_coefficient
    contact.parent_radius .= parent_radius
    contact.A_child_collider .= A_child_collider
    contact.b_child_collider .= b_child_collider
    return nothing
end

function unpack_parameters(θ::Vector, contact::SphereHalfSpace1170{T,D}) where {T,D}
    @assert D == 2
    NC = 1
    off = 0
    friction_coefficient = θ[off .+ (1:1)]; off += 1
    parent_radius = θ[off .+ (1:1)]; off += 1
    A_child_collider = reshape(θ[off .+ (1:NC*D)], (NC,D)); off += NC*D
    b_child_collider = θ[off .+ (1:NC)]; off += NC
    return friction_coefficient, parent_radius, A_child_collider, b_child_collider
end

function residual!(e, x, θ, contact::SphereHalfSpace1170{T,D},
        pbody::Body) where {T,D}
    NC = 1
    # unpack parameters
    friction_coefficient, parent_radius, Ac, bc = unpack_parameters(θ[contact.index.parameters], contact)
    # pp2, vp15, up2, timestep_p, gravity_p, mass_p, inertia_p = unpack_parameters(θ[pbody.index.parameters], pbody)
    pp2, timestep_p = unpack_pose_timestep(θ[pbody.index.parameters], pbody)

    # unpack variables
    γ, ψ, β, sγ, sψ, sβ = unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = zeros(3)

    # analytical contact position in the world frame
    contact_w = pp3[1:2] - parent_radius[1] * Ac[1,:] # assumes the child is fized, other need a rotation here
    # analytical signed distance function
    ϕ = [contact_w' * Ac[1,:]] - bc
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])
    # contact_c is expressed in cbody's frame
    contact_c = x_2d_rotation(pc3[3:3])' * (contact_w - pc3[1:2])

    # contact normal and tangent in the world frame
    # normal_pw = -x_2d_rotation(pp3[3:3]) * Ap' * λp
    normal_pw = Ac[1,:]
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

    slackness = [
        sγ - ϕ;
        sψ - (friction_coefficient[1] * γ - [sum(β)]);
        sβ - ([+tanvel; -tanvel] + ψ[1]*ones(2));
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.index.slackness] .+= slackness
    e[pbody.index.optimality] .-= wrench_p
    return nothing
end

function residual!(e, x, θ, contact::SphereHalfSpace1170, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    residual!(e, x, θ, contact, pbody)
    return nothing
end
