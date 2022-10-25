################################################################################
# contact
################################################################################
struct SphereHalfSpace{T,D,NP,NC} <: Contact{T,D,NP,NC}
    name::Symbol
    parent_name::Symbol
    index::NodeIndices
    parent_shape::Shape{T}
    child_shape::Shape{T}
    friction_coefficient::Vector{T}
end

function SphereHalfSpace(parent_body::AbstractBody{T,D}, child_shape::Shape{T};
        parent_shape_id::Int=1,
        name::Symbol=:contact,
        friction_coefficient=0.2) where {T,D}

    parent_name = parent_body.name
    parent_shape = deepcopy(parent_body.shapes[parent_shape_id])
    child_shape = deepcopy(child_shape)

    index = NodeIndices()

    Np = constraint_dimension(parent_shape)
    Nc = constraint_dimension(child_shape)
    return SphereHalfSpace{T,D,Np,Nc}(
        name,
        parent_name,
        index,
        parent_shape,
        child_shape,
        [friction_coefficient],
        )
end

primal_dimension(contact::SphereHalfSpace) = 0
cone_dimension(contact::SphereHalfSpace{T,2}) where T = 1 + 1 + 2 # γ ψ β
cone_dimension(contact::SphereHalfSpace{T,3}) where T = 1 + 1 + 4 # γ ψ β

function unpack_variables(x::Vector, contact::SphereHalfSpace{T,D}) where {T,D}
    num_cone = cone_dimension(contact)
    NC = 1
    off = 0
    nβ = (D - 1) * 2

    γ = x[off .+ (1:1)]; off += 1
    ψ = x[off .+ (1:1)]; off += 1
    β = x[off .+ (1:nβ)]; off += nβ

    sγ = x[off .+ (1:1)]; off += 1
    sψ = x[off .+ (1:1)]; off += 1
    sβ = x[off .+ (1:nβ)]; off += nβ
    return γ, ψ, β, sγ, sψ, sβ
end

function split_parameters(θ, contact::SphereHalfSpace{T}) where T
    friction_coefficient, parent_parameters, child_parameters = unpack_parameters(θ, contact)
    shape_p = contact.parent_shape
    shape_c = contact.child_shape
    radp, offp = unpack_parameters(shape_p, parent_parameters)
    normalc, offc = unpack_parameters(shape_c, child_parameters)
    return friction_coefficient, radp, offp, normalc, offc
end

function residual!(e, x, θ, contact::SphereHalfSpace, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    residual!(e, x, θ, contact, pbody)
    return nothing
end

function residual!(e, x, θ, contact::SphereHalfSpace{T,2},
        pbody::AbstractBody{T,2}) where T
    # unpack parameters
    friction_coefficient, radp, offp, normalc, offc = split_parameters(θ[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(θ[pbody.index.parameters], pbody)

    # unpack variables
    γ, ψ, β, sγ, sψ, sβ = unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    pp3 = pp2 + timestep_p[1] * vp25

    # analytical contact position in the world frame
    offpw = x_2d_rotation(pp3[3:3]) * offp
    contact_w = pp3[1:2] + offpw - radp[1] .* normalc
    # analytical signed distance function
    ϕ = [(contact_w - offc)' * normalc]

    # contact normal and tangent in the world frame
    normal_pw = normalc
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

function residual!(e, x, θ, contact::SphereHalfSpace{T,3},
        pbody::AbstractBody{T,3}) where T
    # unpack parameters
    friction_coefficient, radp, offp, normalc, offc = split_parameters(θ[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(θ[pbody.index.parameters], pbody)
    xp2 = pp2[1:3]
    qp2 = pp2[4:7]
    # unpack variables
    γ, ψ, β, sγ, sψ, sβ = unpack_variables(x[contact.index.variables], contact)
    vp25, ϕp25 = unpack_variables(x[pbody.index.variables], pbody)
    xp3 = xp2 + timestep_p[1] * vp25
    qp3 = quaternion_increment(qp2, timestep_p[1] * ϕp25)

    # analytical contact position in the world frame
    # offpw = vector_rotate(qp3, offp)
    # contact_w = xp3 + offpw - radp[1] .* normalc
    contact_w = xp3 - radp[1] .* normalc # we are missing offpw
    # analytical signed distance function
    ϕ = [(contact_w - offc)' * normalc]

    # contact normal and tangent in the world frame
    normal_pw = normalc
    tangent_candidate = [1, 0, 0]
    tangent_pw1 = (1 - tangent_candidate'*normal_pw) * tangent_candidate / (1 - tangent_candidate'*normal_pw)
    tangent_pw2 = cross(normal_pw, tangent_pw1)
    # rotation matrix from contact frame to world frame
    wRp = [tangent_pw1 tangent_pw2 normal_pw] # n points towards the parent body, [t,n,z] forms an oriented vector basis
    # @show wRp

    # force at the contact point in the contact frame
    f = [β[1:2] - β[3:4]; γ]
    # force at the contact point in the world frame
    f_pw = +wRp * f # parent
    # torques at the centers of masses in world frame
    τ_pw = skew(contact_w - xp3) * f_pw
    # overall wrench on both bodies in world frame
    # mapping the contact force into the generalized coordinates (at the centers of masses and in the world frame)
    wrench_p = [f_pw; τ_pw]

    # tangential velocities at the contact point
    tanvel_p = vp25 + skew(xp3 - contact_w) * 2ϕp25 # let's assume ϕp25 is the angular velocity
    tanvel_p = (1 - tanvel_p' * normal_pw) * tanvel_p
    tanvel_p = [tanvel_p' * tangent_pw1, tanvel_p' * tangent_pw2]
    tanvel = tanvel_p

    slackness = [
        sγ - ϕ;
        sψ - (friction_coefficient[1] * γ - [sum(β)]);
        sβ - ([+tanvel; -tanvel] + ψ[1]*ones(4));
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.index.slackness] .+= slackness
    e[pbody.index.optimality] .-= wrench_p
    return nothing
end
