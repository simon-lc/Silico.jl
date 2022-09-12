################################################################################
# contact
################################################################################
struct SphereSphere{T,D} <: Node{T}
    name::Symbol
    parent_name::Symbol
    child_name::Symbol
    index::NodeIndices
    friction_coefficient::Vector{T}
    parent_radius::Vector{T}
    parent_position_offset::Vector{T}
    child_radius::Vector{T}
    child_position_offset::Vector{T}
end

function SphereSphere(parent_body::Body{T}, child_body::Body{T};
        parent_collider_id::Int=1,
        child_collider_id::Int=1,
        name::Symbol=:contact,
        friction_coefficient=0.2) where {T}

    parent_name = parent_body.name
    child_name = child_body.name
    radp = copy(parent_body.shapes[parent_collider_id].radius)
    offp = copy(parent_body.shapes[parent_collider_id].position_offset)
    radc = copy(child_body.shapes[child_collider_id].radius)
    offc = copy(child_body.shapes[child_collider_id].position_offset)

    return SphereSphere(parent_name, child_name, friction_coefficient,
        radp, offp, radc, offc;
        name=name)
end

function SphereSphere(
        parent_name::Symbol,
        child_name::Symbol,
        friction_coefficient,
        radp::Vector{T},
        offp::Vector{T},
        radc::Vector{T},
        offc::Vector{T};
        name::Symbol=:contact) where {T}

    index = NodeIndices()
    return SphereSphere{T,2}(
        name,
        parent_name,
        child_name,
        index,
        [friction_coefficient],
        radp,
        offp,
        radc,
        offc,
    )
end

primal_dimension(contact::SphereSphere{T,D}) where {T,D} = 0
cone_dimension(contact::SphereSphere{T,D}) where {T,D} = 1 + 1 + 2 # γ ψ β

function parameter_dimension(contact::SphereSphere{T,D}) where {T,D}
    nθ = 1 + 1 + D + 1 + D
    return nθ
end

function unpack_variables(x::Vector, contact::SphereSphere{T,D}) where {T,D}
    num_cone = cone_dimension(contact)
    off = 0

    γ = x[off .+ (1:1)]; off += 1
    ψ = x[off .+ (1:1)]; off += 1
    β = x[off .+ (1:2)]; off += 2

    sγ = x[off .+ (1:1)]; off += 1
    sψ = x[off .+ (1:1)]; off += 1
    sβ = x[off .+ (1:2)]; off += 2
    return γ, ψ, β, sγ, sψ, sβ
end

function get_parameters(contact::SphereSphere{T,D}) where {T,D}
    θ = [
        contact.friction_coefficient;
        vec(contact.parent_radius); contact.parent_position_offset;
        vec(contact.child_radius); contact.child_position_offset;
        ]
    return θ
end

function set_parameters!(contact::SphereSphere{T,D}, θ) where {T,D}
    friction_coefficient, parent_radius, parent_position_offset, child_radius, child_position_offset =
        unpack_parameters(θ, contact)
    contact.friction_coefficient .= friction_coefficient
    contact.parent_radius .= parent_radius
    contact.parent_position_offset .= parent_position_offset
    contact.child_radius .= child_radius
    contact.child_position_offset .= child_position_offset
    return nothing
end

function unpack_parameters(θ::Vector, contact::SphereSphere{T,D}) where {T,D}
    @assert D == 2
    off = 0
    friction_coefficient = θ[off .+ (1:1)]; off += 1
    parent_radius = θ[off .+ (1:1)]; off += 1
    parent_position_offset = θ[off .+ (1:D)]; off += D
    child_radius = θ[off .+ (1:1)]; off += 1
    child_position_offset = θ[off .+ (1:D)]; off += D
    return friction_coefficient, parent_radius, parent_position_offset, child_radius, child_position_offset
end

function residual!(e, x, θ, contact::SphereSphere{T,D},
        pbody::Body, cbody::Body) where {T,D}

    # unpack parameters
    friction_coefficient, radp, offp, radc, offc = unpack_parameters(θ[contact.index.parameters], contact)
    # pp2, vp15, up2, timestep_p, gravity_p, mass_p, inertia_p = unpack_parameters(θ[pbody.index.parameters], pbody)
    # pc2, vc15, uc2, timestep_c, gravity_c, mass_c, inertia_c = unpack_parameters(θ[cbody.index.parameters], cbody)
    pp2, timestep_p = unpack_pose_timestep(θ[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(θ[cbody.index.parameters], cbody)

    # unpack variables
    γ, ψ, β, sγ, sψ, sβ = unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    vc25 = unpack_variables(x[cbody.index.variables], cbody)
    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25
    # signed distance function
    ϕ = [norm(pp3[1:2] - pc3[1:2])] - radp - radc

    # contact normal and tangent in the world frame
    normal_pw = (pp3 - pc3)[1:2]
    normal_cw = (pp3 - pc3)[1:2]
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw
    tangent_cw = R * normal_cw

    # contact position in the world frame
    n = normal_pw / (1e-6 + norm(normal_pw))
    contact_w = 0.5 * (pp3[1:2] + radp[1] * n + pc3[1:2] - radc[1] * n)

    # rotation matrix from contact frame to world frame
    wRp = [tangent_pw normal_pw] # n points towards the parent body, [t,n,z] forms an oriented vector basis
    wRc = [tangent_cw normal_cw] # n points towards the parent body, [t,n,z] forms an oriented vector basis

    # force at the contact point in the contact frame
    f = [β[1] - β[2]; γ]
    # force at the contact point in the world frame
    f_pw = +wRp * f # parent
    f_cw = -wRc * f # child
    # torques at the centers of masses in world frame
    τ_pw = (skew([contact_w - pp3[1:2]; 0]) * [f_pw; 0])[3:3]
    τ_cw = (skew([contact_w - pc3[1:2]; 0]) * [f_cw; 0])[3:3]
    # overall wrench on both bodies in world frame
    # mapping the contact force into the generalized coordinates (at the centers of masses and in the world frame)
    wrench_p = [f_pw; τ_pw]
    wrench_c = [f_cw; τ_cw]

    # tangential velocities at the contact point
    tanvel_p = vp25[1:2] + (skew([pp3[1:2] - contact_w; 0]) * [zeros(2); vp25[3]])[1:2]
    tanvel_p = tanvel_p' * tangent_pw
    tanvel_c = vc25[1:2] + (skew([pc3[1:2] - contact_w; 0]) * [zeros(2); vc25[3]])[1:2]
    tanvel_c = tanvel_c' * tangent_cw
    tanvel = tanvel_p - tanvel_c

    # contact equality
    slackness = [
        sγ - ϕ;
        sψ - (friction_coefficient[1] * γ - [sum(β)]);
        sβ - ([+tanvel; -tanvel] + ψ[1]*ones(2));
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.index.slackness] .+= slackness
    e[pbody.index.optimality] .-= wrench_p
    e[cbody.index.optimality] .-= wrench_c
    return nothing
end

function residual!(e, x, θ, contact::SphereSphere, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    cbody = find_body(bodies, contact.child_name)
    residual!(e, x, θ, contact, pbody, cbody)
    return nothing
end
