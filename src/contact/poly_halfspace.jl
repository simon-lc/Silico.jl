################################################################################
# contact
################################################################################
struct PolyHalfSpace{T,D,NP,NC} <: Contact{T,D,NP,NC}
    name::Symbol
    parent_name::Symbol
    index::NodeIndices
    parent_shape::Shape{T,NP,D}
    child_shape::Shape{T,NC,D}
    friction_coefficient::Vector{T}
end

function PolyHalfSpace(parent_body::AbstractBody{T,D}, child_shape::Shape{T};
        parent_shape_id::Int=1,
        name::Symbol=:contact,
        friction_coefficient=0.2) where {T,D}

    parent_name = parent_body.name
    parent_shape = deepcopy(parent_body.shapes[parent_shape_id])
    child_shape = deepcopy(child_shape)

    index = NodeIndices()

    Np = constraint_dimension(parent_shape)
    Nc = constraint_dimension(child_shape)
    return PolyHalfSpace{T,D,Np,Nc}(
        name,
        parent_name,
        index,
        parent_shape,
        child_shape,
        [friction_coefficient],
        )
end

primal_dimension(contact::PolyHalfSpace{T,D}) where {T,D} = D # x
cone_dimension(contact::PolyHalfSpace{T,D,NP}) where {T,D,NP} = 1 + 1 + 2*(D-1) + NP # γ ψ β ν λp

function unpack_variables(x::Vector, contact::PolyHalfSpace{T,D,NP}) where {T,D,NP}
    off = 0
    nβ = (D - 1) * 2

    c = x[off .+ (1:D)]; off += D

    γ = x[off .+ (1:1)]; off += 1
    ψ = x[off .+ (1:1)]; off += 1
    β = x[off .+ (1:nβ)]; off += nβ
    λp = x[off .+ (1:NP)]; off += NP

    sγ = x[off .+ (1:1)]; off += 1
    sψ = x[off .+ (1:1)]; off += 1
    sβ = x[off .+ (1:nβ)]; off += nβ
    sp = x[off .+ (1:NP)]; off += NP
    return c, γ, ψ, β, λp, sγ, sψ, sβ, sp
end

function split_parameters(θ, contact::PolyHalfSpace{T}) where T
    friction_coefficient, parent_parameters, child_parameters = unpack_parameters(θ, contact)
    shape_p = contact.parent_shape
    shape_c = contact.child_shape
    Ap, bp, op = unpack_parameters(shape_p, parent_parameters)
    bop = bp + Ap * op
    normalc, offc = unpack_parameters(shape_c, child_parameters)
    return friction_coefficient, Ap, bop, normalc, offc
end

function residual!(e, x, θ, contact::PolyHalfSpace, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    residual!(e, x, θ, contact, pbody)
    return nothing
end

# min c' * n
# s.t. Ax <= b
function residual!(e, x, θ, contact::PolyHalfSpace{T,2,NP},
        pbody::AbstractBody) where {T,NP}

    # unpack parameters
    friction_coefficient, Ap, bop, normalc, offc =
        split_parameters(θ[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(θ[pbody.index.parameters], pbody)

    # unpack variables
    c, γ, ψ, β, λp, sγ, sψ, sβ, sp = unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    pp3 = pp2 + timestep_p[1] * vp25

    # contact position in the world frame
    contact_w = c + pp3[1:2]
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])

    # signed distance function
    ϕ = [(contact_w - offc)' * normalc]

    # contact normal and tangent in the world frame
    normal_pw = -x_2d_rotation(pp3[3:3]) * Ap' * λp
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
        normalc + x_2d_rotation(pp3[3:3]) * Ap' * λp;
    ]
    slackness = [
        sγ - ϕ;
        sψ - (friction_coefficient[1] * γ - [sum(β)]);
        sβ - ([+tanvel; -tanvel] + ψ[1]*ones(2));
        sp - (- Ap * contact_p + bop);
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.index.optimality] .+= optimality
    e[contact.index.slackness] .+= slackness
    e[pbody.index.optimality] .-= wrench_p
    return nothing
end

function residual!(e, x, θ, contact::PolyHalfSpace{T,3,NP},
        pbody::AbstractBody{T,3}) where {T,NP}

    # unpack parameters
    friction_coefficient, Ap, bop, normalc, offc =
        split_parameters(θ[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(θ[pbody.index.parameters], pbody)
    xp2 = pp2[1:3]
    qp2 = pp2[4:7]

    # unpack variables
    c, γ, ψ, β, λp, sγ, sψ, sβ, sp = unpack_variables(x[contact.index.variables], contact)
    vp25, ϕp25 = unpack_variables(x[pbody.index.variables], pbody)
    xp3 = xp2 + timestep_p[1] * vp25
    qp3 = quaternion_increment(qp2, timestep_p[1] * ϕp25)

    # contact position in the world frame
    contact_w = c + xp3
    # contact_p is expressed in pbody's frame
    contact_p = vector_rotate(dagger(qp3), contact_w - xp3)

    # signed distance function
    ϕ = [(contact_w - offc)' * normalc]

    # contact normal and tangent in the world frame
    normal_pw = normalc
    tangent_candidate = [1, 0, 0] # need to use a vector orthogonal to the normal at timestep 2
    tangent_pw1, tangent_pw2, wRp = tangential_plane(normal_pw, tangent_candidate=tangent_candidate)

    # force at the contact point in the contact frame
    f = [β[1:2] - β[3:4]; γ]
    # force at the contact point in the world frame
    f_pw = +wRp * f # parent
    # torques at the centers of masses in world frame
    τ_pw = skew(contact_w - xp3) * f_pw
    # overall wrench on both bodies in world frame
    # mapping the contact force into the generalized coordinates (at the centers of masses and in the world frame)
    angular_velocity_w = vector_rotate(qp2, 2ϕp25)
    torsional_velocity_w = angular_velocity_w' * normalc * normalc
    torsional_drag = -torsional_velocity_w .* friction_coefficient / 20
    wrench_p = [f_pw; vector_rotate(dagger(qp2), τ_pw + torsional_drag)] # we should apply Euler equation (rotational part) in the body frame, not the world frame

    # tangential velocities at the contact point
    tanvel_p = vp25 + skew(xp3 - contact_w) * angular_velocity_w # let's assume 2ϕp25 is the angular velocity
    tanvel_p = (1 - tanvel_p' * normal_pw) * tanvel_p
    tanvel_p = [tanvel_p' * tangent_pw1, tanvel_p' * tangent_pw2]
    tanvel = tanvel_p

    # contact equality
    optimality = [
        normalc + vector_rotate(qp3, Ap' * λp);
    ]
    slackness = [
        sγ - ϕ;
        sψ - (friction_coefficient[1] * γ - [sum(β)]);
        sβ - ([+tanvel; -tanvel] + ψ[1]*ones(4));
        sp - (- Ap * contact_p + bop);
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.index.optimality] .+= optimality
    e[contact.index.slackness] .+= slackness
    e[pbody.index.optimality] .-= wrench_p
    return nothing
end
