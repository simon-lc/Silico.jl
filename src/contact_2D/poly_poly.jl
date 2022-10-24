################################################################################
# contact
################################################################################
struct PolyPoly{T,D,NP,NC} <: Contact{T,D,NP,NC}
    name::Symbol
    parent_name::Symbol
    child_name::Symbol
    index::NodeIndices
    parent_shape::Shape{T,NP,D}
    child_shape::Shape{T,NC,D}
    friction_coefficient::Vector{T}
end

function PolyPoly(parent_body::AbstractBody{T,D}, child_body::AbstractBody{T};
        parent_shape_id::Int=1,
        child_shape_id::Int=1,
        name::Symbol=:contact,
        friction_coefficient=0.2) where {T,D}

    parent_name = parent_body.name
    child_name = child_body.name
    parent_shape = deepcopy(parent_body.shapes[parent_shape_id])
    child_shape = deepcopy(child_body.shapes[child_shape_id])

    index = NodeIndices()

    Np = constraint_dimension(parent_shape)
    Nc = constraint_dimension(child_shape)
    return PolyPoly{T,D,Np,Nc}(
        name,
        parent_name,
        child_name,
        index,
        parent_shape,
        child_shape,
        [friction_coefficient],
        )
end

primal_dimension(contact::PolyPoly{T,D}) where {T,D} = D + 1 # x, ϕ
cone_dimension(contact::PolyPoly{T,D,NP,NC}) where {T,D,NP,NC} = 1 + 1 + 2 + NP + NC # γ ψ β λp, λc

function unpack_variables(x::Vector, contact::PolyPoly{T,D,NP,NC}) where {T,D,NP,NC}
    num_cone = cone_dimension(contact)
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

function residual!(e, x, θ, contact::PolyPoly{T,D,NP,NC},
        pbody::AbstractBody, cbody::AbstractBody) where {T,D,NP,NC}

    # unpack parameters
    friction_coefficient, parent_parameters, child_parameters =
        unpack_parameters(θ[contact.index.parameters], contact)
    shape_p = contact.parent_shape
    shape_c = contact.child_shape
    Ap, bp, op = unpack_parameters(shape_p, parent_parameters)
    Ac, bc, oc = unpack_parameters(shape_c, child_parameters)
    bop = bp + Ap * op
    boc = bc + Ac * oc

    pp2, timestep_p = unpack_pose_timestep(θ[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(θ[cbody.index.parameters], cbody)

    # unpack variables
    c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc = unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    vc25 = unpack_variables(x[cbody.index.variables], cbody)
    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25

    # contact position in the world frame
    contact_w = c + (pp3 + pc3)[1:2] / 2
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])
    # contact_c is expressed in cbody's frame
    contact_c = x_2d_rotation(pc3[3:3])' * (contact_w - pc3[1:2])

    # contact normal and tangent in the world frame
    normal_pw = -x_2d_rotation(pp3[3:3]) * Ap' * λp
    normal_cw = +x_2d_rotation(pc3[3:3]) * Ac' * λc
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw
    tangent_cw = R * normal_cw

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
    optimality = [
        x_2d_rotation(pp3[3:3]) * Ap' * λp + x_2d_rotation(pc3[3:3]) * Ac' * λc;
        1 - sum(λp) - sum(λc)
    ]
    slackness = [
        sγ - ϕ;
        sψ - (friction_coefficient[1] * γ - [sum(β)]);
        sβ - ([+tanvel; -tanvel] + ψ[1]*ones(2));
        sp - (- Ap * contact_p + bop + ϕ .* ones(NP));
        sc - (- Ac * contact_c + boc + ϕ .* ones(NC));
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.index.optimality] .+= optimality
    e[contact.index.slackness] .+= slackness
    e[pbody.index.optimality] .-= wrench_p
    e[cbody.index.optimality] .-= wrench_c
    return nothing
end
