################################################################################
# contact
################################################################################
struct SphereSphere{T,D,NP,NC} <: Contact{T,D,NP,NC}
    name::Symbol
    parent_name::Symbol
    child_name::Symbol
    index::NodeIndices
    parent_shape::Shape{T}
    child_shape::Shape{T}
    friction_coefficient::Vector{T}
end

function SphereSphere(parent_body::AbstractBody{T,D}, child_body::AbstractBody{T,D};
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
    return SphereSphere{T,D,Np,Nc}(
        name,
        parent_name,
        child_name,
        index,
        parent_shape,
        child_shape,
        [friction_coefficient],
        )
end

primal_dimension(contact::SphereSphere) = 0
cone_dimension(contact::SphereSphere{T,D}) where {T,D} = 1 + 1 + 2*(D-1) # γ ψ β

function unpack_variables(x::Vector, contact::SphereSphere{T,D}) where {T,D}
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

function split_parameters(θ, contact::SphereSphere{T}) where T
    friction_coefficient, parent_parameters, child_parameters = unpack_parameters(θ, contact)
    shape_p = contact.parent_shape
    shape_c = contact.child_shape
    radp, offp = unpack_parameters(shape_p, parent_parameters)
    radc, offc = unpack_parameters(shape_c, child_parameters)
    return friction_coefficient, radp, offp, radc, offc
end

function residual!(e, x, θ, contact::SphereSphere{T,2},
        pbody::AbstractBody{T,2}, cbody::AbstractBody{T,2}) where T

    # unpack parameters
    friction_coefficient, radp, offp, radc, offc = split_parameters(θ[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(θ[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(θ[cbody.index.parameters], cbody)

    # unpack variables
    γ, ψ, β, sγ, sψ, sβ = unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    vc25 = unpack_variables(x[cbody.index.variables], cbody)
    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25
    # signed distance function
    ϕ = [norm(pp3[1:2] + offp - pc3[1:2] - offc)] - radp - radc

    # contact normal and tangent in the world frame
    normal_pw = (pp3[1:2] + offp - pc3[1:2] - offc)
    normal_cw = (pp3[1:2] + offp - pc3[1:2] - offc)
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw
    tangent_cw = R * normal_cw

    # contact position in the world frame
    n = normal_pw / (1e-6 + norm(normal_pw))
    contact_w = 0.5 * (pp3[1:2] + offp + radp[1] * n + pc3[1:2] + offc - radc[1] * n)

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

function residual!(e, x, θ, contact::SphereSphere{T,3},
        pbody::AbstractBody{T,3}, cbody::AbstractBody{T,3}) where T

    # unpack parameters
    friction_coefficient, radp, offp, radc, offc = split_parameters(θ[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(θ[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(θ[cbody.index.parameters], cbody)
    xp2 = pp2[1:3]
    qp2 = pp2[4:7]
    xc2 = pc2[1:3]
    qc2 = pc2[4:7]
    # unpack variables
    γ, ψ, β, sγ, sψ, sβ = unpack_variables(x[contact.index.variables], contact)
    vp25, ϕp25 = unpack_variables(x[pbody.index.variables], pbody)
    vc25, ϕc25 = unpack_variables(x[cbody.index.variables], cbody)
    xp3 = xp2 + timestep_p[1] * vp25
    xc3 = xc2 + timestep_c[1] * vc25
    # qp3 = quaternion_increment(qp2, timestep_p[1] * ϕp25)
    # qc3 = quaternion_increment(qc2, timestep_c[1] * ϕc25)


    # analytical contact position in the world frame
    # we are missing offpw and offcw
    normal_w = (xp3 - xc3)
    normal_w ./= norm(xp3 - xc3) + 1e-10
    contact_w = 0.5 * (xp3 - radp[1] .* normal_w) + 0.5 * (xc3 + radc[1] .* normal_w)

    # analytical signed distance function
    ϕ = norm(xp3 - xc3) .- radp - radc

    # contact normal and tangent in the world frame
    normal_pw = normal_w
    normal_cw = normal_w
    tangent_candidate = [1, 0, 0] # need to use a vector orthogonal to the normal at timestep 2
    tangent_pw1, tangent_pw2, wRp = tangential_plane(normal_pw, tangent_candidate=tangent_candidate)
    tangent_cw1, tangent_cw2, wRc = tangential_plane(normal_cw, tangent_candidate=tangent_candidate)

    # force at the contact point in the contact frame
    f = [β[1:2] - β[3:4]; γ]
    # force at the contact point in the world frame
    f_pw = +wRp * f # parent
    f_cw = -wRc * f # child
    # torques at the centers of masses in world frame
    τ_pw = skew(contact_w - xp3) * f_pw
    τ_cw = skew(contact_w - xc3) * f_cw
    # overall wrench on both bodies in world frame
    # mapping the contact force into the generalized coordinates (at the centers of masses and in the world frame)
    wrench_p = [f_pw; τ_pw] # we should apply Euler equation (rotational part) in the body frame, not the world frame
    wrench_c = [f_cw; τ_cw] # we should apply Euler equation (rotational part) in the body frame, not the world frame

    # tangential velocities at the contact point
    tanvel_p = vp25 + skew(xp3 - contact_w) * 2ϕp25 # let's assume 2ϕp25 is the angular velocity
    tanvel_c = vc25 + skew(xc3 - contact_w) * 2ϕc25 # let's assume 2ϕc25 is the angular velocity
    tanvel_p = (1 - tanvel_p' * normal_pw) * tanvel_p
    tanvel_c = (1 - tanvel_c' * normal_cw) * tanvel_c
    tanvel_p = [tanvel_p' * tangent_pw1, tanvel_p' * tangent_pw2]
    tanvel_c = [tanvel_c' * tangent_cw1, tanvel_c' * tangent_cw2]
    tanvel = tanvel_p - tanvel_c

    # contact equality
    slackness = [
        sγ - ϕ;
        sψ - (friction_coefficient[1] * γ - [sum(β)]);
        sβ - ([+tanvel; -tanvel] + ψ[1]*ones(4));
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.index.slackness] .+= slackness
    e[pbody.index.optimality] .-= wrench_p
    e[cbody.index.optimality] .-= wrench_c
    return nothing
end
