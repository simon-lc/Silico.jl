################################################################################
# contact
################################################################################
abstract type BilevelContact{T,D,NP,NC} <: Node{T} end
space_dimension(contact::BilevelContact{T,D}) where {T,D} = D

struct BilevelContact2D{T,D,NP,NC} <: BilevelContact{T,D,NP,NC}
    name::Symbol
    parent_name::Symbol
    child_name::Symbol
    index::NodeIndices
    parent_shape::Shape{T}
    child_shape::Shape{T}
    friction_coefficient::Vector{T}
    detector::CollisionDetector{T,D,NP,NC}
end

struct EnvBilevelContact2D{T,D,NP,NC} <: BilevelContact{T,D,NP,NC}
    name::Symbol
    parent_name::Symbol
    index::NodeIndices
    parent_shape::Shape{T}
    child_shape::Shape{T}
    friction_coefficient::Vector{T}
    detector::CollisionDetector{T,D,NP,NC}
end

function BilevelContact2D(parent_body::AbstractBody{T}, child_body::AbstractBody{T};
        parent_shape_id::Int=1,
        child_shape_id::Int=1,
        name::Symbol=:contact,
        complementarity_tolerance=1e-4,
        friction_coefficient=0.2) where {T}

    parent_name = parent_body.name
    child_name = child_body.name
    parent_shape = deepcopy(parent_body.shapes[parent_shape_id])
    child_shape = deepcopy(child_body.shapes[child_shape_id])

    detector = CollisionDetector(parent_shape, child_shape;
        name=name,
        complementarity_tolerance=complementarity_tolerance)
    index = NodeIndices()

    D = 2
    Np = constraint_dimension(parent_shape)
    Nc = constraint_dimension(child_shape)
    return BilevelContact2D{T,D,Np,Nc}(
        name,
        parent_name,
        child_name,
        index,
        parent_shape,
        child_shape,
        [friction_coefficient],
        detector,
        )
end

function EnvBilevelContact2D(parent_body::AbstractBody{T}, child_shape::Shape{T};
        parent_shape_id::Int=1,
        name::Symbol=:contact,
        complementarity_tolerance=1e-4,
        friction_coefficient=0.2) where {T}

    parent_name = parent_body.name
    parent_shape = deepcopy(parent_body.shapes[parent_shape_id])
    child_shape = deepcopy(child_shape)

    index = NodeIndices()
    detector = CollisionDetector(parent_shape, child_shape;
        name=name,
        complementarity_tolerance=complementarity_tolerance)

    D = 2
    Np = constraint_dimension(parent_shape)
    Nc = constraint_dimension(child_shape)
    return EnvBilevelContact2D{T,D,Np,Nc}(
        name,
        parent_name,
        index,
        parent_shape,
        child_shape,
        [friction_coefficient],
        detector,
        )
end

primal_dimension(contact::BilevelContact{T,D}) where {T,D} = 0
cone_dimension(contact::BilevelContact) = 1 + 1 + 2 # ??, ??, ??
parameter_dimension(contact::BilevelContact) = 1 # friction_coefficient

function unpack_variables(x::Vector, contact::BilevelContact{T,D,NP,NC}) where {T,D,NP,NC}
    off = 0
    ?? = x[off .+ (1:1)]; off += 1
    ?? = x[off .+ (1:1)]; off += 1
    ?? = x[off .+ (1:2)]; off += 2
    s?? = x[off .+ (1:1)]; off += 1
    s?? = x[off .+ (1:1)]; off += 1
    s?? = x[off .+ (1:2)]; off += 2
    return ??, ??, ??, s??, s??, s??
end

function get_parameters(contact::BilevelContact{T,D}) where {T,D}
    ?? = [contact.friction_coefficient;]
    return ??
end

function set_parameters!(contact::BilevelContact{T,D,NP,NC}, ??) where {T,D,NP,NC}
    friction_coefficient = unpack_parameters(??, contact)
    contact.friction_coefficient .= friction_coefficient
    return nothing
end

function unpack_parameters(??::Vector, contact::BilevelContact{T,D,NP,NC}) where {T,D,NP,NC}
    @assert D == 2
    off = 0
    friction_coefficient = ??[off .+ (1:1)]; off += 1
    return friction_coefficient
end

function residual!(e, x, ??, contact::BilevelContact2D{T,D,NP,NC},
        pbody::AbstractBody, cbody::AbstractBody) where {T,D,NP,NC}

    # unpack parameters
    # friction_coefficient, parent_parameters, child_parameters =
    friction_coefficient =
        unpack_parameters(??[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(??[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(??[cbody.index.parameters], cbody)

    # unpack variables
    ??, ??, ??, s??, s??, s?? =
        unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    vc25 = unpack_variables(x[cbody.index.variables], cbody)
    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25

    ??, contact_w, normal_pw, normal_cw = contact_data(pp3, pc3, contact.detector)
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw
    tangent_cw = R * normal_cw

    # rotation matrix from contact frame to world frame
    wRp = [tangent_pw normal_pw] # n points towards the parent body, [t,n,z] forms an oriented vector basis
    wRc = [tangent_cw normal_cw] # n points towards the parent body, [t,n,z] forms an oriented vector basis

    # force at the contact point in the contact frame
    f = [??[1] - ??[2]; ??]
    # force at the contact point in the world frame
    f_pw = +wRp * f # parent
    f_cw = -wRc * f # child
    # torques at the centers of masses in world frame
    ??_pw = (skew([contact_w - pp3[1:2]; 0]) * [f_pw; 0])[3:3]
    ??_cw = (skew([contact_w - pc3[1:2]; 0]) * [f_cw; 0])[3:3]
    # overall wrench on both bodies in world frame
    # mapping the contact force into the generalized coordinates (at the centers of masses and in the world frame)
    wrench_p = [f_pw; ??_pw]
    wrench_c = [f_cw; ??_cw]

    # tangential velocities at the contact point
    tanvel_p = vp25[1:2] + (skew([pp3[1:2] - contact_w; 0]) * [zeros(2); vp25[3]])[1:2]
    tanvel_p = tanvel_p' * tangent_pw
    tanvel_c = vc25[1:2] + (skew([pc3[1:2] - contact_w; 0]) * [zeros(2); vc25[3]])[1:2]
    tanvel_c = tanvel_c' * tangent_cw
    tanvel = tanvel_p - tanvel_c

    # contact equality
    slackness = [
        s?? - ??;
        s?? - (friction_coefficient[1] * ?? - [sum(??)]);
        s?? - ([+tanvel; -tanvel] + ??[1]*ones(2));
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.index.slackness] .+= slackness
    e[pbody.index.optimality] .-= wrench_p
    e[cbody.index.optimality] .-= wrench_c
    return nothing
end

function residual!(e, x, ??, contact::EnvBilevelContact2D{T,D,NP},
        pbody::AbstractBody) where {T,D,NP}

    # unpack parameters
    friction_coefficient =
        unpack_parameters(??[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(??[pbody.index.parameters], pbody)

    # unpack variables
    ??, ??, ??, s??, s??, s?? =
        unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    pp3 = pp2 + timestep_p[1] * vp25

    ??, contact_w, normal_pw, normal_cw = contact_data(pp3, zeros(3), contact.detector)
    normal_pw = [0.0, 1.0]
    normal_cw = [0.0, 1.0]
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw

    # rotation matrix from contact frame to world frame
    wRp = [tangent_pw normal_pw] # n points towards the parent body, [t,n,z] forms an oriented vector basis

    # force at the contact point in the contact frame
    f = [??[1] - ??[2]; ??]
    # force at the contact point in the world frame
    f_pw = +wRp * f # parent
    # torques at the centers of masses in world frame
    ??_pw = (skew([contact_w - pp3[1:2]; 0]) * [f_pw; 0])[3:3]
    # overall wrench on both bodies in world frame
    # mapping the contact force into the generalized coordinates (at the centers of masses and in the world frame)
    wrench_p = [f_pw; ??_pw]

    # tangential velocities at the contact point
    tanvel_p = vp25[1:2] + (skew([pp3[1:2] - contact_w; 0]) * [zeros(2); vp25[3]])[1:2]
    tanvel_p = tanvel_p' * tangent_pw
    tanvel = tanvel_p

    slackness = [
        s?? - ??;
        s?? - (friction_coefficient[1] * ?? - [sum(??)]);
        s?? - ([+tanvel; -tanvel] + ??[1]*ones(2));
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.index.slackness] .+= slackness
    e[pbody.index.optimality] .-= wrench_p
    return nothing
end

function residual!(e, x, ??, contact::BilevelContact2D, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    cbody = find_body(bodies, contact.child_name)
    residual!(e, x, ??, contact, pbody, cbody)
    return nothing
end

function residual!(e, x, ??, contact::EnvBilevelContact2D, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    residual!(e, x, ??, contact, pbody)
    return nothing
end
