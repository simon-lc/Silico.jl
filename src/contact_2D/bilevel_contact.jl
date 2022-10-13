################################################################################
# contact
################################################################################
abstract type BilevelContact{T,D,NP,NC} <: Node{T} end

struct BilevelContact2D30{T,D,NP,NC} <: BilevelContact{T,D,NP,NC}
    name::Symbol
    parent_name::Symbol
    child_name::Symbol
    index::NodeIndices
    parent_shape::Shape{T}
    child_shape::Shape{T}
    friction_coefficient::Vector{T}
    detector::CollisionDetector{T,D,NP,NC}
end

struct EnvBilevelContact2D30{T,D,NP,NC} <: BilevelContact{T,D,NP,NC}
    name::Symbol
    parent_name::Symbol
    index::NodeIndices
    parent_shape::Shape{T}
    child_shape::Shape{T}
    friction_coefficient::Vector{T}
    detector::CollisionDetector{T,D,NC,NP}
end

function BilevelContact2D30(parent_body::AbstractBody{T}, child_body::AbstractBody{T};
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
    return BilevelContact2D30{T,D,Np,Nc}(
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

function EnvBilevelContact2D30(parent_body::AbstractBody{T}, child_shape::Shape{T};
        parent_shape_id::Int=1,
        name::Symbol=:contact,
        friction_coefficient=0.2) where {T}

    parent_name = parent_body.name
    parent_shape = deepcopy(parent_body.shapes[parent_shape_id])
    child_shape = deepcopy(child_shape)

    index = NodeIndices()

    D = 2
    Np = constraint_dimension(parent_shape)
    Nc = constraint_dimension(child_shape)
    return EnvBilevelContact2D30{T,D,Np,Nc}(
        name,
        parent_name,
        index,
        parent_shape,
        child_shape,
        [friction_coefficient],
        )
end

primal_dimension(contact::BilevelContact{T,D}) where {T,D} = 0
cone_dimension(contact::BilevelContact) = 1 + 1 + 2 # γ, ψ, β
parameter_dimension(contact::BilevelContact) = 1 # friction_coefficient
function unpack_variables(x::Vector, contact::BilevelContact{T,D,NP,NC}) where {T,D,NP,NC}
    off = 0
    γ = x[off .+ (1:1)]; off += 1
    ψ = x[off .+ (1:1)]; off += 1
    β = x[off .+ (1:2)]; off += 2
    sγ = x[off .+ (1:1)]; off += 1
    sψ = x[off .+ (1:1)]; off += 1
    sβ = x[off .+ (1:2)]; off += 2
    return γ, ψ, β, sγ, sψ, sβ
end

function get_parameters(contact::BilevelContact{T,D}) where {T,D}
    θ = [contact.friction_coefficient;]
    return θ
end

function set_parameters!(contact::BilevelContact{T,D,NP,NC}, θ) where {T,D,NP,NC}
    friction_coefficient = unpack_parameters(θ, contact)
    contact.friction_coefficient .= friction_coefficient
    return nothing
end

function unpack_parameters(θ::Vector, contact::BilevelContact{T,D,NP,NC}) where {T,D,NP,NC}
    @assert D == 2
    off = 0
    friction_coefficient = θ[off .+ (1:1)]; off += 1
    return friction_coefficient
end

function residual!(e, x, θ, contact::BilevelContact2D30{T,D,NP,NC},
        pbody::AbstractBody, cbody::AbstractBody) where {T,D,NP,NC}

    # unpack parameters
    # friction_coefficient, parent_parameters, child_parameters =
    friction_coefficient =
        unpack_parameters(θ[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(θ[pbody.index.parameters], pbody)
    pc2, timestep_c = unpack_pose_timestep(θ[cbody.index.parameters], cbody)

    # unpack variables
    γ, ψ, β, sγ, sψ, sβ =
        unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    vc25 = unpack_variables(x[cbody.index.variables], cbody)
    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25

    ϕ, contact_w, normal_pw, normal_cw = contact_data(pp3, pc3, contact.detector)

    # contact position in the world frame
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])
    # contact_c is expressed in cbody's frame
    contact_c = x_2d_rotation(pc3[3:3])' * (contact_w - pc3[1:2])

    # contact normal and tangent in the world frame
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

function residual!(e, x, θ, contact::EnvBilevelContact2D30{T,D,NP},
        pbody::AbstractBody) where {T,D,NP}

    # unpack parameters
    friction_coefficient, parent_parameters, child_parameters =
        unpack_parameters(θ[contact.index.parameters], contact)
    pp2, timestep_p = unpack_pose_timestep(θ[pbody.index.parameters], pbody)

    # unpack variables
    c, α, βp, βc, γ, ψ, β, λα, λp, λc, sγ, sψ, sβ, sα, sp, sc =
        unpack_variables(x[contact.index.variables], contact)
    vp25 = unpack_variables(x[pbody.index.variables], pbody)
    pp3 = pp2 + timestep_p[1] * vp25

    #signed distance function
    ϕ = α[1] - 1.0

    # contact position in the world frame
    contact_w = c + pp3[1:2]
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])
    # contact_c is expressed in cbody's frame
    contact_c = contact_w

    # constraints
    shape_p = contact.parent_shape
    shape_c = contact.child_shape
    gp = constraint(shape_p, contact_p, α, βp) # positive
    gc = constraint(shape_c, contact_c, α, βc) # positive
    ∇α_gp = constraint_jacobian_α(shape_p, contact_p, α, βp)
    ∇α_gc = constraint_jacobian_α(shape_c, contact_c, α, βc)
    ∇p_gp = constraint_jacobian_p(shape_p, contact_p, α, βp)
    ∇p_gc = constraint_jacobian_p(shape_c, contact_c, α, βc)
    ∇β_gp = constraint_jacobian_β(shape_p, contact_p, α, βp)
    ∇β_gc = constraint_jacobian_β(shape_c, contact_c, α, βc)
    ∇o_gp = constraint_jacobian_o(shape_p, contact_p, α, βp)
    ∇o_gc = constraint_jacobian_o(shape_c, contact_c, α, βc)

    # contact normal and tangent in the world frame
    normal_pw = -x_2d_rotation(pp3[3:3]) * ∇o_gp' * λp
    # normal_pw = -x_2d_rotation(pp3[3:3]) * ∇o_gc' * λc
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

    optimality = [
        x_2d_rotation(pp3[3:3]) * ∇p_gp' * λp + ∇p_gc' * λc;
        1 .- ∇α_gp' * λp - ∇α_gc' * λc;
        ∇β_gp' * λp;
        ∇β_gc' * λc;
    ]

    slackness = [
        sγ - [ϕ];
        sψ - (friction_coefficient[1] * γ - [sum(β)]);
        sβ - ([+tanvel; -tanvel] + ψ[1]*ones(2));
        sα - α;
        sp - gp;
        sc - gc;
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.index.optimality] .+= optimality
    e[contact.index.slackness] .+= slackness
    e[pbody.index.optimality] .-= wrench_p
    return nothing
end

function residual!(e, x, θ, contact::BilevelContact2D30, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    cbody = find_body(bodies, contact.child_name)
    residual!(e, x, θ, contact, pbody, cbody)
    return nothing
end

function residual!(e, x, θ, contact::EnvBilevelContact2D30, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    # cbody = find_body(bodies, contact.child_name)
    residual!(e, x, θ, contact, pbody)
    return nothing
end

function get_bilevel_polytope_collision(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
    A=[
    [
        +1.0 -0.0;
        +0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        ],
    [
        +1.0 +0.0;
        +0.0 +1.0;
        -1.0 +0.0;
        +0.0 -1.0;
        ],
    ],
    b=[
        0.5*[+1.0, +1.0, +1.0, +1.0],
        0.5*[+1.0, +1.0, +1.0, +1.0],
    ],
    method_type::Symbol=:finite_difference,
    options=Mehrotra.Options(
        # verbose=false,
        complementarity_tolerance=1e-4,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=false,
        warm_start=true,
        )
    )

    N = length(A)
    # nodes
    shapes = [PolytopeShape(A[i], b[i]) for i=1:N]
    floor_shape = HalfspaceShape([0.0, 1.0])

    bodies = [Body(timestep, mass, inertia, shapes[i:i],
        gravity=+gravity, name=Symbol(:body_, i)) for i=1:N]

    body_contacts = vcat([
        [BilevelContact2D30(bodies[i], bodies[j],
            friction_coefficient=friction_coefficient,
            name=Symbol(:contact_, i, :_, j),
            complementarity_tolerance=options.complementarity_tolerance) for i=1:j-1]
        for j=1:N]...)
    env_contacts = [
        EnvContact2D(bodies[i], floor_shape,
            friction_coefficient=friction_coefficient,
            name=Symbol(:env_contact_, i)) for i=1:N]

    contacts = [body_contacts; env_contacts]
    # contacts = [body_contacts;]
    indexing!([bodies; contacts])

    local_mechanism_residual(primals, duals, slacks, parameters) =
        mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)

    mechanism = Mechanism(
        local_mechanism_residual,
        bodies,
        contacts,
        options=options,
        method_type=method_type)

    Mehrotra.initialize_solver!(mechanism.solver)
    return mechanism
end


# vis = Visualizer()
# open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

################################################################################
# demo
################################################################################
A0 = [
    [
        +1.0 -0.0;
        -0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        ],
    [
        +1.0 +0.0;
        +0.0 +1.0;
        -1.0 +0.0;
        +0.0 -1.0;
        ],
    ]
b0 = [
        0.50*[+1.0, +1.0, +1.0, +1.0],
        0.35*[+1.0, +1.0, +1.0, +1.0],# +1.0],
    ]

mech = get_bilevel_polytope_collision(;
# mech = get_polytope_collision(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.00,
    # method_type=:symbolic,
    method_type=:finite_difference,
    A=A0, b=b0,
    options=Mehrotra.Options(
        verbose=true,
        complementarity_tolerance=1e-6,
        compressed_search_direction=false,
        # compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        warm_start=false,
        complementarity_correction=0.5,
        # complementarity_backstep=1e-1,
        )
    )
# Mehrotra.solve!(mech.solver)

################################################################################
# test simulation
################################################################################
xp2 = [+0.25, +0.60, -0.0]
xc2 = [-0.00, +1.50, -0.1]
vp15 = [-0.0, +0.0, -0.0]
vc15 = [+0.0, -0.0, +0.0]
z0 = [xp2; vp15; xc2; vc15]

u0 = zeros(6)
H0 = 20

@elapsed storage = simulate!(mech, z0, H0)

################################################################################
# visualization
################################################################################
visualize!(vis, mech, storage, build=true)

using Plots
storage.x
plot(hcat([storage.x[i][1] for i=1:H0]...)')
plot(hcat([storage.x[i][2] for i=1:H0]...)')
plot(hcat([storage.v[i][2] for i=1:H0]...)')
scatter(storage.iterations)
# plot!(hcat(storage.variables...)')
# RobotVisualizer.convert_frames_to_video_and_gif("sphere_polytope_drop")




function bilevel_methods(equality::Function, equality_jacobian::Function,
        dim::Mehrotra.Dimensions, idx::Mehrotra.Indices)
    parameter_keywords = idx.parameter_keywords

    # in-place evaluation
    function equality_constraint(out, x, θ)
        primals = x[idx.primals]
        duals = x[idx.duals]
        slacks = x[idx.slacks]
        parameters = θ
        out .= equality(primals, duals, slacks, parameters)
    end

    warning = "compressed search direction is not implemented with finite difference methods."
    function equality_constraint_compressed(out, x, θ)
        @warn warning
    end

    # jacobian variables
    function equality_jacobian_variables(vector_cache, x, θ)
        # f(out, x) = equality_constraint(out, x, θ)
        # matrix_cache = reshape(vector_cache, (dim.equality, dim.variables))
        # FiniteDiff.finite_difference_jacobian!(matrix_cache, f, x)
        # matrix_cache[idx.primals, idx.primals] .+= 1e-3*Diagonal(ones(dim.primals))
        # matrix_cache[idx.duals, idx.duals] .-= 1e-3*Diagonal(ones(dim.duals))
        # return nothing

        matrix_cache = reshape(vector_cache, (dim.equality, dim.variables))
        # equality_jacobian(matrix_cache, primals, duals, slacks, parameters)
        matrix_cache[idx.primals, idx.primals] .+= 1e-3*Diagonal(ones(dim.primals))
        matrix_cache[idx.duals, idx.duals] .-= 1e-3*Diagonal(ones(dim.duals))
        return nothing
    end

    # jacobian variables compressed
    function equality_jacobian_variables_compressed(vector_cache, x, θ)
        error(warning)
    end

    # correction
    function correction(c, r, Δza, Δsa, κ)
        c .= r
        c[idx.cone_product] .-= (κ - Mehrotra.cone_product(Δza, Δsa, idx.cone_nonnegative, idx.cone_second_order))
        return nothing
    end

    # correction compressed
    function correction_compressed(cc, r, Δza, Δsa, x, κ)
        error(warning)
    end

    # slack direction
    function slack_direction(Δs, Δz, x, rs)
        error(warning)
    end

    # jacobian parameters
    function equality_jacobian_parameters(vector_cache, x, θ)
        error(warning)
    end

    equality_jacobian_keywords = Vector{Function}()
    for k in eachindex(parameter_keywords)
        function func(vector_cache, x, θ)
            error(warning)
        end
        push!(equality_jacobian_keywords, func)
    end

    ex_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.variables)))[1:2]...]...))
    exc_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.equality)))[1:2]...]...))
    eθ_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.parameters)))[1:2]...]...))
    eθ_indices = zeros(Int, dim.equality, dim.parameters)
    vec(eθ_indices) .= 1:length(eθ_indices)
    ek_indices = Vector{Vector{Int}}()
    for (key, val) in parameter_keywords
        indices = vec(eθ_indices[:,val])
        push!(ek_indices, indices)
    end

    methods = Mehrotra.ProblemMethods(
        equality_constraint,
        equality_constraint_compressed,
        equality_jacobian_variables,
        equality_jacobian_variables_compressed,
        equality_jacobian_parameters,
        equality_jacobian_keywords,
        correction,
        correction_compressed,
        slack_direction,
        zeros(length(ex_sparsity)),
        zeros(length(exc_sparsity)),
        zeros(length(eθ_sparsity)),
        ex_sparsity,
        exc_sparsity,
        eθ_sparsity,
        ek_indices,
    )

    return methods
end


using SparseArrays


num_primals = mech.solver.dimensions.primals
num_cone = mech.solver.dimensions.cone
parameters = mech.solver.parameters
options = mech.solver.options

bodies = mech.bodies
contacts = mech.contacts

# dimensions
dim = mech.solver.dimensions

# indices
idx = mech.solver.indices

equality(primals, duals, slacks, parameters) = mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)
Mehrotra.finite_difference_methods(equality, dim, idx)

equality_jacobian = x -> x
custom_methods = bilevel_methods(equality, equality_jacobian, dim, idx)
custom_solver = Mehrotra.Solver(
        nothing,
        num_primals,
        num_cone,
        parameters=parameters,
        nonnegative_indices=collect(1:num_cone),
        second_order_indices=[collect(1:0)],
        methods=custom_methods,
        options=options
        )

custom_mech = Mechanism(nothing, bodies, contacts;
        options=options,
        methods=custom_methods,
        )

Mehrotra.solve!(custom_solver)
