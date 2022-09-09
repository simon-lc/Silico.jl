################################################################################
# methods
################################################################################
function simulate!(context::Context, state::Vector, H::Int)
    mechanism = context.mechanism
    nu = mechanism.dimensions.input
    u = zeros(nu)

    measurements = Vector{Measurement}()
    for i = 1:H
        dynamics(state, mechanism, state, u, nothing)
        measurement = measure(context, state)
        push!(measurements, measurement)
    end
    return measurements
end

function measure(context::CvxContext1320, state::Vector)
    mechanism = context.mechanism
    z = deepcopy(state)
    d = get_point_cloud(mechanism, context.cameras, state)
    measurement = CvxMeasurement1320(z, d)
    return measurement
end

function get_point_cloud(mechanism::Mechanism1170, cameras::Vector{Camera1320{T}}, z) where T
    d = Vector{Matrix{T}}()
    for camera in cameras
        nr = length(camera.camera_rays)
        di = zeros(T, 2, nr)
        A, b, o = get_halfspaces(mechanism, z)
        A, b, o = add_floor(A, b, o)
        julia_point_cloud!(di, camera.eye_position, camera.camera_rays, camera.softness, A, b, o)
        push!(d, di) # TODO maybe this is not copyng the data correctly
    end
    return d
end

function get_halfspaces(mechanism::Mechanism1170, z::Vector{T}) where T
    A = Vector{Matrix{T}}()
    b = Vector{Vector{T}}()
    o = Vector{Vector{T}}()

    set_current_state!(mechanism, z)
    for body in mechanism.bodies
        for shape in body.shapes
            Ai, bi, oi = halfspace_transformation(body.pose, shape.A, shape.b, shape.o)
            push!(A, Ai)
            push!(b, bi)
            push!(o, oi)
        end
    end
    return A, b, o
end

function set_halfspaces!(mechanism::Mechanism1170,
        A::Vector{<:Matrix}, b::Vector{<:Vector}, o::Vector{<:Vector}) where T

    bodies = mechanism.bodies
    contacts = mechanism.contacts

    i = 0
    for body in bodies
        for shape in body.shapes
            i += 1
            shape.A .= A[i]
            shape.b .= b[i]
            shape.o .= o[i]
            contacts[i].A_parent_collider .= A[i]
            contacts[i].b_parent_collider .= b[i] + A[i] * o[i]
        end
    end
    return nothing
end

# A = [ones(4,2)]
# b = [ones(4)]
# o = [ones(2)]
# set_halfspaces!(mech, A, b, o)
# z = zeros(mech.dimensions.state)
# A1, b1, o1 = get_halfspaces(mech, z)
# norm.(A1 - A)[1]
# norm.(b1 - b)[1]
# norm.(o1 - o)[1]



function process_model(context::CvxContext1320, state::Vector, params::Vector)
    set_parameters!(context, params)

    mechanism = context.mechanism
    nz = mechanism.dimensions.state
    nu = mechanism.dimensions.input
    nθ = mechanism.dimensions.parameters
    num_parameters = length(params)

    input = zeros(nu)
    next_state = zeros(nz)
    # jacobian_state = zeros(nz, nz)
    jacobian_parameters = zeros(nz, nθ)

    dynamics(next_state, mechanism, state, input, nothing)
    # dynamics_jacobian_state(jacobian_state, mechanism, state, input, nothing)
    dynamics_jacobian_parameters(jacobian_parameters, mechanism, state, input, nothing)

    function parameters_mapping(context::CvxContext1320, params::Vector)
        set_parameters!(context, params)
        return context.mechanism.parameters
    end
    parameters_jacobian = FiniteDiff.finite_difference_jacobian(params -> parameters_mapping(context, params), params)

    # return next_state, jacobian_state, jacobian_parameters * parameters_jacobian
    return next_state, jacobian_parameters * parameters_jacobian
end


# mech.dimensions
# mech.solver.dimensions
# nz = mech.dimensions.state
# nu = mech.dimensions.input
# nθ = mech.dimensions.parameters
# dθ = zeros(nz, nθ)
# z = rand(nz)
# u = rand(nu)
# θ = rand(nθ)
# dynamics_jacobian_parameters(dθ, mech, z, u, nothing)
# plot(Gray.(abs.(Matrix(dθ))))

function measurement_model(context::CvxContext1320, state::Vector, params::Vector{T}) where T
    pose = state[1:3] # TODO this is not safe, we need to make sure these dimensions are the pose dimensions

    mass, inertia, friction_coefficient, Ab, bb, ob = unpack(params, context)
    Aw, bw, ow = halfspace_transformation(pose, Ab, bb, ob)
    Aw, bw, ow = add_floor(Aw, bw, ow)

    d = Vector{Matrix{T}}()
    for camera in context.cameras
        di = julia_point_cloud(camera.eye_position, camera.camera_rays, camera.softness, Aw, bw, ow)
        push!(d, di)
    end
    measurement = CvxMeasurement1320(state, d)
    return measurement
end

function julia_point_cloud(context::CvxContext1320, state::Vector, params::Vector,
        camera_idx::Int, ray_idx::Int) where T
    pose = state[1:3] # TODO this is not safe, we need to make sure these dimensions are the pose dimensions

    mass, inertia, friction_coefficient, Ab, bb, ob = unpack(params, context)
    Aw, bw, ow = halfspace_transformation(pose, Ab, bb, ob)
    Aw, bw, ow = add_floor(Aw, bw, ow)

    camera = context.cameras[camera_idx]
    d = julia_intersection(camera.eye_position, camera.camera_rays[ray_idx], camera.softness, Aw, bw, ow)
    return d
end

function julia_point_cloud_jacobian_state(context::CvxContext1320, state::Vector, params::Vector{T},
        camera_idx::Int, ray_idx::Int) where T
    ForwardDiff.jacobian(state -> julia_point_cloud(context, state, params, camera_idx, ray_idx), state)
end

function julia_point_cloud_jacobian_parameters(context::CvxContext1320, state::Vector, params::Vector{T},
        camera_idx::Int, ray_idx::Int) where T
    ForwardDiff.jacobian(params -> julia_point_cloud(context, state, params, camera_idx, ray_idx), params)
end

function get_parameters(context::CvxContext1320)
    mechanism = context.mechanism

    z = zeros(mechanism.dimensions.state)
    A, b, o = get_halfspaces(mechanism, z)
    θ_geometry, polytope_dimensions = pack_halfspaces(A, b, o)

    mass = mechanism.bodies[1].mass[1]
    inertia = mechanism.bodies[1].inertia[1]
    friction_coefficient = mechanism.contacts[1].friction_coefficient[1]

    θ_dynamics = [mass; inertia; friction_coefficient]
    params = [θ_dynamics; θ_geometry]
    return params
end

function set_parameters!(context::CvxContext1320, params::Vector)
    mechanism = context.mechanism
    bodies = mechanism.bodies
    contacts = mechanism.contacts

    # set geometry
    θ_geometry = params[4:end]
    A, b, o = unpack_halfspaces(θ_geometry, context.polytope_dimensions)
    set_halfspaces!(mechanism, A, b, o)

    # set dynamics
    θ_dynamics = params[1:3]
    mass, inertia, friction_coefficient = unpack_dynamics(θ_dynamics)
    # TODO we need to generalize to many contacts and bodies
    bodies[1].mass[1] = mass
    bodies[1].inertia[1] = inertia
    contacts[1].friction_coefficient[1] = friction_coefficient

    solver_parameters = vcat(get_parameters.(bodies)..., get_parameters.(contacts)...)
    set_parameters!(mechanism.solver, solver_parameters)
    return nothing
end

# # initial guess
# params0 = get_parameters(context)
# params0.θ[1:end] .= rand(17)
# set_parameters!(context, params0)
# params1 = get_parameters(context)
# norm(params0.θ - params1.θ)




function projection(context::CvxContext1320, x;
        bound_mass=[1e-1, 1e1],
        bound_inertia=[1e-1, 1e1],
        bound_friction_coefficient=[0.0, 2.0],
        bound_b=[5e-2, 1e0],
        bound_o=[-3.0, 3.0],
        )
    nz = context.mechanism.dimensions.state
    state = deepcopy(x[1:nz])
    params = deepcopy(x[nz+1:end])
    mass, inertia, friction_coefficient, A, b, o = unpack(params, context)
    mass = clamp(mass, bound_mass...)
    inertia = clamp(inertia, bound_inertia...)
    friction_coefficient = clamp(friction_coefficient, bound_friction_coefficient...)

    for i = 1:length(A)
        for j = 1:size(A[i],1)
            A[i][j,:] ./= norm(A[i][j,:]) + 1e-5
        end
    end
    b = [clamp.(bi, bound_b...) for bi in b]
    o = [clamp.(oi, bound_o...) for oi in o]

    π_params = pack(mass, inertia, friction_coefficient, A, b, o)
    π_x = [state; π_params]
    return π_x
end


################################################################################
# test filtering
################################################################################
function filtering_objective(context::CvxContext1320, obj::CvxObjective1320,
        state::Vector, params::Vector, state_prior::Vector, params_prior::Vector,
        measurement::CvxMeasurement1320)

    z1 = state
    θ1 = params
    θ0 = params_prior
    ẑ1 = measurement.z
    d̂1 = measurement.d
    nz = length(z1)
    nθ = length(θ0)

    # process model
    z̄1, dz̄1dθ1 = process_model(context, state_prior, params)
    # measurement model
    predicted_measurement = measurement_model(context, state, params)
    d1 = predicted_measurement.d

    c = objective_function(obj, z1, θ1, θ0, z̄1, ẑ1, d1, d̂1)

    dcdz1 = ForwardDiff.gradient(z1 -> objective_function(obj, z1, θ1, θ0, z̄1, ẑ1, d1, d̂1), z1)
    dcdθ1 = ForwardDiff.gradient(θ1 -> objective_function(obj, z1, θ1, θ0, z̄1, ẑ1, d1, d̂1), θ1)
    dcdz̄1 = ForwardDiff.gradient(z̄1 -> objective_function(obj, z1, θ1, θ0, z̄1, ẑ1, d1, d̂1), z̄1)

    DcDz1 = dcdz1
    DcDθ1 = dcdθ1 + dz̄1dθ1' * dcdz̄1

    num_cameras = length(d1)
    for i = 1:num_cameras
        camera = context.cameras[i]
        num_rays = size(d1[i], 2)
        for j = 1:num_rays
            di = d1[i][:,j]
            d̂i = d̂1[i][:,j]
            dcdd1 = ForwardDiff.gradient(di -> point_cloud_objective_function(obj, di, d̂i), di)
            dd1dz1 = julia_point_cloud_jacobian_state(context, state, params, i, j)
            dd1dθ1 = julia_point_cloud_jacobian_parameters(context, state, params, i, j)
            DcDz1 += dd1dz1' * dcdd1
            DcDθ1 += dd1dθ1' * dcdd1
        end
    end
    g = [DcDz1; DcDθ1]

    # DcD2z1 = dcd2z1# + dd1dz1' * dcd2d1 * dd1dz1
    # DcD2θ1 = dcd2θ1# +
    # DcDz1θ1 = dcdz1θ1# +
    # H = [DcD2z1 DcDz1θ1; DcDz1θ1' DcD2θ1]
    H = Diagonal([diag(obj.P_state + obj.M_observation); diag(obj.P_parameters)])
    return c, g, H
end

function objective_function(obj::CvxObjective1320, z1, θ1, θ0, z̄1, ẑ1, d1, d̂1)
    c = 0.0
    # prior parameters cost
    c += 0.5 * (θ1 - θ0)' * obj.P_parameters * (θ1 - θ0)
    # prior state cost (process model = dynamics)
    c += 0.5 * (z1 - z̄1)' * obj.P_state * (z1 - z̄1)
    # measurement state cost (measurement model = identity)
    c += 0.5 * (z1 - ẑ1)' * obj.M_observation * (z1 - ẑ1)
    # measurement parameter cost (measurement model = point cloud)
    num_cameras = length(d1)
    for i = 1:num_cameras
        num_rays = size(d1[i],2)
        for j = 1:num_rays
            di = d1[i][:,j]
            d̂i = d̂1[i][:,j]
            c += point_cloud_objective_function(obj, di, d̂i)
        end
    end
    return c
end

function point_cloud_objective_function(obj::CvxObjective1320, d::Vector, d̂::Vector)
    0.5 * obj.M_point_cloud * (d - d̂)' * (d - d̂) + 0.5 * obj.M_point_cloud * norm(d - d̂)
end
