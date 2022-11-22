function jacobian_transpose_ik!(state::MechanismState,
                               body::RigidBody,
                               points::Vector{<:Point3D},
                               desired::Vector{<:Point3D};
                               α=0.20,
                               iterations=200)
    mechanism = state.mechanism
    world = root_frame(mechanism)

    ## Compute the joint path from world to our target body
    p = path(mechanism, root_body(mechanism), body)
    ## Allocate the point jacobian (we'll update this in-place later)
    Jp0 = point_jacobian(state, p, transform(state, points[1], world))
    Jp1 = point_jacobian(state, p, transform(state, points[2], world))
    Jp2 = point_jacobian(state, p, transform(state, points[3], world))

    q = copy(configuration(state))

    for i in 1:iterations
        ## Update the position of the point
        point0_in_world = transform(state, points[1], world)
        point1_in_world = transform(state, points[2], world)
        point2_in_world = transform(state, points[3], world)
        desired0_in_world = transform(state, desired[1], world)
        desired1_in_world = transform(state, desired[2], world)
        desired2_in_world = transform(state, desired[3], world)

        ## Update the point's jacobian
        point_jacobian!(Jp0, state, p, point0_in_world)
        point_jacobian!(Jp1, state, p, point1_in_world)
        point_jacobian!(Jp2, state, p, point2_in_world)

        residual = [
            (desired0_in_world - point0_in_world).v;
            (desired1_in_world - point1_in_world).v;
            (desired2_in_world - point2_in_world).v;
            ]
        jacobian = [Array(Jp0)' Array(Jp1)' Array(Jp2)']
        ## Compute an update in joint coordinates using the jacobian transpose
        Δq = α * jacobian * residual
        # println(round.(residual, digits=3))
        ## Apply the update
        q .= configuration(state) .+ Δq
        set_configuration!(state, q)
        # set_configuration!(mvis, configuration(state))
    end
    return state
end

function set_panda!(mvis::MechanismVisualizer, q)
    set_configuration!(mvis, q)
    return nothing
end

function panda_configuration(q, state, mvis)
    # select points on the body
    body = findbody(robot, "link7")
    point_0 = Point3D(default_frame(body), 0.00, 0.00, 0.16)
    point_1 = Point3D(default_frame(body), 0.10, 0.00, 0.16)
    point_2 = Point3D(default_frame(body), 0.00, 0.00, 0.16 + 0.10)
    points = [point_0, point_1, point_2]

    # render the points
    radius = 0.025
    setelement!(mvis, point_0, radius, "point_0")
    setelement!(mvis, point_1, radius, "point_1")
    setelement!(mvis, point_2, radius, "point_2")

    # state = MechanismState(robot)
    world = root_frame(mvis.state.mechanism)
    point0_in_world = transform(mvis.state, points[1], world)
    point1_in_world = transform(mvis.state, points[2], world)
    point2_in_world = transform(mvis.state, points[3], world)

    set_configuration!(mvis, q)

    # Choose a desired location. We'll move the tip of the arm to
    desired_point_0 = [0; state[1:2]]
    desired_point_1 = desired_point_0 + [-0.10, 0, 0]
    # desired_point_1 = desired_point_0 + [0, -0.10, 0]
    # desired_point_1 = desired_point_0 + [0, 0, -0.10]
    θ = state[3]
    desired_point_2 = desired_point_0 + 0.10 * [0, cos(θ), sin(θ)]

    mat = MeshPhongMaterial(color=RGBA(0,0,0,1))
    setobject!(vis[:desired_0], HyperSphere(Point(desired_point_0...), radius), mat)
    setobject!(vis[:desired_1], HyperSphere(Point(desired_point_1...), radius), mat)
    setobject!(vis[:desired_2], HyperSphere(Point(desired_point_2...), radius), mat)

    desired_tip_locations = [
        Point3D(root_frame(robot), desired_point_0...),
        Point3D(root_frame(robot), desired_point_1...),
        Point3D(root_frame(robot), desired_point_2...),
        ]
    # Run the IK, updating `state` in place
    jacobian_transpose_ik!(mvis.state, body, points, desired_tip_locations, iterations=2000, α=0.50)
    set_configuration!(mvis, configuration(mvis.state))

    q_new = copy(mvis.state.q)
    return q_new
end

function panda_trajectory(state_trajectory, mvis::MechanismVisualizer;
        initial_q=[-π/2, +0.60, +0.00, -1.80, +0.00, +2.40, +π/2, 0.05, 0.05])

    q = initial_q
    q_trajectory = []
    for i = 1:length(state_trajectory)
        state = deepcopy(state_trajectory[i])
        q_new = panda_configuration(q, state, mvis)
        push!(q_trajectory, q_new)
        q = q_new
    end
    return q_trajectory
end

function visualize!(mvis::MechanismVisualizer, q_trajectory;
        animation=MeshCat.Animation(Int(floor(1/0.05)))
        )

    for i = 1:length(q_trajectory)
        atframe(animation, i) do
            set_panda!(mvis, q_trajectory[i])
        end
    end
    MeshCat.setanimation!(vis, animation)
    return mvis, animation
end
