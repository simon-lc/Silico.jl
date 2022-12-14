function get_quasistatic_manipulation(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.2,
    finger_friction_coefficient=0.9,
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

    Ap1 = [
        1.0  0.0;
        0.0  1.0;
        -1.0  0.0;
        0.0 -1.0;
        ] .- 0.30ones(4,2);
    bp1 = 0.3*[
        +1,
        +1,
        +1,
        1,
        ];
    finger_radius = 0.2

    # nodes
    object_shapes = [PolytopeShape(Ap1, bp1)]
    finger1_shapes = [SphereShape(finger_radius)]
    finger2_shapes = [SphereShape(finger_radius)]
    bodies = [
        QuasistaticObject(timestep, mass, inertia, object_shapes, gravity=gravity, name=:object),
        # QuasistaticObject(timestep, mass, inertia, finger1_shapes, gravity=gravity, name=:finger1),
        # QuasistaticObject(timestep, mass, inertia, finger2_shapes, gravity=gravity, name=:finger2),
        QuasistaticRobot(timestep, mass, inertia, finger1_shapes, gravity=gravity, name=:finger1),
        QuasistaticRobot(timestep, mass, inertia, finger2_shapes, gravity=gravity, name=:finger2),
        ]
    normal = [0.0, 1.0]
    position_offset = [0.0, 0.0]
    floor_shape = HalfspaceShape(normal, position_offset)
    contacts = [
        PolySphere(bodies[1], bodies[2],
            friction_coefficient=finger_friction_coefficient,
            name=:object_finger1),
        PolySphere(bodies[1], bodies[3],
            friction_coefficient=finger_friction_coefficient,
            name=:object_finger2),
        SphereSphere(bodies[2], bodies[3],
            friction_coefficient=friction_coefficient,
            name=:object_finger2),
        PolyHalfSpace(bodies[1], floor_shape,
            friction_coefficient=friction_coefficient,
            name=:object_halfspace),
        SphereHalfSpace(bodies[2], floor_shape,
            friction_coefficient=friction_coefficient,
            name=:finger1_halfspace),
        SphereHalfSpace(bodies[3], floor_shape,
            friction_coefficient=friction_coefficient,
            name=:finger2_halfspace),
        ]
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
