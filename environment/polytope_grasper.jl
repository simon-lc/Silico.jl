function get_polytope_grasper(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
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

    A = [
        +1.0 +0.0;
        +0.0 +1.0;
        -1.0 +0.0;
        +0.0 -1.0;
        ]
    b = 0.25*[1,1,1,1]

    # nodes
    grasper_shapes = [
        CapsuleShape(0.05, 0.5),
        CapsuleShape(0.05, 0.5),
        CapsuleShape(0.05, 0.5),
        CapsuleShape(0.05, 0.5)]
    object_shapes = [PolytopeShape(A, b)]
    floor_shape = HalfspaceShape([0, 1.0])

    bodies = [
        QuasistaticObject(timestep, mass, inertia, object_shapes, gravity=+gravity, name=:sphere),
        QuasistaticRobot(timestep, mass, inertia, grasper_shapes[1:2], gravity=+0*gravity, name=:grasper_2),
        QuasistaticRobot(timestep, mass/3, inertia, grasper_shapes[3:4], gravity=+0*gravity, name=:grasper_4),
        ]
    contacts = [
        Contact2D(bodies[2], bodies[1],
            friction_coefficient=friction_coefficient,
            name=:contact_1),
        Contact2D(bodies[3], bodies[1],
            friction_coefficient=friction_coefficient,
            name=:contact_2),
        PolyHalfSpace(bodies[1], floor_shape,
            friction_coefficient=friction_coefficient,
            name=:object_halfspace),
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
