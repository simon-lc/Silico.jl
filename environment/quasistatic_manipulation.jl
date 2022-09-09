function get_quasistatic_manipulation(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.2,
    finger_friction_coefficient=0.9,
    method_type::Symbol=:finite_difference,
    options=Options(
        # verbose=false,
        complementarity_tolerance=1e-4,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=false,
        warm_start=true,
        )
    )

    Af = [0.0  +1.0]
    bf = [0.0]
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
    object_shapes = [PolytopeShape1170(Ap1, bp1)]
    finger1_shapes = [SphereShape1170(finger_radius)]
    finger2_shapes = [SphereShape1170(finger_radius)]
    bodies = [
        QuasistaticObject1170(timestep, mass, inertia, object_shapes, gravity=gravity, name=:object),
        # QuasistaticObject1170(timestep, mass, inertia, finger1_shapes, gravity=gravity, name=:finger1),
        # QuasistaticObject1170(timestep, mass, inertia, finger2_shapes, gravity=gravity, name=:finger2),
        QuasistaticRobot1170(timestep, mass, inertia, finger1_shapes, gravity=gravity, name=:finger1),
        QuasistaticRobot1170(timestep, mass, inertia, finger2_shapes, gravity=gravity, name=:finger2),
        ]
    contacts = [
        PolySphere1170(bodies[1], bodies[2],
            friction_coefficient=finger_friction_coefficient,
            name=:object_finger1),
        PolySphere1170(bodies[1], bodies[3],
            friction_coefficient=finger_friction_coefficient,
            name=:object_finger2),
        SphereSphere1170(bodies[2], bodies[3],
            friction_coefficient=friction_coefficient,
            name=:object_finger2),
        PolyHalfSpace1170(bodies[1], Af, bf,
            friction_coefficient=friction_coefficient,
            name=:object_halfspace),
        SphereHalfSpace1170(bodies[2], Af, bf,
            friction_coefficient=friction_coefficient,
            name=:finger1_halfspace),
        SphereHalfSpace1170(bodies[3], Af, bf,
            friction_coefficient=friction_coefficient,
            name=:finger2_halfspace),
        ]
    indexing!([bodies; contacts])

    local_mechanism_residual(primals, duals, slacks, parameters) =
        mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)

    mechanism = Mechanism1170(
        local_mechanism_residual,
        bodies,
        contacts,
        options=options,
        method_type=method_type)

    initialize_solver!(mechanism.solver)
    return mechanism
end
