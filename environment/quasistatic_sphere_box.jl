function get_quasistatic_sphere_box2(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
    method_type::Symbol=:finite_difference,
    control_mode::Symbol=:robot, # :robot or :object
    options=Mehrotra.Options(
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
        ]
    bp1 = 0.4*[
        +1,
        +1,
        +1,
        +1,
        ];
    child_radius = 0.1

    # nodes
    parent_shapes = [PolytopeShape(Ap1, bp1)]
    child_shapes = [SphereShape(child_radius)]

    box = QuasistaticObject(timestep, mass, inertia, parent_shapes, gravity=+gravity, name=:box)
    sphere = (control_mode == :robot) ?
        QuasistaticRobot(timestep, mass, inertia, child_shapes, gravity=+0.000000000000000000000*gravity, name=:sphere) :
        QuasistaticObject(timestep, mass, inertia, child_shapes, gravity=+gravity, name=:sphere)
    bodies = [box, sphere]

    contacts = [
        PolySphere(bodies[1], bodies[2],
            friction_coefficient=friction_coefficient,
            name=:box_sphere),
        PolyHalfSpace(bodies[1], Af, bf,
            friction_coefficient=friction_coefficient,
            name=:halfspace_box),
        SphereHalfSpace(bodies[2], Af, bf,
            friction_coefficient=friction_coefficient,
            name=:halfspace_sphere),
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

    initialize_solver!(mechanism.solver)
    return mechanism
end
