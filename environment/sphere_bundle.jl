function get_sphere_bundle(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
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
    bp1 = 0.2*[
        +1,
        +1,
        +1,
        1,
        ];
    Ap2 = [
        1.0  0.0;
        0.0  1.0;
        -1.0  0.0;
        0.0 -1.0;
        ] .+ 0.20ones(4,2);
    bp2 = 0.2*[
        -0.5,
        +1,
        +1.5,
        1,
        ];
    child_radius = 0.2

    # nodes
    parent_shapes = [PolytopeShape1170(Ap1, bp1), PolytopeShape1170(Ap2, bp2)]
    child_shapes = [SphereShape1170(child_radius)]
    bodies = [
        Body1170(timestep, mass, inertia, parent_shapes, gravity=+gravity, name=:pbody),
        Body1170(timestep, mass, inertia, child_shapes, gravity=+gravity, name=:cbody),
        ]
    contacts = [
        PolySphere1170(bodies[1], bodies[2],
            friction_coefficient=friction_coefficient,
            name=:contact_1),
        PolySphere1170(bodies[1], bodies[2],
            parent_collider_id=2,
            friction_coefficient=friction_coefficient,
            name=:contact_2),
        PolyHalfSpace1170(bodies[1], Af, bf,
            friction_coefficient=friction_coefficient,
            name=:halfspace_p1),
        PolyHalfSpace1170(bodies[1], Af, bf,
            parent_collider_id=2,
            friction_coefficient=friction_coefficient,
            name=:halfspace_p2),
        SphereHalfSpace1170(bodies[2], Af, bf,
            friction_coefficient=friction_coefficient,
            name=:halfspace_c),
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
