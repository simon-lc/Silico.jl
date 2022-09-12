function get_bundle_drop(;
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

    Af = [0.0  +1.0]
    bf = [0.0]
    Ap1 = [
        +1.0 -0.2;
        +0.0 +1.0;
        -1.0 -0.2;
        +0.0 -1.0;
        ]
    for i = 1:4
        Ap1[i,:] ./= norm(Ap1[i,:])
    end
    bp1 = 0.5*[
        +1.5,
        +1,
        +1.5,
        +0,
        ];
    Ap2 = [
        +1.0 +0.3;
        +0.0 +1.0;
        -1.0 +0.1;
        +0.0 -1.0;
        ]
    for i = 1:4
        Ap2[i,:] ./= norm(Ap2[i,:])
    end
    bp2 = 0.5*[
        +0.5,
        +2,
        +1,
        -1,
        ];
    Ap3 = [
        +1.0 -0.3;
        +0.0 +1.0;
        -1.0 +0.5;
        +0.0 -1.0;
        ]
    for i = 1:4
        Ap3[i,:] ./= norm(Ap3[i,:])
    end
    bp3 = 0.5*[
        +0.75,
        +3,
        +0,
        -1,
        ];
    # Ap1 = [
    #     1.0  0.0;
    #     0.0  1.0;
    #     -1.0  0.0;
    #     0.0 -1.0;
    #     ] .- 0.30ones(4,2);
    # bp1 = 0.2*[
    #     +1,
    #     +1,
    #     +1,
    #     1,
    #     ];
    # Ap2 = [
    #     1.0  0.0;
    #     0.0  1.0;
    #     -1.0  0.0;
    #     0.0 -1.0;
    #     ] .+ 0.20ones(4,2);
    # bp2 = 0.2*[
    #     -0.5,
    #     +1,
    #     +1.5,
    #     1,
    #     ];

    # nodes
    parent_shapes = [PolytopeShape(Ap1, bp1), PolytopeShape(Ap2, bp2), PolytopeShape(Ap3, bp3)]
    bodies = [
        Body(timestep, mass, inertia, parent_shapes, gravity=+gravity, name=:pbody),
        ]
    contacts = [
        PolyHalfSpace(bodies[1], Af, bf,
            friction_coefficient=friction_coefficient,
            name=:halfspace_p1),
        PolyHalfSpace(bodies[1], Af, bf,
            parent_collider_id=2,
            friction_coefficient=friction_coefficient,
            name=:halfspace_p2),
        PolyHalfSpace(bodies[1], Af, bf,
            parent_collider_id=3,
            friction_coefficient=friction_coefficient,
            name=:halfspace_p3),
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
