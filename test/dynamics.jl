@testset "dynamics gradients" begin
    # ################################################################################
    # # visualizer
    # ################################################################################vis = Visualizer()
    # vis = Visualizer()
    # open(vis)
    # set_floor!(vis)
    # set_light!(vis)
    # set_background!(vis)

    ################################################################################
    # example
    ################################################################################
    timestep = 0.05;
    gravity = -9.81;
    mass = 1.0;
    inertia = 0.2 * ones(1,1);
    friction_coefficient = 0.1;

    mech = Silico.get_polytope_drop(;
        timestep=timestep,
        gravity=gravity,
        mass=mass,
        inertia=inertia,
        friction_coefficient=friction_coefficient,
        method_type=:symbolic,
        options=Silico.Mehrotra.Options(
            verbose=false,
            complementarity_tolerance=1e-3,
            compressed_search_direction=true,
            max_iterations=30,
            sparse_solver=true,
            differentiate=true,
            )
        );

    ################################################################################
    # test simulation
    ################################################################################
    xp2 = [+0.0,1.5,-0.25]
    vp15 = [-0,0,-0.0]
    z0 = [xp2; vp15]
    H0 = 100
    storage = Silico.simulate!(mech, z0, H0)

    # ################################################################################
    # # visualization
    # ################################################################################
    # Silico.build_mechanism!(vis, mech)
    # Silico.visualize!(vis, mech, storage, build=false)

    ################################################################################
    # set_state_control_parameters
    ################################################################################
    z0 = [4.0, 5.0, 6.0, 1.0, 3.0, 2.0]
    u0 = [0.1, 0.2, 0.3]
    w0 = [0.5]
    idx_parameters = [14]
    Silico.set_state_control_parameters!(mech, z0, u0; w=w0, idx_parameters=idx_parameters)

    @test norm(Silico.get_current_state(mech) - z0, Inf) < 1e-8
    @test norm(Silico.get_input(mech) - u0, Inf) < 1e-8
    @test norm(Silico.get_parameters(mech)[idx_parameters] - w0) < 1e-8


    ################################################################################
    # dynamics
    ################################################################################
    nz = mech.dimensions.state
    nu = mech.dimensions.input
    nw = length(idx_parameters)

    z1 = zeros(nz)
    dz = zeros(nz, nz)
    du = zeros(nz, nu)
    dw = zeros(nz, nw)

    Silico.dynamics(z1, mech, z0, u0; w=w0, idx_parameters=idx_parameters)
    Silico.dynamics_jacobian_state(dz, mech, z0, u0; w=w0, idx_parameters=idx_parameters)
    Silico.dynamics_jacobian_input(du, mech, z0, u0; w=w0, idx_parameters=idx_parameters)
    Silico.dynamics_jacobian_parameters(dw, mech, z0, u0; w=w0, idx_parameters=idx_parameters)

    # dynamics
    z10 = Silico.explicit_dynamics(mech, z0, u0; w=w0, idx_parameters=idx_parameters)
    norm(z1 - z10, Inf)
    # jacobian state
    dz0 = FiniteDiff.finite_difference_jacobian(z0 ->
        Silico.explicit_dynamics(mech, z0, u0; w=w0, idx_parameters=idx_parameters),
        z0)
    # jacobian input
    du0 = FiniteDiff.finite_difference_jacobian(u0 ->
        Silico.explicit_dynamics(mech, z0, u0; w=w0, idx_parameters=idx_parameters),
        u0)
    # jacobian parameters
    dw0 = FiniteDiff.finite_difference_jacobian(w0 ->
        Silico.explicit_dynamics(mech, z0, u0; w=w0, idx_parameters=idx_parameters),
        w0)

    @test norm(dz - dz0, Inf) < 1e-4
    @test norm(du - du0, Inf) < 1e-4
    @test norm(dw - dw0, Inf) < 1e-4

end
