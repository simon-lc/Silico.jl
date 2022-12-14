@testset "visuals" begin
    ###############################################################################
    # visualizer
    ################################################################################vis = Visualizer()
    vis = Silico.Visualizer()
    Silico.open(vis)
    Silico.set_floor!(vis)
    Silico.set_light!(vis)
    Silico.set_background!(vis)

    ################################################################################
    # example
    ################################################################################
    mech = Silico.get_sphere_drop(
        options=Silico.Mehrotra.Options(
            verbose=false,
            ),
        )

    ################################################################################
    # test simulation
    ################################################################################
    xp2 = [+0.0,1.5,-0.25]
    vp15 = [-0,0,-0.0]
    z0 = [xp2; vp15]
    H0 = 100
    storage = Silico.simulate!(mech, z0, H0)

    ################################################################################
    # visualization
    ################################################################################
    Silico.build_mechanism!(vis, mech)
    Silico.visualize!(vis, mech, storage, build=false)

    @test true
end
