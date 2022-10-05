@testset "visuals" begin
    ###############################################################################
    # visualizer
    ################################################################################vis = Visualizer()
    vis = DojoLight.Visualizer()
    DojoLight.open(vis)
    DojoLight.set_floor!(vis)
    DojoLight.set_light!(vis)
    DojoLight.set_background!(vis)

    ################################################################################
    # example
    ################################################################################
    mech = DojoLight.get_sphere_drop(verbose=false)

    ################################################################################
    # test simulation
    ################################################################################
    xp2 = [+0.0,1.5,-0.25]
    vp15 = [-0,0,-0.0]
    z0 = [xp2; vp15]
    H0 = 100
    storage = DojoLight.simulate!(mech, z0, H0)

    ################################################################################
    # visualization
    ################################################################################
    DojoLight.build_mechanism!(vis, mech)
    DojoLight.visualize!(vis, mech, storage, build=false)

    @test true
end
