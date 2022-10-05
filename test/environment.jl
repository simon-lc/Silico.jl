@testset "environment" begin
    ################################################################################
    # polytope_drop
    ################################################################################
    mech = DojoLight.get_polytope_drop(
        options=DojoLight.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    vp15 = [-0,0,-0.0]
    z0 = [xp2; vp15]
    H0 = 100
    storage = DojoLight.simulate!(mech, z0, H0)
    @test true


    ################################################################################
    # bundle_drop
    ################################################################################
    mech = DojoLight.get_bundle_drop(
        options=DojoLight.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    vp15 = [-0,0,-0.0]
    z0 = [xp2; vp15]
    H0 = 100
    storage = DojoLight.simulate!(mech, z0, H0)
    @test true

    ################################################################################
    # bundle_collsion
    ################################################################################
    mech = DojoLight.get_bundle_collsion(
        options=DojoLight.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    vp15 = [-0,0,-0.0]
    z00 = [xp2; vp15]

    xp2 = [+0.0,3.5,-0.25]
    vp15 = [-0,0,-0.0]
    z01 = [xp2; vp15]

    z0 = [z00; z01]
    H0 = 100
    storage = DojoLight.simulate!(mech, z0, H0)
    @test true

    ################################################################################
    # sphere_drop
    ################################################################################
    mech = DojoLight.get_sphere_drop(
        options=DojoLight.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    vp15 = [-0,0,-0.0]
    z0 = [xp2; vp15]
    H0 = 100
    storage = DojoLight.simulate!(mech, z0, H0)
    @test true


    ################################################################################
    # sphere_bundle
    ################################################################################
    mech = DojoLight.get_sphere_bundle(
        options=DojoLight.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    vp15 = [-0,0,-0.0]
    z00 = [xp2; vp15]

    xp2 = [+0.0,3.5,-0.25]
    vp15 = [-0,0,-0.0]
    z01 = [xp2; vp15]

    z0 = [z00; z01]
    H0 = 100
    storage = DojoLight.simulate!(mech, z0, H0)
    @test true


    ################################################################################
    # sphere_collision
    ################################################################################
    mech = DojoLight.get_sphere_collision(
        options=DojoLight.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    vp15 = [-0,0,-0.0]
    z00 = [xp2; vp15]

    xp2 = [+0.0,3.5,-0.25]
    vp15 = [-0,0,-0.0]
    z01 = [xp2; vp15]

    z0 = [z00; z01]
    H0 = 100
    storage = DojoLight.simulate!(mech, z0, H0)
    @test true


    ################################################################################
    # quasistatic_sphere_drop
    ################################################################################
    mech = DojoLight.get_quasistatic_sphere_drop(
        options=DojoLight.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    z0 = [xp2;]

    H0 = 100
    storage = DojoLight.simulate!(mech, z0, H0)
    @test true


    ################################################################################
    # quasistatic_sphere_bundle
    ################################################################################
    mech = DojoLight.get_quasistatic_sphere_bundle(
        options=DojoLight.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    z00 = [xp2;]

    xp2 = [+0.0,3.5,-0.25]
    z01 = [xp2;]

    z0 = [z00; z01]
    H0 = 100
    storage = DojoLight.simulate!(mech, z0, H0)
    @test true


    ################################################################################
    # quasistatic_manipulation
    ################################################################################
    mech = DojoLight.get_quasistatic_manipulation(
        options=DojoLight.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    z00 = [xp2;]

    xp2 = [-1.0,0.5,-0.25]
    z01 = [xp2;]

    xp2 = [+1.0,0.5,-0.25]
    z02 = [xp2;]

    z0 = [z00; z01; z02]
    H0 = 100
    storage = DojoLight.simulate!(mech, z0, H0)
    @test true

end


# ###############################################################################
# # visualizer
# ################################################################################vis = Visualizer()
# vis = DojoLight.Visualizer()
# DojoLight.open(vis)
# DojoLight.set_floor!(vis)
# DojoLight.set_light!(vis)
# DojoLight.set_background!(vis)
#
#
#
# ################################################################################
# # quasistatic_manipulation
# ################################################################################
# mech = DojoLight.get_quasistatic_manipulation(
#     options=DojoLight.Mehrotra.Options(verbose=false))
#
# xp2 = [+0.0,1.5,-0.25]
# z00 = [xp2;]
#
# xp2 = [-1.0,0.5,-0.25]
# z01 = [xp2;]
#
# xp2 = [+1.0,0.5,-0.25]
# z02 = [xp2;]
#
# z0 = [z00; z01; z02]
# H0 = 100
# storage = DojoLight.simulate!(mech, z0, H0)
# @test true
#
# ################################################################################
# # visualization
# ################################################################################
# DojoLight.build_mechanism!(vis, mech)
# DojoLight.visualize!(vis, mech, storage, build=false)
