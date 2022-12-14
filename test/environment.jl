@testset "environment" begin
    ################################################################################
    # polytope_drop
    ################################################################################
    mech = Silico.get_polytope_drop(
        options=Silico.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    vp15 = [-0,0,-0.0]
    z0 = [xp2; vp15]
    H0 = 100
    storage = Silico.simulate!(mech, z0, H0)
    @test true


    ################################################################################
    # bundle_drop
    ################################################################################
    mech = Silico.get_bundle_drop(
        options=Silico.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    vp15 = [-0,0,-0.0]
    z0 = [xp2; vp15]
    H0 = 100
    storage = Silico.simulate!(mech, z0, H0)
    @test true

    ################################################################################
    # bundle_collsion
    ################################################################################
    mech = Silico.get_bundle_collsion(
        options=Silico.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    vp15 = [-0,0,-0.0]
    z00 = [xp2; vp15]

    xp2 = [+0.0,3.5,-0.25]
    vp15 = [-0,0,-0.0]
    z01 = [xp2; vp15]

    z0 = [z00; z01]
    H0 = 100
    storage = Silico.simulate!(mech, z0, H0)
    @test true

    ################################################################################
    # sphere_drop
    ################################################################################
    mech = Silico.get_sphere_drop(
        options=Silico.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    vp15 = [-0,0,-0.0]
    z0 = [xp2; vp15]
    H0 = 100
    storage = Silico.simulate!(mech, z0, H0)
    @test true


    ################################################################################
    # sphere_bundle
    ################################################################################
    mech = Silico.get_sphere_bundle(
        options=Silico.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    vp15 = [-0,0,-0.0]
    z00 = [xp2; vp15]

    xp2 = [+0.0,3.5,-0.25]
    vp15 = [-0,0,-0.0]
    z01 = [xp2; vp15]

    z0 = [z00; z01]
    H0 = 100
    storage = Silico.simulate!(mech, z0, H0)
    @test true


    ################################################################################
    # sphere_collision
    ################################################################################
    mech = Silico.get_sphere_collision(
        options=Silico.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    vp15 = [-0,0,-0.0]
    z00 = [xp2; vp15]

    xp2 = [+0.0,3.5,-0.25]
    vp15 = [-0,0,-0.0]
    z01 = [xp2; vp15]

    z0 = [z00; z01]
    H0 = 100
    storage = Silico.simulate!(mech, z0, H0)
    @test true


    ################################################################################
    # quasistatic_sphere_drop
    ################################################################################
    mech = Silico.get_quasistatic_sphere_drop(
        options=Silico.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    z0 = [xp2;]

    H0 = 100
    storage = Silico.simulate!(mech, z0, H0)
    @test true


    ################################################################################
    # quasistatic_sphere_bundle
    ################################################################################
    mech = Silico.get_quasistatic_sphere_bundle(
        options=Silico.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    z00 = [xp2;]

    xp2 = [+0.0,3.5,-0.25]
    z01 = [xp2;]

    z0 = [z00; z01]
    H0 = 100
    storage = Silico.simulate!(mech, z0, H0)
    @test true


    ################################################################################
    # quasistatic_manipulation
    ################################################################################
    mech = Silico.get_quasistatic_manipulation(
        options=Silico.Mehrotra.Options(verbose=false))

    xp2 = [+0.0,1.5,-0.25]
    z00 = [xp2;]

    xp2 = [-1.0,0.5,-0.25]
    z01 = [xp2;]

    xp2 = [+1.0,0.5,-0.25]
    z02 = [xp2;]

    z0 = [z00; z01; z02]
    H0 = 100
    storage = Silico.simulate!(mech, z0, H0)
    @test true

end


# ###############################################################################
# # visualizer
# ################################################################################vis = Visualizer()
# vis = Silico.Visualizer()
# Silico.open(vis)
# Silico.set_floor!(vis)
# Silico.set_light!(vis)
# Silico.set_background!(vis)
#
#
#
# ################################################################################
# # quasistatic_manipulation
# ################################################################################
# mech = Silico.get_quasistatic_manipulation(
#     options=Silico.Mehrotra.Options(verbose=false))
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
# storage = Silico.simulate!(mech, z0, H0)
# @test true
#
# ################################################################################
# # visualization
# ################################################################################
# Silico.build_mechanism!(vis, mech)
# Silico.visualize!(vis, mech, storage, build=false)
