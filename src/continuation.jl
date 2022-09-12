
################################################################################
# Continuation
################################################################################
function reset!(mechanism::Mechanism; residual_tolerance=1e-4, complementarity_tolerance=1e-3)
    msolver = mechanism.solver
    moptions = msolver.options
    moptions.residual_tolerance = residual_tolerance
    moptions.complementarity_tolerance = complementarity_tolerance
    return nothing
end

function continuation_callback!(solver, mechanism::Mechanism; ρ=1.5, visualize=false)
    msolver = mechanism.solver
    moptions = msolver.options
    # contact smoothness continuation
    moptions.residual_tolerance = max(1e-6, moptions.residual_tolerance/ρ)
    moptions.complementarity_tolerance = max(1e-4, moptions.complementarity_tolerance/ρ)

    # visualize current policy
    if visualize
        ū = solver.problem.actions
        z̄ = IterativeLQR.rollout(model, z1, ū)
        visualize!(vis, mech, z̄, build=false)
    end

    println("r_tol", moptions.residual_tolerance,
        "κ_tol", moptions.complementarity_tolerance)
    return nothing
end
