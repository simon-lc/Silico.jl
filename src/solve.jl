function Mehrotra.solve!(solver)
    # options
    options = solver.options
    compressed = options.compressed_search_direction
    decoupling = options.complementarity_decoupling
    complementarity_correction = options.complementarity_correction
    sparse_solver = options.sparse_solver

    # initialize
    solver.trace.iterations = 0
    warm_start = solver.options.warm_start
    # TODO replace with initialize_solver!
    !warm_start && Mehrotra.initialize_primals!(solver)
    !warm_start && Mehrotra.initialize_duals!(solver)
    !warm_start && Mehrotra.initialize_slacks!(solver)
    !warm_start && Mehrotra.initialize_interior_point!(solver)

    warm_start && (solver.solution.duals .= solver.solution.duals .+ options.complementarity_backstep)
    warm_start && (solver.solution.slacks .= solver.solution.slacks .+ options.complementarity_backstep)

    # indices
    indices = solver.indices

    # variables
    solution = solver.solution
    y = solution.primals
    z = solution.duals
    s = solution.slacks

    # candidate
    candidate = solver.candidate
    ŷ = candidate.primals
    ẑ = candidate.duals
    ŝ = candidate.slacks

    # parameters
    parameters = solver.parameters

    # solver data
    data = solver.data

    # search direction
    step = data.step
    Δy = step.primals
    Δz = step.duals
    Δs = step.slacks

    # problem
    problem = solver.problem
    methods = solver.methods
    cone_methods = solver.cone_methods

    # barrier + augmented Lagrangian
    α = solver.step_sizes
    κ = solver.central_paths

    # info
    options.verbose && Mehrotra.solver_info(solver)

    # evaluate
    Mehrotra.evaluate!(problem, methods, cone_methods, solution, parameters,
        equality_constraint=true,
        cone_constraint=true,
        sparse_solver=sparse_solver,
        compressed=compressed,
    )
    # violation
    equality_violation, cone_product_violation = Mehrotra.violation(problem, κ.tolerance_central_path)

    for i = 1:options.max_iterations
        solver.trace.iterations += 1
        # check for convergence
        if (equality_violation <= options.residual_tolerance &&
            cone_product_violation <= options.residual_tolerance)
            # set the state of the solver to :solved
            solver.consistency.solved .= true

            # differentiate
            options.differentiate && Mehrotra.differentiate!(solver)

            options.verbose && Mehrotra.solver_status(solver, true)
            return true
        end

        # evaluate everything
        Mehrotra.evaluate!(problem, methods, cone_methods, solution, parameters,
            equality_constraint=true,
            equality_jacobian_variables=true,
            cone_constraint=true,
            cone_jacobian=true,
            sparse_solver=sparse_solver,
            compressed=compressed,
        )

        ## Predictor step
        # residual
        Mehrotra.residual!(data, problem, indices,
            residual=true,
            jacobian_variables=true,
            compressed=compressed,
            sparse_solver=sparse_solver)

        # add correction to aim at the tolerance central path
        Mehrotra.correction!(methods, data, α.affine_step_size, step, data.step_correction, solution, κ.tolerance_central_path;
            compressed=compressed, complementarity_correction=0.0)
        # search direction
        Mehrotra.search_direction!(solver)
        # affine line search
        α.affine_step_size .= 1.0
        # cone search duals
        Mehrotra.cone_search!(α.affine_step_size, z, Δz,
            indices.cone_nonnegative, indices.cone_second_order;
            τ_nn=0.9500, τ_soc=0.9500, ϵ=1e-14, decoupling=decoupling)
        # cone search slacks
        Mehrotra.cone_search!(α.affine_step_size, s, Δs,
            indices.cone_nonnegative, indices.cone_second_order;
            τ_nn=0.9500, τ_soc=0.9500, ϵ=1e-14, decoupling=decoupling)

        # centering
        Mehrotra.centering!(κ.target_central_path, solution, step, α.affine_step_size, indices, options=options)

        ## Corrector step
        # remove correction aiming at the tolerance central path
        # add correction aiming at the target central path - second order correction
        @. κ.correction_central_path .= κ.target_central_path .- κ.tolerance_central_path
        Mehrotra.correction!(methods, data, α.affine_step_size, step, data.step_correction, solution, κ.correction_central_path;
            compressed=compressed, complementarity_correction=complementarity_correction)
        Mehrotra.search_direction!(solver, factorize=false)
        # line search
        α.step_size .= 1.0
        # cone search duals
        Mehrotra.cone_search!(α.step_size, z, Δz,
            indices.cone_nonnegative, indices.cone_second_order;
            τ_nn=0.9500, τ_soc=0.9500, ϵ=1e-14, decoupling=decoupling)
        # cone search slacks
        Mehrotra.cone_search!(α.step_size, s, Δs,
            indices.cone_nonnegative, indices.cone_second_order;
            τ_nn=0.9500, τ_soc=0.9500, ϵ=1e-14, decoupling=decoupling)

        # violation
        equality_violation, cone_product_violation = Mehrotra.violation(problem, κ.tolerance_central_path)

        for i = 1:options.max_iteration_line_search
            # update candidate
            for i = 1:solver.dimensions.primals
                ŷ[i] = y[i] + minimum(α.step_size) * Δy[i]
            end
            for i = 1:solver.dimensions.duals
                ẑ[i] = z[i] + α.step_size[i] * Δz[i]
            end
            for i = 1:solver.dimensions.slacks
                ŝ[i] = s[i] + α.step_size[i] * Δs[i]
            end

            # evaluate residual
            Mehrotra.evaluate!(problem, methods, cone_methods, candidate, parameters,
                equality_constraint=true,
                cone_constraint=true,
                sparse_solver=sparse_solver,
                compressed=compressed,
            )

            # violations
            equality_violation_candidate, cone_product_violation_candidate = Mehrotra.violation(problem, κ.tolerance_central_path)

            # Test progress
            if (equality_violation_candidate <= equality_violation ||
                cone_product_violation_candidate <= cone_product_violation)
                equality_violation = equality_violation_candidate
                cone_product_violation = cone_product_violation_candidate
                break
            end

            # decrease step size
            α.step_size .= options.scaling_line_search .* α.step_size

            i == options.max_iteration_line_search && (options.verbose && (@warn "line search failure"); break)
        end

        # update
        for i = 1:solver.dimensions.primals
            y[i] = ŷ[i]
        end
        for i = 1:solver.dimensions.duals
            z[i] = ẑ[i]
        end
        for i = 1:solver.dimensions.slacks
            s[i] = ŝ[i]
        end

        # status
        options.verbose && Mehrotra.iteration_status(
            i,
            equality_violation,
            cone_product_violation,
            κ.target_central_path[1],
            minimum(α.step_size))
    end

    # failure
    options.verbose && Mehrotra.solver_status(solver, false)
    return false
end
