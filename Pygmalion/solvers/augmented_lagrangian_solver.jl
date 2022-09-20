function augmented_lagrangian_solver!(xinit,
        objective,
        objective_grad,
        objective_hess,
        constraint,
        constraint_jac;
        projection=x->x,
        step_projection=x->x,
        outer_iterations=20,
        inner_iterations=20,
        line_search_iterations=15,
        reg_min=1e-4,
        reg_max=1e+0,
        reg_step=1.05,
        ρ_min=1e-6,
        ρ_max=1e+0,
        ρ_step=0.3,
        objective_tolerance=1e-4,
        constraint_tolerance=1e-4,
        D=Diagonal(ones(length(xinit))))

    # merit function
    function merit(x, λ, ρ)
        c = constraint(x)
        l = objective(x) + λ' * c + 0.5/ρ * c' * c
    end

    # initialization
    stall = 0
    x = deepcopy(xinit)
    trace = [deepcopy(x)]
    nx = length(x)
    nλ = length(constraint(x))
    λ = zeros(nλ)
    ρ = 1.0
    reg = reg_max
    iterations = 0

    for i = 1:outer_iterations
        for j = 1:inner_iterations
            iterations += 1
            c = constraint(x)
            l = objective(x)
            (l < objective_tolerance) && (norm(c, Inf) < constraint_tolerance) && break
            ∇c = constraint_jac(x)
            g = objective_grad(x) + ∇c' * λ + 1/ρ * ∇c' * c
            H = objective_hess(x) + 1/ρ * ∇c' * ∇c + reg * D
            # r = [
            #     g + ∇c' * λ + 1/ρ * ∇c' * c;
            #     c;
            #     ]
            # ∇r = [
            #     H + 1/ρ * ∇c' * ∇c + reg * D  +∇c';
            #     ∇c                            -ρ*I(nλ);
            #     ]
            # Δ = - ∇r \ r
            # Δx = [1:nx]
            # Δλ = [nx .+ (1:nλ)]
            Δx = - H \ g

            # linesearch
            α = 1.0
            for j = 1:line_search_iterations
                l_candidate = merit(projection(x + step_projection(α * Δx)), λ, ρ)
                if l_candidate <= l
                    reg = clamp(reg/reg_step, reg_min, reg_max)
                    stall = 0
                    break
                end
                α /= 2
                if j == line_search_iterations
                    stall += 1
                    α = 0.0
                    reg = clamp(reg*exp(3.0*log(reg_step)), reg_min, reg_max)
                end
            end

            augmented_lagrangian_header(iterations)
            # iteration information
            @printf("%3d  %3d   %9.2e   %9.2e   %9.2e   %9.2e   %9.2e\n",
                i,
                j,
                l,
                mean(α),
                norm(step_projection(α * Δx), Inf),
                norm(g, Inf),
                reg,
                )
            x = projection(x + step_projection(α * Δx))
            push!(trace, deepcopy(x))
        end
        λ = λ + 1/ρ * constraint(x)
        ρ = clamp(ρ * ρ_step, ρ_min, ρ_max)
    end
    return x, trace
end


function augmented_lagrangian_header(iterations)
    # header
    if rem(iterations - 1, 10) == 0
        @printf "-------------------------------------------------------------------\n"
        @printf "out inn   objective        step        |step|∞     |objective_grad|∞     reg         \n"
        @printf "-------------------------------------------------------------------\n"
    end
    return nothing
end
