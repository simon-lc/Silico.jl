function bilevel_methods(equality::Function, equality_jacobian::Function,
        dim::Mehrotra.Dimensions, idx::Mehrotra.Indices)
    parameter_keywords = idx.parameter_keywords

    # in-place evaluation
    function equality_constraint(out, x, θ)
        primals = x[idx.primals]
        duals = x[idx.duals]
        slacks = x[idx.slacks]
        parameters = θ
        out .= equality(primals, duals, slacks, parameters)
    end

    warning = "compressed search direction is not implemented with finite difference methods."
    function equality_constraint_compressed(out, x, θ)
        @warn warning
    end

    # jacobian variables
    function equality_jacobian_variables(vector_cache, x, θ)
        matrix_cache = reshape(vector_cache, (dim.equality, dim.variables))
        equality_jacobian(matrix_cache, primals, duals, slacks, parameters)
        matrix_cache[idx.primals, idx.primals] .+= 1e-3*Diagonal(ones(dim.primals))
        matrix_cache[idx.duals, idx.duals] .-= 1e-3*Diagonal(ones(dim.duals))
        return nothing
    end

    # jacobian variables compressed
    function equality_jacobian_variables_compressed(vector_cache, x, θ)
        error(warning)
    end

    # correction
    function correction(c, r, Δza, Δsa, κ)
        c .= r
        c[idx.cone_product] .-= (κ - Mehrotra.cone_product(Δza, Δsa, idx.cone_nonnegative, idx.cone_second_order))
        return nothing
    end

    # correction compressed
    function correction_compressed(cc, r, Δza, Δsa, x, κ)
        error(warning)
    end

    # slack direction
    function slack_direction(Δs, Δz, x, rs)
        error(warning)
    end

    # jacobian parameters
    function equality_jacobian_parameters(vector_cache, x, θ)
        error(warning)
    end

    equality_jacobian_keywords = Vector{Function}()
    for k in eachindex(parameter_keywords)
        function func(vector_cache, x, θ)
            error(warning)
        end
        push!(equality_jacobian_keywords, func)
    end

    ex_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.variables)))[1:2]...]...))
    exc_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.equality)))[1:2]...]...))
    eθ_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.parameters)))[1:2]...]...))
    eθ_indices = zeros(Int, dim.equality, dim.parameters)
    vec(eθ_indices) .= 1:length(eθ_indices)
    ek_indices = Vector{Vector{Int}}()
    for (key, val) in parameter_keywords
        indices = vec(eθ_indices[:,val])
        push!(ek_indices, indices)
    end

    methods = Mehrotra.ProblemMethods(
        equality_constraint,
        equality_constraint_compressed,
        equality_jacobian_variables,
        equality_jacobian_variables_compressed,
        equality_jacobian_parameters,
        equality_jacobian_keywords,
        correction,
        correction_compressed,
        slack_direction,
        zeros(length(ex_sparsity)),
        zeros(length(exc_sparsity)),
        zeros(length(eθ_sparsity)),
        ex_sparsity,
        exc_sparsity,
        eθ_sparsity,
        ek_indices,
    )

    return methods
end


using SparseArrays


num_primals = mech.solver.dimensions.primals
num_cone = mech.solver.dimensions.cone
parameters = mech.solver.parameters
options = mech.solver.options

bodies = mech.bodies
contacts = mech.contacts

# dimensions
dim = mech.solver.dimensions

# indices
idx = mech.solver.indices

equality(primals, duals, slacks, parameters) = mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)
Mehrotra.finite_difference_methods(equality, dim, idx)

equality_jacobian = x -> x
custom_methods = bilevel_methods(equality, equality_jacobian, dim, idx)
custom_solver = Mehrotra.Solver(
        nothing,
        num_primals,
        num_cone,
        parameters=parameters,
        nonnegative_indices=collect(1:num_cone),
        second_order_indices=[collect(1:0)],
        methods=custom_methods,
        options=options
        )

custom_mech = Mechanism(nothing, bodies, contacts;
        options=options,
        methods=custom_methods,
        )

Mehrotra.solve!(custom_solver)


function mechanism_residual_jacobian(jac, primals, duals, slacks, parameters, bodies, contacts)
    num_primals = length(primals)
    num_cone = length(duals)
    num_parameters = length(parameters)
    num_equality = num_primals + num_cone
    num_variables = num_primals + 2 * num_cone

    x = [primals; duals; slacks]
    e = zeros(eltype(x), num_equality)
    θ = parameters

    ############################################################################
    x = Symbolics.variables(:x, 1:num_variables) # variables
    θ = Symbolics.variables(:θ, 1:num_parameters) # parameters
    y = x[1:num_primals]
    z = x[num_primals .+ (1:num_cone)]
    s = x[num_primals + num_nome .+ (1:num_cone)]

    # equality residual
    f = num_parameters > 0 ?
        func(y, z, s, θ) :
        func(y, z, s)

    # equality jacobians
    fx = Symbolics.sparsejacobian(f, x)
    # sparsity
    fx_sparsity = collect(zip([findnz(fx)[1:2]...]...))
    ############################################################################


    # body
    for body in bodies
        # residual!(e, x, θ, body)
        residual_jacobian!(jac, x, θ, body)
    end

    # contact
    for contact in contacts
        # residual!(e, x, θ, contact, bodies)
        residual_jacobian!(jac, x, θ, contact, bodies)
    end

    return nothing
end
