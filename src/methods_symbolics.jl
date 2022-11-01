function Mehrotra.generate_symbolic_gradients(func::Function, dim::Dimensions, ind::Indices;
        parameter_keywords=Dict{Symbol,Vector{Int}}(:all => ind.parameters),
        primal_regularizer=1e-3,
        dual_regularizer=1e-6,
        checkbounds=true,
        threads=false)

    parallel = threads ? Mehrotra.Symbolics.MultithreadedForm() : Mehrotra.Symbolics.SerialForm()
    parallel_parameters = (threads && num_parameters > 0) ? Mehrotra.Symbolics.MultithreadedForm() : Mehrotra.Symbolics.SerialForm()

    idx_nn = ind.cone_nonnegative
    idx_soc = ind.cone_second_order

    x = Mehrotra.Symbolics.variables(:x, 1:dim.variables) # variables
    θ = Mehrotra.Symbolics.variables(:θ, 1:dim.parameters) # parameters
    κ = Mehrotra.Symbolics.variables(:κ, 1:dim.cone) # central path
    r = Mehrotra.Symbolics.variables(:r, 1:dim.variables) # residual
    rs = Mehrotra.Symbolics.variables(:rs, 1:dim.cone) # cone product residual with corrections
    Δz = Mehrotra.Symbolics.variables(:Δz, 1:dim.cone) # dual step
    Δza = Mehrotra.Symbolics.variables(:Δza, 1:dim.cone) # dual affine step
    Δsa = Mehrotra.Symbolics.variables(:Δsa, 1:dim.cone) # slack affine step
    y = x[ind.primals]
    z = x[ind.duals]
    s = x[ind.slacks]

    # equality residual
    f = dim.parameters > 0 ?
        func(y, z, s, θ) :
        func(y, z, s)

    # equality jacobians
    fx = Mehrotra.Symbolics.sparsejacobian(f, x)
    fx[ind.primals, ind.primals] += primal_regularizer * I(dim.primals)
    fx[ind.duals, ind.duals] -= dual_regularizer * I(dim.duals)
    fθ = Mehrotra.Symbolics.sparsejacobian(f, θ)
    fk = [Mehrotra.Symbolics.sparsejacobian(f, θ[parameter_keywords[k]]) for k in eachindex(parameter_keywords)]

    # compressed search direction
    D = fx[ind.slackness, ind.slacks]
    Zi = Mehrotra.cone_product_jacobian_inverse(s, z, idx_nn, idx_soc)
    S = Mehrotra.cone_product_jacobian(z, s, idx_nn, idx_soc)

    # compressed equality residual
    fc = copy(f)
    fc[ind.slackness] .-= D * Zi * Mehrotra.cone_product(s, z, idx_nn, idx_soc)

    # compressed equality jacobians
    fxc = copy(fx[ind.equality, [ind.primals; ind.duals]])
    fxc[ind.slackness, ind.duals] .-= D * Zi * S
    fxc = Mehrotra.SparseArrays.sparse(fxc)

    # correction
    c = copy(r)
    c[ind.cone_product] .-= (κ - Mehrotra.cone_product(Δza, Δsa, idx_nn, idx_soc))

    # correction compressed
    cc = copy(r)
    cc[ind.slackness] .+= D * Zi * (κ - Mehrotra.cone_product(Δza, Δsa, idx_nn, idx_soc))
    # cc[ind.cone_product] .-= κ

    # slack direction
    Δs = Zi * (-rs - S * Δz)

    # sparsity
    fx_sparsity = collect(zip([Mehrotra.SparseArrays.findnz(fx)[1:2]...]...))
    fxc_sparsity = collect(zip([Mehrotra.SparseArrays.findnz(fxc)[1:2]...]...))
    fθ_sparsity = collect(zip([Mehrotra.SparseArrays.findnz(fθ)[1:2]...]...))
    fθ_indices = similar(fθ, Int)
    fθ_indices.nzval .= 1:Mehrotra.SparseArrays.nnz(fθ_indices)
    fk_indices = Vector{Vector{Int}}()
    for (key, val) in parameter_keywords
        idx = fθ_indices[:,val].nzval
        push!(fk_indices, idx)
    end

    # expressions
    f_expr = Mehrotra.Symbolics.build_function(f, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fc_expr = Mehrotra.Symbolics.build_function(fc, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fx_expr = Mehrotra.Symbolics.build_function(fx.nzval, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fxc_expr = Mehrotra.Symbolics.build_function(fxc.nzval, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fθ_expr = Mehrotra.Symbolics.build_function(fθ.nzval, x, θ,
        parallel=parallel_parameters,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fk_expr = [Mehrotra.Symbolics.build_function(fki.nzval, x, θ,
        parallel=parallel_parameters,
        checkbounds=checkbounds,
        expression=Val{false})[2] for fki in fk]
    c_expr = Mehrotra.Symbolics.build_function(c, r, Δza, Δsa, κ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    cc_expr = Mehrotra.Symbolics.build_function(cc, r, Δza, Δsa, x, κ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    s_expr = Mehrotra.Symbolics.build_function(Δs, Δz, x, rs,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]

    return f_expr, fc_expr, fx_expr, fxc_expr, fθ_expr, fk_expr, c_expr, cc_expr, s_expr, fx_sparsity, fxc_sparsity, fθ_sparsity, fk_indices
end
