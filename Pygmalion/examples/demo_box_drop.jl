include(joinpath(module_dir(), "Pygmalion/Pygmalion.jl"))
include("../flux/shape_loss.jl")


vis = Visualizer()
# open(vis)
render(vis)
set_background!(vis)
set_light!(vis, direction="Negative")
set_floor!(vis, color=RGBA(0.4,0.4,0.4,0.4))
iterate_color = RGBA(1,1,0,0.6);

Ap0 = [
	+1.0 +0.0;
	+0.0 +1.0;
	-1.0 +0.0;
	+0.0 -1.0;
	]
normalize_A!(Ap0)
bp0 = 0.5*[
	+1,
	+1,
	+1,
	+1,
	]
op0 = [0.0, 0.0]
Af = [0.0  +1.0]
bf = [0.0]
of = [0.0]

Ap = [Ap0, Af]
bp = [bp0, bf]
op = [op0, of]
θ_p, polytope_dimensions_p = pack_halfspaces(Ap, bp, op)

# build_2d_polytope!(vis[:polytope], Ap0, bp0 + Ap0 * op0, name=:poly0, color=RGBA(1,1,1,1.0))

################################################################################
# inertial parameters
################################################################################
timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);

mech = get_polytope_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=0.1,
	method_type=:symbolic,
    # method_type=:finite_difference,
	A=Ap0,
	b=bp0,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-3,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=false,
        differentiate=false,
        warm_start=false,
        complementarity_correction=0.5,
        )
    );
Mehrotra.solve!(mech.solver)

################################################################################
# test simulation
################################################################################
vp15 = [+3.0, +0.0, +2.0]
xp2  = [+0.0, +0.75, +0.0]
z0 = [xp2; vp15]

H0 = 15
storage = simulate!(mech, z0, H0)
vis, anim = visualize!(vis, mech, storage)


configuration_ref = [[storage.x[i][1]; storage.v[i][1]] for i = 1:H0]
JLD2.save(joinpath(@__DIR__, "deps/configuration_ref_$H0.jld2"), "configuration_ref", configuration_ref)



mech.solver.parameters[14:end]
mech.bodies
mech.contacts
storage.variables



################################################################################
# initial guess
################################################################################
Ap1 = [
	+1.0 +0.0;
	+0.0 +1.0;
	-1.0 +0.0;
	+0.0 -1.0;
	]
bp1 = 0.75*[
	+1,
	+1,
	+1,
	+1,
	]
friction_coefficient1 = 0.5

configuration_init = [[deepcopy(z0); zeros(12+13)] for i = 1:H0]
for i = 2:H0
	mech.solver.parameters[14] = friction_coefficient1
	mech.solver.parameters[end-12-3+1:end-3] .= [vec(Ap1); bp1]
	update_nodes!(mech)

	storage_init = simulate!(mech, configuration_init[i-1][1:6], 2)
	configuration_init[i] .= [
		deepcopy(storage_init.x[2][1]);
		deepcopy(storage_init.variables[1][1:15]);
		deepcopy(vec(Ap1));
		deepcopy(bp1);
		friction_coefficient1;
		]
end

configuration_init[1][7:end] .= configuration_init[2][7:end]
JLD2.save(joinpath(@__DIR__, "deps/configuration_init_$H0.jld2"), "configuration_init", configuration_init)

plot(hcat(configuration_init...)'[:,1:3])
