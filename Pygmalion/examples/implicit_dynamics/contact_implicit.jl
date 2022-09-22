function unpack_contact_implicit_variables(x, mechanism::Mechanism;
		# A::Bool=true,
		# b::Bool=true,
		# friction_coefficient::Bool=true,
		)

	NC = 1
	off = 0

	# body
	body = mechanism.bodies[1]
	num_variables = length(body.index.variables)
	p3 = x[off .+ (1:num_variables)]; off += num_variables
	v25 = x[off .+ (1:num_variables)]; off += num_variables
	body_variables = [p3, v25]

	# contacts
	contact_variables = []
	for contact in mechanism.contacts
		NP = length(contact.b_parent_collider)
		c = x[off .+ (1:2)]; off += 2
		ϕ = x[off .+ (1:1)]; off += 1

		γ = x[off .+ (1:1)]; off += 1
		ψ = x[off .+ (1:1)]; off += 1
		β = x[off .+ (1:2)]; off += 2
		λp = x[off .+ (1:NP)]; off += NP
		λc = x[off .+ (1:NC)]; off += NC
		push!(contact_variables, [c, ϕ, γ, ψ, β, λp, λc])
	end

	contact_parameters = []
	friction_coefficient = x[off .+ (1:1)]; off += 1
	for contact in mechanism.contacts
		NP = length(contact.b_parent_collider)
		A = x[off .+ (1:2NP)]; off += 2NP
		A = reshape(A, (NP,2))
		b = x[off .+ (1:NP)]; off += NP
		push!(contact_parameters, [friction_coefficient, A, b])
	end

	return body_variables, contact_variables, contact_parameters
end

function polytope_dynamics(y, x, u, mechanism;
        timestep=0.05,
        gravity=-10.0,
        mass=1.0,
        inertia=0.2,
        Ac=[0 1.0],
        bc=[0.0],
        )

    # unpack
	body_variables_2, contact_variables_2, contact_parameters_2 = unpack_contact_implicit_variables(x, mechanism)
	body_variables_3, contact_variables_3, contact_parameters_3 = unpack_contact_implicit_variables(y, mechanism)

	p2, v15 = body_variables_2
	p3, v25 = body_variables_3

	c, ϕ, γ, ψ, β, λp, λc = contact_variables_3[1]

	friction_coefficient_2, A2, b2 = contact_parameters_2[1]
	friction_coefficient_3, A3, b3 = contact_parameters_3[1]

    # mass matrix
    M = Diagonal([mass; mass; inertia])

    # contact points
    pp3 = p3
    pc3 = zeros(3)
	wrench_p = poly_halfspace_wrench(pp3, pc3, c, γ, β, λp, A3)
	p1 = p2 - timestep * v15

    res = [
        # dynamics
        M ./ timestep * (p3 - 2p2 + p1) - timestep .* [0; mass * gravity; 0] - wrench_p;

		# integrator
		p2 + timestep * v25 - p3;

		# contact optimality
        x_2d_rotation(pp3[3:3]) * A3' * λp + x_2d_rotation(pc3[3:3]) * Ac' * λc;
        1 - sum(λp) - sum(λc);

        # parameter consistency
		friction_coefficient_2 - friction_coefficient_3;
        vec(A2 - A3);
        b2 - b3;
    ]
    return res
end

function poly_halfspace_frames(pp3, pc3, c, λp, A3)
	# contact position in the world frame
	contact_w = c + pp3[1:2]
	# contact_p is expressed in pbody's frame
	contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])
	# contact_c is expressed in cbody's frame
	contact_c = x_2d_rotation(pc3[3:3])' * (contact_w - pc3[1:2])

	# contact normal and tangent in the world frame
	normal_pw = -x_2d_rotation(pp3[3:3]) * A3' * λp
	R = [0 1; -1 0]
	tangent_pw = R * normal_pw

	# rotation matrix from contact frame to world frame
	wRp = [tangent_pw normal_pw] # n points towards the parent body, [t,n,z] forms an oriented vector basis

	return contact_p, contact_c, contact_w, tangent_pw, wRp
end

function poly_halfspace_wrench(pp3, pc3, c, γ, β, λp, A3)
	contact_p, contact_c, contact_w, tangent_pw, wRp = poly_halfspace_frames(pp3, pc3, c, λp, A3)

	# force at the contact point in the contact frame
	f = [β[1] - β[2]; γ]
	# force at the contact point in the world frame
	f_pw = +wRp * f # parent
	# torques at the centers of masses in world frame
	τ_pw = (skew([contact_w - pp3[1:2]; 0]) * [f_pw; 0])[3:3]
	# overall wrench on both bodies in world frame
	# mapping the contact force into the generalized coordinates (at the centers of masses and in the world frame)
	wrench_p = [f_pw; τ_pw]

	return wrench_p
end

function poly_halfspace_slackness(x, mechanism;
        Ac=[0 1.0],
        bc=[0.0],
        )

	# unpack
	body_variables_3, contact_variables_3, contact_parameters_3 = unpack_contact_implicit_variables(x, mechanism)
	p3, v25 = body_variables_3
	c, ϕ, γ, ψ, β, λp, λc = contact_variables_3[1]
	friction_coefficient_3, A3, b3 = contact_parameters_3[1]

	pp3 = p3
    pc3 = zeros(3)

	nh = length(b3)
	contact_p, contact_c, contact_w, tangent_pw, wRp = poly_halfspace_frames(pp3, pc3, c, λp, A3)

    # tangential velocities at the contact point
    tanvel_p = v25[1:2] + (skew([pp3[1:2] - contact_w; 0]) * [zeros(2); v25[3]])[1:2]
    tanvel_p = tanvel_p' * tangent_pw
    tanvel = tanvel_p

    return [
        ϕ;
        (friction_coefficient_3[1] * γ - [sum(β)]);
        ([+tanvel; -tanvel] + ψ[1]*ones(2));
        (- A3 * contact_p + b3 + ϕ .* ones(nh));
        (- Ac * contact_c + bc + ϕ .* ones(1));
    ]
end
