function module_dir()
    return joinpath(@__DIR__, "..")
end

function normalize_A!(A::Matrix; ϵ=1e-6)
	nh = size(A,1)
	for i = 1:nh
		A[i,:] ./= norm(A[i,:]) + ϵ
	end
	return nothing
end
