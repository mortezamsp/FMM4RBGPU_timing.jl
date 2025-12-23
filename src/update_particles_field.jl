export BruteForce
export FMM, FMMGPU
export update_particles_field!
import Dates

import Base.@kwdef

struct BruteForce end

@kwdef struct FMM{T}
    n::Int
    N0::Int
    eta::T
end

@kwdef struct FMMGPU{T}
    n::Int
    N0::Int
    eta::T
end

function update_particles_field!(particles::Particles{T}, alg::BruteForce; lambda) where {T}
    q = particles.charge
    npar = particles.npar
    @inbounds for i in 1:npar
        particles.efields[i] = SVector(0.0, 0.0, 0.0)
        particles.bfields[i] = SVector(0.0, 0.0, 0.0)
        xi = particles.positions[i]
        amp = 2.8179403699772166e-15 * q / lambda
        for j in 1:npar
            xj = particles.positions[j]
            pj = particles.momenta[j]
            R = xi-xj
            Kij = R / sqrt(dot(R, R) + dot(pj, R)^2 + eps())^3
            particles.efields[i] += amp * sqrt(1.0 + dot(pj, pj)) * Kij
            particles.bfields[i] += amp * cross(pj, Kij)
        end
    end
end

function update_particles_field!(particles::Particles{T}, alg::FMM; lambda) where {T}
    (;n, N0, eta) = alg

    begin_time = time_ns()	
	q = particles.charge
    N = particles.npar
    p_avg = sum(particles.momenta) / particles.npar
    g_avg = sqrt(1.0 + dot(p_avg, p_avg))
    stretch = SVector(1.0,1.0,g_avg)
    max_level = maxlevel(N, N0)
    ct = ClusterTree(particles; N0=N0, stretch=stretch)
    mp = MacroParticles(ct.clusters, n)
	end_time = time_ns()
	collection_time = end_time - begin_time
	
    upwardpass!(mp, ct; max_level=max_level)
    itlists = InteractionLists(ct.clusters; stretch=stretch, eta=eta)
    M2L_time, P2P_time = interact!(mp, ct, itlists; p_avg=p_avg)
    downwardpass!(mp, ct; max_level=max_level)

	begin_time = time_ns()
    efields = particles.efields
    bfields = particles.bfields
    amp = 2.8179403699772166e-15 * q / lambda
    @inbounds for i in 1:N
        efields[i] *= amp
        bfields[i] *= amp
    end
	end_time = time_ns()
	update_time = end_time - begin_time
	
    num_clus = length(ct.clusters.parlohis)
	
	return collection_time, M2L_time, P2P_time, update_time, max_level, num_clus
end

function update_particles_field!(particles::Particles{T}, alg::FMMGPU; lambda) where {T}
    (;n, N0, eta) = alg

    q = particles.charge
    N = particles.npar

    d_pr_positions = CuArray(particles.positions)
    d_pr_momenta = CuArray(particles.momenta)
    d_pr_efields = CUDA.fill(SVector{3,T}(0.0,0.0,0.0),N)
    d_pr_bfields = CUDA.fill(SVector{3,T}(0.0,0.0,0.0),N)

    p_avg = reduce(+, d_pr_momenta) / N
    g_avg = sqrt(1.0 + dot(p_avg, p_avg))
    stretch = SVector(1.0,1.0,g_avg)
    
    nc = ncluster(N,N0)
    ct = ClusterTree(particles; N0=N0, stretch=stretch)

    d_ct_parindices = CuArray(ct.parindices)

    d_cl_parlohis = CuArray(ct.clusters.parlohis)
    d_cl_parents = CuArray(ct.clusters.parents)
    d_cl_children = CuArray(ct.clusters.children)
    d_cl_bboxes = CuArray(ct.clusters.bboxes)

    d_mp_gammas = CUDA.fill(zero(T),n+1,n+1,n+1,nc)
    d_mp_momenta = CUDA.fill(SVector{3,T}(0.0,0.0,0.0),n+1,n+1,n+1,nc)
    d_mp_efields = CUDA.fill(SVector{3,T}(0.0,0.0,0.0),n+1,n+1,n+1,nc)
    d_mp_bfields = CUDA.fill(SVector{3,T}(0.0,0.0,0.0),n+1,n+1,n+1,nc)

    max_level = maxlevel(N,N0)
    lfindices = leafindexrange(N, N0)
    nleafnode = length(lfindices)

    # upwardpass
    @cuda blocks=nleafnode threads=(n+1,n+1,n+1) gpu_P2M!(d_pr_positions, d_pr_momenta, d_ct_parindices,
                                                            d_mp_gammas, d_mp_momenta, Val(n),
                                                            d_cl_bboxes, d_cl_parlohis, lfindices)

    for l in (max_level-1):-1:0
        nodeindicies = nodeindexrangeat(l)
        nc_in_level = 2^l
        @cuda blocks=nc_in_level threads=(n+1,n+1,n+1) gpu_M2M!(d_mp_gammas, d_mp_momenta, Val(n),
                                                            d_cl_bboxes, d_cl_children, nodeindicies)
    end

    # dual-tree traversal
	start_time = time_ns()
    itlists_gpu = InteractionListsGPU(ct.clusters; stretch=stretch, eta=eta)
    end_time = time_ns()
	data_collection_time = end_time - start_time
	#avg_neis = itlists_gpu.avg_neis
	min_neis = itlists_gpu.min_neis
	m2l_size = length(itlists_gpu.m2l_lists)
	p2p_size = length(itlists_gpu.p2p_lists)
	
	CUDA.synchronize()
	start_time = end_time
    d_m2l_lists = CuArray(itlists_gpu.m2l_lists)
    d_m2l_lists_ptrs = CuArray(itlists_gpu.m2l_lists_ptrs)
    nm2lgroup = itlists_gpu.nm2lgroup
	end_time = time_ns()
	M2L_data_transfer = end_time - start_time
	
    # interaction
	start_time = end_time
    @cuda blocks=nm2lgroup threads=(n+1,n+1,n+1) gpu_M2L!(d_mp_gammas, d_mp_momenta, d_mp_efields, d_mp_bfields, Val(n),
                                                        d_cl_bboxes, d_m2l_lists, d_m2l_lists_ptrs,
                                                        p_avg)
	CUDA.synchronize()
	end_time = time_ns()
	GPU_M2L_time = end_time - start_time
	
	start_time = end_time
    d_p2p_lists = CuArray(itlists_gpu.p2p_lists)
    d_p2p_lists_ptrs = CuArray(itlists_gpu.p2p_lists_ptrs)
    np2pgroup = itlists_gpu.np2pgroup
	P2P_data_transfer = end_time - start_time
	end_time = time_ns()
	
	start_time = end_time
    @cuda blocks=np2pgroup threads=N0 gpu_P2P!(d_pr_positions, d_pr_momenta, d_pr_efields, d_pr_bfields, d_ct_parindices,
                                            d_cl_parlohis, Val(N0), d_p2p_lists, d_p2p_lists_ptrs)
	CUDA.synchronize()
	end_time = time_ns()
	GPU_P2P_time = end_time - start_time
	
    # downwardpass
    for l in 1:max_level
        nodeindicies = nodeindexrangeat(l)
        nc_in_level = 2^l
        @cuda blocks=nc_in_level threads=(n+1,n+1,n+1) gpu_L2L!(d_mp_efields, d_mp_bfields, Val(n),
                                                                d_cl_bboxes, d_cl_parents, nodeindicies)
    end

    @cuda blocks=nleafnode threads=N0 gpu_L2P!(d_pr_positions, d_pr_efields, d_pr_bfields, d_ct_parindices,
                                                d_mp_efields, d_mp_bfields, Val(n),
                                                d_cl_bboxes, d_cl_parlohis, lfindices)

    amp = 2.8179403699772166e-15 * q / lambda
    d_pr_efields .*= amp
    d_pr_bfields .*= amp

    CUDA.synchronize()
	start_time = time_ns()
	copyto!(particles.efields, d_pr_efields)
    copyto!(particles.bfields, d_pr_bfields)
	end_time = time_ns()
	update_time = end_time - start_time
	
	#return data_collection_time, M2L_data_transfer, GPU_M2L_time, P2P_data_transfer, GPU_P2P_time, update_time, avg_neis, m2l_size, p2p_size
	return data_collection_time, M2L_data_transfer, GPU_M2L_time, P2P_data_transfer, GPU_P2P_time, update_time, min_neis, m2l_size, p2p_size
end
