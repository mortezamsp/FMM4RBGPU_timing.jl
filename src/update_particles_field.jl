export BruteForce
export FMM, FMMGPU
export update_particles_field!

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

# Timing results structures
struct TimingResults
    collection_time::Float64
    M2L_transfer_time::Float64
    M2L_computation_time::Float64
    M2L_time::Float64
    P2P_transfer_time::Float64
    P2P_computation_time::Float64
    P2P_time::Float64
    Update_time::Float64
	min_n::Int
	mean_n::Int
	max_n::Int
end

struct TimingResults_CPU
    collection_time::Float64
    M2L_time::Float64
    P2P_time::Float64
    Update_time::Float64
	max_level::Int
	num_clus::Int
end


# Existing BruteForce method
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
    return nothing
end

# FMM CPU method
function update_particles_field!(particles::Particles{T}, alg::FMM; lambda) where {T}
    (;n, N0, eta) = alg
   
    #println("begin CPU function...")

    # Measure collection time
    q = particles.charge
    N = particles.npar
    p_avg = sum(particles.momenta) / particles.npar
    g_avg = sqrt(1.0 + dot(p_avg, p_avg))
    stretch = SVector(1.0,1.0,g_avg)
    max_level = maxlevel(N, N0)
    #println("Max level = $max_level")
    ct = ClusterTree(particles; N0=N0, stretch=stretch)
	num_clus = length(ct.clusters.parlohis)
    #println("ClusterTree created: $num_clus clusters")
    mp = MacroParticles(ct.clusters, n)
    #println("MacroParticles created")
	
	upwardpass!(mp, ct; max_level=max_level)
    #println("Upward pass completed")
	
    start_time = Dates.now()
    itlists = InteractionLists(ct.clusters; stretch=stretch, eta=eta)
	end_time = Dates.now()
    collection_time = Float64(Dates.value(end_time - start_time)) 
	PartialTimingResults = interact!(mp, ct, itlists; p_avg=p_avg)
    downwardpass!(mp, ct; max_level=max_level)
    
    # Update time
    start_time = Dates.now()
    efields = particles.efields
    bfields = particles.bfields
    amp = 2.8179403699772166e-15 * q / lambda
    @inbounds for i in 1:N
        efields[i] *= amp
        bfields[i] *= amp
    end
    end_time = Dates.now()
    Update_time = Float64(Dates.value(end_time - start_time)) 

    return TimingResults_CPU(collection_time, PartialTimingResults.M2L_time, PartialTimingResults.P2P_time, Update_time, max_level, num_clus)
end

# FMM GPU method
function update_particles_field!(particles::Particles{T}, alg::FMMGPU; lambda) where {T}
    (;n, N0, eta) = alg

    q = particles.charge
    N = particles.npar

    # Collection time
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

    # Upward pass (placeholder)
    @cuda blocks=nleafnode threads=(n+1,n+1,n+1) gpu_P2M!(d_pr_positions, d_pr_momenta, d_ct_parindices, d_mp_gammas, d_mp_momenta, Val(n), d_cl_bboxes, d_cl_parlohis, lfindices)

    for l in (max_level-1):-1:0
        nodeindicies = nodeindexrangeat(l)
        nc_in_level = 2^l
        @cuda blocks=nc_in_level threads=(n+1,n+1,n+1) gpu_M2M!(d_mp_gammas, d_mp_momenta, Val(n), d_cl_bboxes, d_cl_children, nodeindicies)
    end

	#extracting the interactions list
    start_time = Dates.now()
    itlists_gpu = InteractionListsGPU(ct.clusters; stretch=stretch, eta=eta)
	end_time = Dates.now()
    collection_time = Float64(Dates.value(end_time - start_time))
	
    # M2L transfer time
    start_time = Dates.now()
    d_m2l_lists = CuArray(itlists_gpu.m2l_lists)
    d_m2l_lists_ptrs = CuArray(itlists_gpu.m2l_lists_ptrs)
    nm2lgroup = itlists_gpu.nm2lgroup
    end_time = Dates.now()
    M2L_transfer_time = Float64(Dates.value(end_time - start_time)) 

    # M2L computation time
    start_time = Dates.now()
    @cuda blocks=nm2lgroup threads=(n+1,n+1,n+1) gpu_M2L!(d_mp_gammas, d_mp_momenta, d_mp_efields, d_mp_bfields, Val(n), d_cl_bboxes, d_m2l_lists, d_m2l_lists_ptrs, p_avg)
    end_time = Dates.now()
    M2L_computation_time = Float64(Dates.value(end_time - start_time)) 
    M2L_time = M2L_transfer_time + M2L_computation_time

    # P2P transfer time
    start_time = Dates.now()
    d_p2p_lists = CuArray(itlists_gpu.p2p_lists)
    d_p2p_lists_ptrs = CuArray(itlists_gpu.p2p_lists_ptrs)
    np2pgroup = itlists_gpu.np2pgroup
    end_time = Dates.now()
    P2P_transfer_time = Float64(Dates.value(end_time - start_time)) 

    # P2P computation time
    start_time = Dates.now()
    @cuda blocks=np2pgroup threads=N0 gpu_P2P!(d_pr_positions, d_pr_momenta, d_pr_efields, d_pr_bfields, d_ct_parindices, d_cl_parlohis, Val(N0), d_p2p_lists, d_p2p_lists_ptrs)
    end_time = Dates.now()
    P2P_computation_time = Float64(Dates.value(end_time - start_time))
    P2P_time = P2P_transfer_time + P2P_computation_time

    # Downward pass (placeholder)
    for l in 1:max_level
        nodeindicies = nodeindexrangeat(l)
        nc_in_level = 2^l
        @cuda blocks=nc_in_level threads=(n+1,n+1,n+1) gpu_L2L!(d_mp_efields, d_mp_bfields, Val(n), d_cl_bboxes, d_cl_parents, nodeindicies)
    end

    @cuda blocks=nleafnode threads=N0 gpu_L2P!(d_pr_positions, d_pr_efields, d_pr_bfields, d_ct_parindices, d_mp_efields, d_mp_bfields, Val(n), d_cl_bboxes, d_cl_parlohis, lfindices)

    # Update time
    start_time = Dates.now()
    amp = 2.8179403699772166e-15 * q / lambda
    d_pr_efields .*= amp
    d_pr_bfields .*= amp
    copyto!(particles.efields, d_pr_efields)
    copyto!(particles.bfields, d_pr_bfields)
    end_time = Dates.now()
    Update_time = Float64(Dates.value(end_time - start_time))

    return TimingResults(collection_time, M2L_transfer_time, M2L_computation_time, M2L_time, P2P_transfer_time, P2P_computation_time, P2P_time, Update_time, itlists_gpu.min_n, itlists_gpu.mean_n, itlists_gpu.max_n)
end

#end # module