using Revise
using .FMM4RBGPU_timing
using CUDA
using Dates
using DataFrames
using CSV

# Define TimingResults structs based on what the package actually returns

struct TimingResults
    collection_time::Int
    M2L_transfer_time::Int
    M2L_computation_time::Int
    P2P_transfer_time::Int
    P2P_computation_time::Int
    Update_time::Int
	avg_neis::Float64
	m2l_size::Int
	p2p_size::Int
end

struct TimingResults_CPU
    collection_time::Int
    M2L_time::Int
    P2P_time::Int
    Update_time::Int
	max_level::Int
	num_clus::Int
end
TimingResults_CPU() = TimingResults_CPU(0, 0, 0, 0, 0, 0)
		
function main()
    ## Let's first check what's available
    #println("Available methods for update_particles_field!:")
    #println(methods(update_particles_field!))


    # Parameters (using smaller values for testing)
    N_values = [2^16, 2^17, 2^18, 2^19, 2^20]
	n_values = [3, 4, 5, 6, 7]
	eta_values = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
	filename = "fmm_experiment_results.csv"

    # Initialize DataFrame
    df = DataFrame(
        experiment_num = Int64[],
        N = Int64[],
        n = Int64[],
        N0 = Int64[],
        eta = Float64[],
		max_level = Int64[],
		num_clus = Int64[],
		min_nei = Float64[],
		#mean_nei = Float64[],
		#max_nei = Int64[],
        gpu_time = Int64[],
        cpu_time = Int64[],
        speedup = Float64[],
        # GPU timing details
        gpu_collection_time = Int64[],
        gpu_M2L_transfer_time = Int64[],
        gpu_M2L_computation_time = Int64[],
        gpu_M2L_time = Int64[],
        gpu_P2P_transfer_time = Int64[],
        gpu_P2P_computation_time = Int64[],
        gpu_P2P_time = Int64[],
        gpu_Update_time = Int64[],
        # CPU timing details
        cpu_collection_time = Int64[],
        cpu_M2L_time = Int64[],
        cpu_P2P_time = Int64[],
        cpu_Update_time = Int64[],
        total_gpu_time = Int64[]
    )

    experiment_num = 1

    # If FMMGPU doesn't work, let's try a fallback approach
    function safe_run_experiment(experiment_num, N, n, eta)
		println("Experiment $experiment_num: Setting up N=$N, n=$n, N0=$((n+1)^3), eta=$eta")
		
		# Create position and momentum distribution of N particles
		positions = rand(3, N)
		#println(positions)
		momenta = zeros(3, N)
		
		# Create particle beam
		beam = Particles(; pos=positions, mom=momenta, charge=-1.0, mass=1.0)
		
		# Try GPU execution
		gpu_success = false
		total_gpu_time = 0
		gpuresult = nothing
		gpu_timing_results = TimingResults(0, 0, 0, 0, 0, 0, 0.0, 0, 0)
		
		try
			println("  Starting GPU execution...")
			start_time = time_ns()
			gpuresult = update_particles_field!(beam, FMMGPU(eta=eta, N0=(n+1)^3, n=n); lambda=1.0)
			end_time = time_ns()
			total_gpu_time = end_time - start_time
			gpu_success = true
			println("$total_gpu_time");

			# Validate GPU results
			if gpuresult === nothing
				println("  GPU returned nothing")
				gpu_success = false
			else
				println("  GPU execution successful")#: $gpuresult")
			end
		catch e
			println("  GPU execution failed: ", e)
			# Create proper dummy timing results based on what type is expected
			gpuresult = TimingResults(0, 0, 0, 0, 0, 0, 0.0, 0, 0)
		end
		
		# Recreate beam for CPU (important to start fresh)
		gpu_timing_results = TimingResults( 
			Int(gpuresult[1]), Int(gpuresult[2]), Int(gpuresult[3]), Int(gpuresult[4]), Int(gpuresult[5]), Int(gpuresult[6]), Float64(gpuresult[7]), Int(gpuresult[8]), Int(gpuresult[9]))
		println("regenerating beams...");
		beam_cpu = Particles(; pos=positions, mom=momenta, charge=-1.0, mass=1.0)
		println("done");
		
		# CPU execution time
		cpureslts = nothing
		cpu_timing_results = TimingResults_CPU(0, 0, 0, 0, 0, 0)
		cpu_time = 0.0
		cpu_success = false
		
		
		try
			println("  Starting CPU execution...")
			start_time = time_ns()
			cpureslts = update_particles_field!(beam_cpu, FMM(eta=eta, N0=(n+1)^3, n=n); lambda=1.0)
			end_time = time_ns()
			cpu_time = end_time - start_time
			cpu_success = true
			println("$cpu_time")
			
			# Validate CPU results
			if cpureslts === nothing
				println("  CPU returned nothing")
				cpu_success = false
			else
				println("  CPU execution successful: $cpureslts")
			end
		catch e
			println("  CPU execution failed: ", e)
			# Create proper dummy timing results
			cpureslts = TimingResults_CPU(0, 0, 0, 0, 0, 0)
		end
		cpu_timing_results = TimingResults_CPU(Int(cpureslts[1]), Int(cpureslts[2]), Int(cpureslts[3]), Int(cpureslts[4]), Int(cpureslts[5]), Int(cpureslts[6]))
		
		# Calculate speedup (if both succeeded)
		speedup = (gpu_success && cpu_success && cpu_time > 0) ? Float64(cpu_time) / Float64(total_gpu_time) : 0.0
		#println("$speedup")
		
		return total_gpu_time, cpu_time, speedup, gpu_timing_results, cpu_timing_results, gpu_success
	end

    # Run experiments with safe wrapper
    for N in N_values
        println("\n=== Running experiments for N = $N ===")
        
        for n in n_values
            N0 = (n + 1)^3
            
            for eta in eta_values
                total_gpu_time, cpu_time, speedup, gpu_timing_results, cpu_timing_results, gpu_success = safe_run_experiment(experiment_num, N, n, eta)                
                println("Exp $experiment_num | N=$N | n=$n | N0=$N0 | eta=$eta | GPU: $(total_gpu_time)s | CPU: $(cpu_time)s | Speedup: $(round(speedup, digits=2))X")
                
				println("creating tupple...")
				new_row = (
					experiment_num = experiment_num,
					N = N,
					n = n,
					N0 = N0,
					eta = eta,
					max_level = cpu_timing_results.max_level,
					num_clus = cpu_timing_results.num_clus,
					min_nei = gpu_timing_results.avg_neis, #min_nei = avg_nei = max_nei
					#mean_nei = gpu_timing_results.mean_n,
					#max_nei = gpu_timing_results.max_n,
					gpu_time = total_gpu_time,
					cpu_time = cpu_time,
					speedup = speedup,
					# GPU timing details
					gpu_collection_time = gpu_timing_results.collection_time,
					gpu_M2L_transfer_time = gpu_timing_results.M2L_transfer_time,
					gpu_M2L_computation_time = gpu_timing_results.M2L_computation_time,
					gpu_M2L_time = gpu_timing_results.M2L_transfer_time + gpu_timing_results.M2L_computation_time,
					gpu_P2P_transfer_time = gpu_timing_results.P2P_transfer_time,
					gpu_P2P_computation_time = gpu_timing_results.P2P_computation_time,
					gpu_P2P_time = gpu_timing_results.P2P_transfer_time + gpu_timing_results.P2P_computation_time,
					gpu_Update_time = gpu_timing_results.Update_time,
					# CPU timing details
					cpu_collection_time = cpu_timing_results.collection_time,
					cpu_M2L_time = cpu_timing_results.M2L_time,
					cpu_P2P_time = cpu_timing_results.P2P_time,
					cpu_Update_time = cpu_timing_results.Update_time,
					total_gpu_time = total_gpu_time
				)
                println("adding to dataframe...")
                push!(df, new_row)
                println("inserting to csv file...")
                CSV.write(filename, df)
                
                experiment_num += 1
            end
        end
    end

    println("\n=== Test completed ===")
    println("Results saved to $filename")
end

# Run the main function
main()