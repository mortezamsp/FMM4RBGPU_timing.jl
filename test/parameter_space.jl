using Revise
using .FMM4RBGPU_timing
using CUDA
using Dates
using DataFrames
using CSV

# Define TimingResults structs based on what the package actually returns
struct TimingResults
	collection_time::Float64
	M2L_transfer_time::Float64
	M2L_computation_time::Float64
	M2L_time::Float64
	P2P_transfer_time::Float64
	P2P_computation_time::Float64
	P2P_time::Float64
	Update_time::Float64
end

struct TimingResults_CPU
	collection_time::Float64
	M2L_time::Float64
	P2P_time::Float64
	Update_time::Float64
end

function main()
    ## Let's first check what's available
    #println("Available methods for update_particles_field!:")
    #println(methods(update_particles_field!))


    # Parameters (using smaller values for testing)
    N_values = [2^16]  # Start with just one value for testing
    n_values = [3]     # Start with just one value for testing  
    eta_values = [0.2] # Start with just one value for testing
    filename = "fmm_experiment_results.csv"

    # Initialize DataFrame
    df = DataFrame(
        experiment_num = Int64[],
        N = Int64[],
        n = Int64[],
        N0 = Int64[],
        eta = Float64[],
        gpu_time = Float64[],
        cpu_time = Float64[],
        speedup = Float64[],
        # GPU timing details
        gpu_collection_time = Float64[],
        gpu_M2L_transfer_time = Float64[],
        gpu_M2L_computation_time = Float64[],
        gpu_M2L_time = Float64[],
        gpu_P2P_transfer_time = Float64[],
        gpu_P2P_computation_time = Float64[],
        gpu_P2P_time = Float64[],
        gpu_Update_time = Float64[],
        # CPU timing details
        cpu_collection_time = Float64[],
        cpu_M2L_time = Float64[],
        cpu_P2P_time = Float64[],
        cpu_Update_time = Float64[],
        total_gpu_time = Float64[]
    )

    experiment_num = 1

    # If FMMGPU doesn't work, let's try a fallback approach
    function safe_run_experiment(experiment_num, N, n, eta)
        println("Experiment $experiment_num: Setting up N=$N, n=$n, N0=$((n+1)^3), eta=$eta")
        
        # Create position and momentum distribution of N particles
        positions = rand(3, N)
        momenta = zeros(3, N)
        
        # Create particle beam
        beam = Particles(; pos=positions, mom=momenta, charge=-1.0, mass=1.0)
        
        # Try GPU execution
        gpu_success = false
        total_gpu_time = 0.0
        gpu_timing_results = nothing
        
        try
            start_time = Dates.now()
            gpu_timing_results = update_particles_field!(beam, FMMGPU(eta=eta, N0=(n+1)^3, n=n); lambda=1.0)
            end_time = Dates.now()
            total_gpu_time = Float64(Dates.value(end_time - start_time)) / 1000.0
            gpu_success = true
            println("GPU execution successful")
        catch e
            println("GPU execution failed: ", e)
            # Create dummy timing results
            gpu_timing_results = TimingResults(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        end
        
        # Recreate beam for CPU
        beam_cpu = Particles(; pos=positions, mom=momenta, charge=-1.0, mass=1.0)
        
        # CPU execution time
        cpu_timing_results = nothing
        cpu_time = 0.0
        
        try
            start_time = Dates.now()
            cpu_timing_results = update_particles_field!(beam_cpu, FMM(eta=eta, N0=(n+1)^3, n=n); lambda=1.0)
            end_time = Dates.now()
            cpu_time = Float64(Dates.value(end_time - start_time)) / 1000.0
            println("CPU execution successful")
        catch e
            println("CPU execution failed: ", e)
            # Create dummy timing results
            cpu_timing_results = TimingResults_CPU(0.0, 0.0, 0.0, 0.0)
        end
        
        # Calculate speedup (if both succeeded)
        speedup = gpu_success ? (cpu_time > 0 ? cpu_time / total_gpu_time : 0.0) : 0.0
        
        return total_gpu_time, cpu_time, speedup, gpu_timing_results, cpu_timing_results, gpu_success
    end

    # Run experiments with safe wrapper
    for N in N_values
        println("\n=== Running experiments for N = $N ===")
        
        for n in n_values
            N0 = (n + 1)^3
            
            for eta in eta_values
                total_gpu_time, cpu_time, speedup, gpu_timing_results, cpu_timing_results, gpu_success = safe_run_experiment(experiment_num, N, n, eta)
                
                # Create new row with all timing data
                new_row = (
                    experiment_num = experiment_num,
                    N = N,
                    n = n,
                    N0 = N0,
                    eta = eta,
                    gpu_time = total_gpu_time,
                    cpu_time = cpu_time,
                    speedup = speedup,
                    # GPU timing details
                    gpu_collection_time = gpu_timing_results.collection_time,
                    gpu_M2L_transfer_time = gpu_timing_results.M2L_transfer_time,
                    gpu_M2L_computation_time = gpu_timing_results.M2L_computation_time,
                    gpu_M2L_time = gpu_timing_results.M2L_time,
                    gpu_P2P_transfer_time = gpu_timing_results.P2P_transfer_time,
                    gpu_P2P_computation_time = gpu_timing_results.P2P_computation_time,
                    gpu_P2P_time = gpu_timing_results.P2P_time,
                    gpu_Update_time = gpu_timing_results.Update_time,
                    # CPU timing details
                    cpu_collection_time = cpu_timing_results.collection_time,
                    cpu_M2L_time = cpu_timing_results.M2L_time,
                    cpu_P2P_time = cpu_timing_results.P2P_time,
                    cpu_Update_time = cpu_timing_results.Update_time,
                    total_gpu_time = total_gpu_time
                )
                
                println("Exp $experiment_num | N=$N | n=$n | N0=$N0 | eta=$eta | GPU: $(round(total_gpu_time, digits=4))s | CPU: $(round(cpu_time, digits=4))s | Speedup: $(round(speedup, digits=2))X")
                
                push!(df, new_row)
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