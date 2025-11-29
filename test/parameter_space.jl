using Revise
using FMM4RBGPU_timing
using CUDA
using Dates
using DataFrames
using CSV

# Define TimingResults structs (make sure these match what's returned from update_particles_field!)
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

# Parameters
N_values = [2^16, 2^17, 2^18, 2^19, 2^20]
n_values = [3, 4, 5, 6, 7]
eta_values = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
filename = "fmm_experiment_results.csv"

# Initialize or load existing DataFrame
if isfile(filename)
    println("Loading existing results from $filename")
    df = CSV.read(filename, DataFrame)
    
    # Find the last completed experiment
    if nrow(df) > 0
        last_experiment = maximum(df.experiment_num)
        last_N = df.N[end]
        last_n = df.n[end]
        last_eta = df.eta[end]
        
        println("Last experiment: #$(last_experiment) (N=$last_N, n=$last_n, eta=$last_eta)")
        
        # Find the next experiment to run
        global experiment_num = last_experiment + 1
        
        # Find current position in parameter space
        current_N_index = findfirst(==(last_N), N_values)
        current_n_index = findfirst(==(last_n), n_values)
        current_eta_index = findfirst(==(last_eta), eta_values)
        
        # Calculate next indices
        if current_eta_index < length(eta_values)
            # Continue with next eta value
            next_N_index = current_N_index
            next_n_index = current_n_index
            next_eta_index = current_eta_index + 1
        elseif current_n_index < length(n_values)
            # Continue with next n value, reset eta
            next_N_index = current_N_index
            next_n_index = current_n_index + 1
            next_eta_index = 1
        elseif current_N_index < length(N_values)
            # Continue with next N value, reset n and eta
            next_N_index = current_N_index + 1
            next_n_index = 1
            next_eta_index = 1
        else
            # All experiments completed
            println("All experiments already completed!")
            exit()
        end
        
        # Create subsets for remaining experiments
        global remaining_N_values = N_values[next_N_index:end]
        global remaining_n_values = n_values[next_n_index:end]
        global remaining_eta_values = eta_values[next_eta_index:end]
        
    else
        # File exists but is empty
        global df = DataFrame(
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
            cpu_Update_time = Float64[]
        )
        global experiment_num = 1
        global remaining_N_values = N_values
        global remaining_n_values = n_values
        global remaining_eta_values = eta_values
    end
else
    # File doesn't exist, start fresh
    println("Creating new results file: $filename")
    global df = DataFrame(
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
    global experiment_num = 1
    global remaining_N_values = N_values
    global remaining_n_values = n_values
    global remaining_eta_values = eta_values
end

# Function to run a single experiment
function run_experiment(experiment_num, N, n, eta)
    println("Experiment $experiment_num: Setting up N=$N, n=$n, N0=$((n+1)^3), eta=$eta")
    
    # Create position and momentum distribution of N particles
    positions = rand(3, N)
    momenta = zeros(3, N)
    
    # Create particle beam
    beam = Particles(; pos=positions, mom=momenta, charge=-1.0, mass=1.0)
    
    # GPU execution time (total)
    start_time = Dates.now()
    gpu_timing_results = update_particles_field!(beam, FMMGPU(eta=eta, N0=(n+1)^3, n=n); lambda=1.0)
    end_time = Dates.now()
    total_gpu_time = Float64(Dates.value(end_time - start_time)) / 1000.0  # Convert to seconds
    
    # Recreate beam for CPU to ensure same initial conditions
    beam_cpu = Particles(; pos=positions, mom=momenta, charge=-1.0, mass=1.0)
    
    # CPU execution time
    start_time = Dates.now()
    cpu_timing_results = update_particles_field!(beam_cpu, FMM(eta=eta, N0=(n+1)^3, n=n); lambda=1.0)
    end_time = Dates.now()
    cpu_time = Float64(Dates.value(end_time - start_time)) / 1000.0  # Convert to seconds
    
    # Calculate speedup
    speedup = cpu_time / total_gpu_time
    
    return total_gpu_time, cpu_time, speedup, gpu_timing_results, cpu_timing_results
end

# Run remaining experiments
println("\n=== Continuing experiments ===")
global total_experiments = length(remaining_N_values) * length(remaining_n_values) * length(remaining_eta_values)
global completed_count = 0

for N in remaining_N_values
    println("\n=== Running experiments for N = $N ===")
    
    for n in remaining_n_values
        N0 = (n + 1)^3
        
        for eta in remaining_eta_values
            # Skip if this combination was already processed in current iteration
            if completed_count == 0 && 
               N == remaining_N_values[1] && 
               n == remaining_n_values[1] && 
               eta == remaining_eta_values[1] &&
               experiment_num > 1
                # This is the first iteration after resuming, skip the first combination
                # as it was already processed in the previous run
                global completed_count += 1
                global experiment_num += 1
                continue
            end
            
            total_gpu_time, cpu_time, speedup, gpu_timing_results, cpu_timing_results = run_experiment(experiment_num, N, n, eta)
            
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
                cpu_Update_time = cpu_timing_results.Update_time
            )
            
            # Print results in raw text format (without detailed timings)
            println("Exp $experiment_num | N=$N | n=$n | N0=$N0 | eta=$eta | GPU: $(round(total_gpu_time, digits=4))s | CPU: $(round(cpu_time, digits=4))s | Speedup: $(round(speedup, digits=2))X")
            
            # Add to DataFrame
            push!(df, new_row)
            
            # Append to CSV file
            CSV.write(filename, df)
            
            # Increment counters
            global experiment_num += 1
            global completed_count += 1
            
            # Progress update
            progress = round(completed_count / total_experiments * 100, digits=1)
            println("Progress: $completed_count/$total_experiments ($progress%)")
        end
        # Reset eta values for next n (only for the first iteration after resume)
        global remaining_eta_values = eta_values
    end
    # Reset n values for next N (only for the first iteration after resume)
    global remaining_n_values = n_values
end

println("\n=== All experiments completed ===")
println("Results saved to $filename")
println("Total experiments conducted: $(experiment_num - 1)")
