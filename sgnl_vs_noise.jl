using Random
using LinearAlgebra
using Plots
using Distributions

function generate_data(A, N; seed=nothing)
    #=
    Generate a noisy time series dataset {t_i, y_i}_{i=1}^N.
    
    Parameters
    ----------
    A : float
        Amplitude of the signal (A >= 0)
    N : int
        Number of data points. Time points uniformly sampled from [0, 2pi]
    seed : int or nothing, optional
        Random seed for reproducibility
    =#

    rng = isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed)
    t = range(0.0, 2*π, length=N)
    dt = (2*π) / (N - 1)
    s_hat = sqrt(dt / π) * sin.(t)  # ||s_hat||_2 == 1 exactly on this grid
    s = A * s_hat
    n = randn(rng, N)
    y = s + n
    return t, y, s, n, s_hat, dt
end

function snr_calculate(y, s_hat)
    #=
    Computes $\rho(A)$
    =#
    rho = dot(y, s_hat)
    return rho
end

function find_best_threshold(rho_noise, rho_signal; nsteps=100)
    #=
    Finds best threshold 
    =#

    possible_thresholds = range(-1, 8, length=nsteps)

    best_threshold = 0.0
    best_diff = Inf

    for thr in possible_thresholds
        false_pos = mean(rho_noise .>= thr)
        false_neg = mean(rho_signal .< thr)
        diff = abs(false_pos - false_neg)

        if diff < best_diff
            best_diff = diff
            best_threshold = thr
        end
    end
    return best_threshold
end

N = 1000
A = [0, 3, 10]
rho_res = Dict()

for a in A

    rho = zeros(N)

    for i in 1:N
        t, y, s, n, s_hat, dt = generate_data(a, N)
        rho[i] = snr_calculate(y, s_hat)
    end

    rho_res[a] = rho
end

p = plot(size=(800, 600), legend=:topright, 
         xlabel="p", ylabel="Density", 
         title="Empirical Distribution of p(A)")

colors = [:blue, :green, :orange]

for (idx, a) in enumerate(A)
    rho_samples = rho_res[a]
    
    # Empirical statistics
    empirical_mean = mean(rho_samples)
    empirical_std = std(rho_samples)
    
    # Theoretical parameters
    theoretical_mean = a
    theoretical_std = 1.0
    
    # Histogram
    histogram!(p, rho_samples, 
               bins=50, 
               normalize=:pdf, 
               alpha=0.4, 
               color=colors[idx],
               label="A = $a (E)")
    
    # Theoretical Gaussian
    x_range = range(a - 4, a + 4, length=200)
    theoretical_pdf = pdf.(Normal(theoretical_mean, theoretical_std), x_range)
    plot!(p, x_range, theoretical_pdf, 
          linewidth=2, 
          color=colors[idx], 
          linestyle=:dash,
          label="A = $a (T)")
end

savefig(p, "rho_histograms.png")

best_threshold = find_best_threshold(rho_res[0], rho_res[3])
println("Chosen threshold = ", best_threshold)

# Build confusion matrix
tp = sum(rho_res[3] .>= best_threshold)
fn = sum(rho_res[3] .< best_threshold)
fp = sum(rho_res[0] .>= best_threshold)
tn = sum(rho_res[0] .< best_threshold)

confusion_matrix = [
    tp fn;
    fp tn
]

println("Confusion Matrix:")
println(confusion_matrix)
