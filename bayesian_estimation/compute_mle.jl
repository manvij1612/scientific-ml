using LinearAlgebra, CSV, DataFrames, Plots

# Bayesian estimation of satellite's orbital frequency

function compute_mle(t, theta)
    #= 
    1(c): Maximimum likelihood estimation
    =#
    w_num = sum(t .* theta)
    w_denom = sum(t.^2)

    w = w_num / w_denom
    return w
end

function noise_variance(t, theta, w)
    #= 
    1(e): Estimate noise vatiance
    =#
    n = length(t)
    residuals = theta .- w .* t
    sigma2 = sum(residuals.^2) / (n-1)
    return sigma2
end

function posterior_variance(t, theta, w)
    #= 
    2(b): Compute bayesian posterior variance
    =#
    sigma2 = noise_variance(t, theta, w)
    a1 = sum(t.^2) / sigma2
    a2 = 1/ 0.001
    tau_n2 = 1 / (a1 + a2)
    return tau_n2
end

function posterior_mean(t, theta, w)
    #= 
    2(b): Compute bayesian posterior mean
    =#
    sigma2 = noise_variance(t, theta, w)
    tau_n2 = posterior_variance(t, theta, w)
    b1 = sum(t .* theta) / sigma2
    b2 = 0.001 / 0.001
    mu_n = tau_n2 * (b1 + b2)
    return mu_n
end

function sequential_update(t, theta, w, tau02, mu0)
    #= 
    3(a): Sequential Bayesian update
    =#
    k = length(t)
    sigma2 = noise_variance(t, theta, w)

    a2 = 1/ tau02
    b2 = mu0 / tau02

    mu_k = zeros(k)
    tau_k2 = zeros(k)

    S_tt = 0.0
    S_ttheta = 0.0

    for i in 1:k
        S_tt += t[i]^2
        S_ttheta += t[i] * theta[i]
        a1 = S_tt / sigma2
        tau_k2[i] = 1 / (a1 + a2)
        b1 = S_ttheta / sigma2   
        mu_k[i] = tau_k2[i] * (b1 + b2)
    end
    return mu_k, tau_k2
end

function entropy(t, theta, w)
    #= 
    3(a): Sequential Entropy update
    =#
    k = length(t)
    sigma2 = noise_variance(t, theta, w)

    tau0_sq = 1.0
    H0 = 0.5 + 0.5 * log(2 * pi * tau0_sq)

    a2 = 1
    tau_k2 = zeros(k)

    S_tt = 0.0
    H = zeros(k)
    delta_H = zeros(k)
    for i in 1:k
        S_tt += t[i]^2
        a1 = S_tt / sigma2
        tau_k2[i] = 1 / (a1 + a2)
        H[i] = 0.5 + 0.5 * log(2 * pi * tau_k2[i])
        if i == 1
            delta_H[i] = H[i] - H0
        else
            delta_H[i] = H[i] - H[i-1]
        end
    end
    return H, delta_H
end

data = CSV.read("compute_mle.csv", DataFrame, header=false)
t = data[:,1]
theta = data[:, 2]
w = compute_mle(t, theta)
println("Computed MLE w: $w")

noise = noise_variance(t, theta, w)
println("Estimated noise variance: $noise")

tau = posterior_variance(t, theta, w)
println("Posterior variance: $tau")
mu = posterior_mean(t, theta, w)
println("Posterior mean: $mu")


p = plot(size=(800, 600), legend=:topright, 
         xlabel="t", ylabel="theta", 
         title="Noisy observations and MLE fit")
scatter!(p, t, theta, label="Noisy observations", markersize=2, color=:blue)
plot!(p, t, w*t, label="MLE fit", linewidth=2, color=:red)
savefig(p, "mle_fit.png")

k = length(t)

mu_k, tau_k2 = sequential_update(t, theta, w, 0.001, 0.001)

lower = mu_k .- 1.96 .* sqrt.(tau_k2)
upper = mu_k .+ 1.96 .* sqrt.(tau_k2)


p1 = plot(mu_k, 1:k,
    size=(800, 600), 
    legend=:right,
    xlabel="mean (rad/s)",
    ylabel="Number of Observations (k)",
    title="Sequential Bayesian Update",
    label="Posterior Mean",
    lw=2,
    color=:blue,
    xlims=(0.00095, 0.02))

# Add 95% credible interval band
plot!(p1, lower, 1:k,
    fillrange=upper,
    fillalpha=0.3,
    label="95% Credible Interval",
    color=:lightblue,
    lw=0)

vline!(p1, [w],
    label="MLE",
    ls=:dash,
    color=:red,
    lw=2)

savefig(p1, "comp_fit.png")

entropy_vals, delta_entropy = entropy(t, theta, w)

p2 = plot(1:k, entropy_vals,
    size=(800, 600), 
    legend=:right,
    xlabel="Number of Observations (k)",
    ylabel="Entropy H",
    title="Sequential Entropy Update",
    label="Entropy",
    lw=2,
    color=:blue)

savefig(p2, "entropy_fit.png")

p3 = plot(1:k, delta_entropy,
    size=(800, 600), 
    legend=:right,
    xlabel="Number of Observations (k)",
    ylabel="Change in entropy",
    title="Sequential Entropy Update",
    label="Change in entropy",
    lw=2,
    color=:blue)

savefig(p3, "entropy_change_fit.png")
