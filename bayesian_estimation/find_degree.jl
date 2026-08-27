using LinearAlgebra, CSV, Plots, DataFrames, Random, Statistics

Random.seed!(42)
ENV["GKSwstype"] = "100"

function design_matrix(t::Vector{Float64}, M::Int)
    #=
    Construct a design matrix
    =#
    N = length(t)
    A = ones(N, M + 1)
    for i in 1:N
        for j in 1:M+1
            A[i, j] = t[i] ^ (j-1)
        end
    end
    return A
end

function fit_model(t::Vector{Float64}, y::Vector{Float64}, M::Int)
    #=
    Fit a polynomial model of degree M to the data (t, y)
    =#
    A = design_matrix(t, M)
    w = pinv(A) * y
    U, S, V = svd(A)
    K = maximum(S) / minimum(S[S .> 1e-10])

    return w, A, K
end

function mse(y_true::Vector{Float64}, y_pred::Vector{Float64})
    #=
    Compute Mean Squared Error
    =#
    return mean((y_true .- y_pred).^2)
end

function plot_fit(t::Vector{Float64}, y::Vector{Float64}, w::Vector{Float64}, M::Int)
    #=
    Plot the data and the fitted polynomial
    =#
    t_fit = range(minimum(t), maximum(t), length=200)
    A_fit = design_matrix(collect(t_fit), M)
    y_fit = A_fit * w

    scatter(t, y, label="Given Data", legend=:topleft)
    plot!(t_fit, y_fit, label="Fitted Polynomial (M=$M)", lw=2)
    xlabel!("t")
    ylabel!("y")
    title!("Polynomial Fit of Degree M=$M")
    savefig("plot(M=$M).png")
    return y_fit
end

function data_split(t::Vector{Float64}, y::Vector{Float64}, N, m)
    #=
    Split data into training and testing sets
    =#
    train_data_size = Int(floor(0.8 * N))
    index = randperm(N)
    train_indices = index[1:train_data_size]
    test_indices = index[train_data_size+1:end]

    t_train = t[train_indices]
    y_train = y[train_indices]
    t_test = t[test_indices]
    y_test = y[test_indices]

    train_mse_values = []
    test_mse_values = []
    condition_numbers = []
    #Compute MSE for training and testing sets
    for M in m
        w_train, A_train, K_train = fit_model(t_train, y_train, M)
        A_test= design_matrix(t_test, M)
        
        y_train_pred = A_train * w_train
        y_test_pred = A_test * w_train
        
        train_mse = mse(y_train, y_train_pred)
        test_mse = mse(y_test, y_test_pred)

        push!(train_mse_values, train_mse)
        push!(test_mse_values, test_mse)
        push!(condition_numbers, K_train)
        println("Polynomial Degree = $M: Train MSE=$(train_mse), Test MSE=$(test_mse), Condition # =$(K_train)")
    end

    return t_train, y_train, t_test, y_test, train_mse_values, test_mse_values, condition_numbers
end

fitted_models = Dict()
# Load data
data = CSV.read("find_degree.csv", DataFrame, header=false)

t = data[:, 1]
y = data[:, 2]
# Fit models of degree 1, 3, and 9 and plot
m = [1, 3, 9]
for M in m
    w, A, K = fit_model(t, y, M)
    fitted_models[M] = w
    y_fit = plot_fit(t, y, w, M)
    println("M=$M coefficients: w = $w")
end

# Split data into training and testing sets
train_mse_v = Dict()
test_mse_v = Dict()
condition_num = Dict()
for N in [10, 40, 100]
    condition_num[N] = []
    train_mse_v[N] = []
    test_mse_v[N] = []
    t_train, y_train, t_test, y_test, train_mse_values, test_mse_values, condition_numbers = data_split(t, y, N, m)
    push!(train_mse_v[N], (train_mse_values))
    push!(test_mse_v[N], (test_mse_values))
    push!(condition_num[N], (condition_numbers))
end

for N in [10, 40, 100]
    p1 = plot(m, train_mse_v[N], 
        label="Training MSE", 
        marker=:circle, 
        linewidth=2,
        markersize=4,
        color=:blue,
        title="Training vs Testing MSE",
        xlabel="M",
        ylabel="MSE",
        yscale=:log10)

    plot!(p1, m, test_mse_v[N], 
    label="Testing MSE", 
    marker=:square, 
    linewidth=2,
    markersize=4,
    color=:red)

    p2 = plot(m, condition_num[N],
        label="Condition Number",
        marker=:diamond,
        linewidth=2,
        markersize=4,
        color=:green,
        title="Condition Number vs M",
        xlabel=" M",
        ylabel="K",
        yscale=:log10)

    # Combined plot
    p_combined = plot(p1, p2, layout=(2,1), size=(800, 600))
    savefig(p_combined, "model_selection_analysis_$N.png")
end

cond_results = Dict()

#Condition number and stability
for N in [10, 40, 100]
    cond_results[N] = []
    t_sub = t[1:N]
    y_sub = y[1:N]
    for M in m
        w_sub, A_sub, K_sub = fit_model(t_sub, y_sub, M)
        println("N=$N, M=$M: Condition Number = $K_sub")
        push!(cond_results[N], (K_sub))
        if N == 10 && M == 9
            println("    ^ This is interpolation (N=M+1)")
        end
    end
end

plot_cond = plot(title="Condition Number vs M", 
                xlabel="M", 
                ylabel="Condition Number", 
                yscale=:log10,
                legend=:topleft)

for N_sub in [10, 40, 100]
    plot!(plot_cond, m, cond_results[N_sub], 
          label="N = $N_sub", marker=:o, linewidth=2, markersize=6)
end

savefig(plot_cond, "condition_numbers.png")

M_final = 3
w_final, A_final, K_final = fit_model(t, y, M_final)

# Getting final coeffeicients
coefficients_final = w_final
coefficients_padded = zeros(10)
coefficients_padded[1:4] = w_final
println("Final coefficients (w0 to w9): ", coefficients_final)
println("Padded coefficients (w0 to w9): ", coefficients_padded)
