using Random
using LinearAlgebra
using Plots
using Distributions
using Flux
using Statistics


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

function labeled_dataset(M, N; seed=nothing)
    rng = isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed)

    M_2 = div(M, 2)

    y_data = []
    c_data = zeros(Int, M)
    s_hat_data = []

    #data for A = 0
    for k in 1:M_2
        t, y, s, n, s_hat, dt = generate_data(0, N; seed=isnothing(seed) ? nothing : seed + M_2 + k)
        push!(y_data, y)
        push!(s_hat_data, s_hat)
        c_data[k] = 0
    end

    #data for A = 3
    for k in 1:M_2
        t, y, s, n, s_hat, dt = generate_data(3, N; seed=isnothing(seed) ? nothing : seed + M_2 + k)
        push!(y_data, y)
        push!(s_hat_data, s_hat)
        c_data[M_2 + k] = 1
    end

    return y_data, s_hat_data, c_data

end

function preprocess_data(data)
    #Convert to a matrix
    data_matrix = hcat(data...)
    #Standardization
    mu = mean(data_matrix, dims=2)
    std_dev = std(data_matrix, dims=2)
    scaled_data = (data_matrix .- mu) ./ (std_dev .+ 1e-8)

    #Reshape data to account for 3 inputs
    N, M = size(scaled_data)
    return reshape(scaled_data, N, 1, M)
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

    possible_thresholds = range(-1, 4, length=nsteps)

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

function filter_classifier(y, s_hat, rho_not)
    #=
    Outputs predicted class
    =#
    rho = snr_calculate(y, s_hat)
    class_pred = rho >= rho_not ? 1 : 0
    return class_pred
end


function confusion_matrix(M, N, rho_not)
    #=
    Compute confusion matrix using labeled dataset
    =#
    tp, tn, fp, fn = 0, 0, 0, 0
    
    # Generate labeled dataset
    y_data, s_hat_data, c_data = labeled_dataset(M, N; seed=42)
    
    for k in 1:M
        y = y_data[k]
        s_hat = s_hat_data[k]
        true_label = c_data[k]
        pred_class = filter_classifier(y, s_hat, rho_not)
        
        if true_label == 1 && pred_class == 1
            tp += 1
        elseif true_label == 0 && pred_class == 0
            tn += 1
        elseif true_label == 0 && pred_class == 1
            fp += 1
        elseif true_label == 1 && pred_class == 0
            fn += 1
        end
    end
    return tp, tn, fp, fn
end

function cnn_model(N)
    model = Chain(
        Conv((25,), 1=>8, pad=12),
        BatchNorm(8),
        relu,
        MaxPool((4,)),
        Conv((15,), 8 => 12, pad=7),
        BatchNorm(12),
        relu,
        MaxPool((4,)),
        Flux.flatten,
        Dense(div(div(N, 4), 4) * 12 => 32),
        BatchNorm(32),
        relu,
        Dropout(0.4),
        Dense(32 => 2),
        softmax
    )
    return model
end

function make_batches(x, y, batch_size)
    n_samples = size(x, 3)
    indices = shuffle(1:n_samples)
    batches = []
    for i in 1:batch_size:n_samples
        batch_end = min(i + batch_size -1, n_samples)
        batch_indices = indices[i:batch_end]
        x_batch = x[:, :, batch_indices]
        y_batch = y[:, batch_indices]
        push!(batches, (x_batch, y_batch))
    end
    return batches
end

function my_accuracy(y_pred, y_true)

    pred_labels = Flux.onecold(y_pred, 0:1)
    true_labels = Flux.onecold(y_true, 0:1)

    return mean(pred_labels .== true_labels)
end

function train_model!(model, x_train, y_train, x_val, y_val; epochs = 50, batch_size = 32, lr = 0.001)
    opt = Flux.Adam(lr)
    opt_state = Flux.setup(opt, model) 
    train_losses = Float64[]
    val_losses = Float64[]
    train_accs = Float64[]
    val_accs = Float64[]

    for epoch in 1:epochs
        batches = make_batches(x_train, y_train, batch_size)
        Flux.trainmode!(model)

        for (batch_idx, (x_batch, y_batch)) in enumerate(batches)
            val, grads = Flux.withgradient(model) do m
                y_pred = m(x_batch)
                loss = Flux.crossentropy(y_pred, y_batch)
            end
            # Detect loss of Inf or NaN. Print a warning, and then skip update! (taken from documentation)
            if !isfinite(val)
                @warn "loss is $val on item $i" epoch
                continue
            end

            Flux.update!(opt_state, model, grads[1]) 
        end
        Flux.testmode!(model)

        y_pred_train = model(x_train)
        train_loss = Flux.crossentropy(y_pred_train, y_train)
        train_acc = my_accuracy(y_pred_train, y_train)

        # validation metrics
        y_pred_val = model(x_val)
        val_loss = Flux.crossentropy(y_pred_val, y_val)
        val_acc = my_accuracy(y_pred_val, y_val)

        push!(train_losses, train_loss)
        push!(val_losses, val_loss)
        push!(train_accs, train_acc)
        push!(val_accs, val_acc)

        if epoch % 5 == 0 || epoch == 1
            println("Epoch $epoch/$epochs - train loss=$(round(train_loss,digits=4)) val loss=$(round(val_loss,digits=4)) train acc=$(round(train_acc*100,digits=2))% val acc=$(round(val_acc*100,digits=2))%")
        end
    end
    return train_losses, val_losses, train_accs, val_accs
end

function auc(fpr, tpr)
    #= 
    Calculate are under the curve
    =#
    idx = sortperm(fpr)
    fpr_sorted = fpr[idx]
    tpr_sorted = tpr[idx]
    lent = length(fpr_sorted)
    area = 0.0
    for i in 2:lent
        area += (fpr_sorted[i] - fpr_sorted[i-1]) *
                (tpr_sorted[i] + tpr_sorted[i-1]) / 2
    end
    return area
end

function inspect_size(model, N)
    x = rand(Float32, N, 1, 1)

    println("Input: ", size(x))
    for (i, layer) in enumerate(model.layers)
        x = layer(x)
        println("Layer $i ($(typeof(layer))): ", size(x))
    end
end

N = 2048
M = 1000
A = [0, 3]
rho_res = Dict()

#Load data stream
y_data, s_hat_data, labels = labeled_dataset(M, N; seed=123)

#Splitting dataset
train_size = Int(floor(0.8*M))
test_size = M - train_size

rng = MersenneTwister(123)
indices = randperm(rng, M)
train_indices = indices[1:train_size]
test_indices = indices[train_size + 1:end]

y_train = y_data[train_indices]
s_hat_train = s_hat_data[train_indices]
labels_train = labels[train_indices]

y_test = y_data[test_indices]
s_hat_test = s_hat_data[test_indices]
labels_test = labels[test_indices]


rho_train = zeros(train_size)
rho_res = Dict()
for k in 1:train_size
    y = y_train[k]
    s_hat = s_hat_train[k]
    rho_train[k] = snr_calculate(y, s_hat)
end

rho_res[0] = rho_train[labels_train .== 0]  # No-signal
rho_res[3] = rho_train[labels_train .== 1]   # Signal

p = plot(size=(800, 600), legend=:topright, 
         xlabel="p", ylabel="Density", 
         title="Empirical Distribution of p(A)")

colors = [:blue, :green]

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
    x_range = range(a - 5, a + 5, length=200)
    theoretical_pdf = pdf.(Normal(theoretical_mean, theoretical_std), x_range)
    plot!(p, x_range, theoretical_pdf, 
          linewidth=2, 
          color=colors[idx], 
          linestyle=:dash,
          label="A = $a (T)")
end

savefig(p, "rho_histograms.png")


rho_not = find_best_threshold(rho_res[0], rho_res[3])
println("Chosen threshold = ", rho_not)

#part 2: Test classifier on labeled dataset
tp, tn, fp, fn = confusion_matrix(M, N, rho_not)

confusion_mat = [
    tp fn;
    fp tn
]

println("Confusion Matrix: $confusion_mat ")

# Calculate performance metrics
accuracy = (tp + tn) / M
false_pos_rate = fp / (fp + tn)
false_neg_rate = fn / (fn + tp)

println("Accuracy: $(round(accuracy * 100))%")
println("FP Rate: $(round(false_pos_rate * 100))%")
println("FN Rate: $(round(false_neg_rate * 100))%")

#ROC analysis on testing data
rho_test = zeros(test_size)
test_size = length(y_test)
for k in 1:test_size
    rho_test[k] = snr_calculate(y_test[k], s_hat_test[k])
end

labels_test_int = labels_test

# Threshold sweep
rho_thresholds = range(-1, 4, length=200)
tpr_mf = Float64[]
fpr_mf = Float64[]

for thr in rho_thresholds
    preds = rho_test .>= thr

    tp = sum((preds .== 1) .& (labels_test_int .== 1))
    fp = sum((preds .== 1) .& (labels_test_int .== 0))
    fn = sum((preds .== 0) .& (labels_test_int .== 1))
    tn = sum((preds .== 0) .& (labels_test_int .== 0))

    push!(tpr_mf, tp / (tp + fn))
    push!(fpr_mf, fp / (fp + tn))
end

# AUC for matched filter
auc_mf = auc(fpr_mf, tpr_mf)
println("Matched Filter AUC = ", round(auc_mf, digits=4))

#Load CNN model

model = cnn_model(N)

#Calculate total number of trainable parameters
total_params = sum(length, Flux.params(model))
println("Total trainable parameters: $total_params")

# Preprocessing for CNN
# Convert lists of vectors into tensors and standardize
x_train_all = Float32.(preprocess_data(y_train))
x_test = Float32.(preprocess_data(y_test))

# Convert integer labels (0/1) into one-hot matrices (2 x n)
y_train_all_oh = Float32.(Flux.onehotbatch(labels_train, 0:1))
y_test_oh = Float32.(Flux.onehotbatch(labels_test, 0:1))

# Validation set from the training data :10% of training
val_frac = 0.2
val_size = Int(floor(val_frac * size(x_train_all, 3)))
perm = shuffle(1:size(x_train_all, 3))
val_idx = perm[1:val_size]
train_idx = perm[val_size+1:end]

x_tr = x_train_all[:, :, train_idx]
y_tr = y_train_all_oh[:, train_idx]

x_val = x_train_all[:, :, val_idx]
y_val = y_train_all_oh[:, val_idx]

println("Training with $(size(x_tr,3)) samples, validating on $(size(x_val,3)) samples, testing on $(size(x_test,3)) samples")

# Training hyperparameters
epochs = 50
batch_size = 32
lr = 1e-3

# Train
train_losses, val_losses, train_accs, val_accs = train_model!(model, x_tr, y_tr, x_val, y_val; epochs=epochs, batch_size=batch_size, lr=lr)

# Plot training/validation loss and accuracy
p_loss = plot(1:epochs, train_losses, label="train loss", xlabel="Epoch", ylabel="Loss", title="Loss vs Epoch")
plot!(p_loss, 1:epochs, val_losses, label="val loss")
savefig(p_loss, "train_val_loss.png")

p_acc = plot(1:epochs, train_accs, label="train acc", xlabel="Epoch", ylabel="Accuracy", title="Accuracy vs Epoch")
plot!(p_acc, 1:epochs, val_accs, label="val acc")
savefig(p_acc, "train_val_acc.png")

# Evaluate on test set
Flux.testmode!(model)
y_pred_test = model(x_test) # shape (2, n_test)
# Use probability of class 1  and threshold 0.5
p_hat = Array(y_pred_test[2, :])
pred_labels = Int.(p_hat .>= 0.5)

# compute confusion matrix
tp_t = sum((pred_labels .== 1) .& (labels_test .== 1))
tn_t = sum((pred_labels .== 0) .& (labels_test .== 0))
fp_t = sum((pred_labels .== 1) .& (labels_test .== 0))
fn_t = sum((pred_labels .== 0) .& (labels_test .== 1))

conf_mat_test = [tp_t fn_t; fp_t tn_t]
println("CNN Test Confusion Matrix: $conf_mat_test")

accuracy_test = (tp_t + tn_t) / length(labels_test)
println("CNN Test Accuracy: $(round(accuracy_test * 100, digits=3))%")
println("CNN Test FP Rate: $(round(fp_t / (fp_t + tn_t) * 100, digits=3))%")
println("CNN Test FN Rate: $(round(fn_t / (fn_t + tp_t) * 100, digits=3))%")

savefig(p_loss, "train_val_loss.png")
savefig(p_acc, "train_val_acc.png")

p_thresholds = range(0, 1, length=200)
tpr_cnn = Float64[]
fpr_cnn = Float64[]

for p0 in p_thresholds
    preds = p_hat .>= p0

    tp = sum((preds .== 1) .& (labels_test_int .== 1))
    fp = sum((preds .== 1) .& (labels_test_int .== 0))
    fn = sum((preds .== 0) .& (labels_test_int .== 1))
    tn = sum((preds .== 0) .& (labels_test_int .== 0))

    push!(tpr_cnn, tp / (tp + fn))
    push!(fpr_cnn, fp / (fp + tn))
end

#AUC for CNN
auc_cnn = auc(fpr_cnn, tpr_cnn)
println("CNN AUC = ", round(auc_cnn, digits=4))

#Plot ROC curves

p_roc = plot(
    fpr_mf, tpr_mf,
    label = "Matched Filter (AUC=$(round(auc_mf,digits=3)))",
    xlabel = "False Positive Rate",
    ylabel = "True Positive Rate",
    linewidth = 2,
    legend = :bottomright,
    title = "ROC Curve Comparison"
)

plot!(
    p_roc,
    fpr_cnn,  tpr_cnn,
    label = "CNN (AUC=$(round(auc_cnn,digits=3)))",
    linewidth = 2,
)

plot!(p_roc, [0,1], [0,1], linestyle=:dot)

savefig(p_roc, "roc_comparison.png")

inspect_size(model, N)
