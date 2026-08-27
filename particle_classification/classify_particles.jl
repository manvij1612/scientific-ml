using CSV, DataFrames, Plots, StatsPlots
using Statistics, Distributions
using MLJ

#Particle Physics Problem: A physics-informed classifier to distinguish between different particles using Gaussian Discriminant Analysis (GDA) and Random Forest
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree

filename1 = "train.csv" 
filename2 = "test.csv"

function load_data(filename)
    #=
    Load DataFrame 
    =#
    df = CSV.read(filename, DataFrame)
    return df
end

function compute_pmag(df)
    #=
    Compute |p| for daughter particles
    =#
    p1_mag = sqrt.(df.px1 .^2 + df.py1 .^2 + df.pz1 .^2)
    p2_mag = sqrt.(df.px2 .^2 + df.py2 .^2 + df.pz2 .^2)

    return p1_mag, p2_mag
end

function parent_calc(df)
    #= 
    Calculation for parent particle
    =#
    m = sqrt.((df.E1 .+ df.E2) .^ 2 .- (df.px1 .+ df.px2) .^ 2 .- (df.py1 .+ df.py2) .^ 2 .- (df.pz1 .+ df.pz2) .^ 2)
    pT = sqrt.((df.px1 .+ df.px2) .^ 2 .+ (df.py1 .+ df.py2) .^ 2)
    return m, pT
end

function means_std(m)
    mu = mean(m)
    sigma = std(m)
    return mu, sigma
end

function random_forest_classify(filename1, filename2)
    train_df = load_data(filename1)
    test_df = load_data(filename2)

    m_train, pT_train = parent_calc(train_df)

    X_train = DataFrame(m = m_train, pT = pT_train)
    y_train = categorical(train_df.label)

    model = RandomForestClassifier(n_trees=200, max_depth=-1, min_samples_leaf=1)
    mach = machine(model, X_train, y_train)
    fit!(mach, verbosity=0)
    # Load test data
    m_test, pT_test = parent_calc(test_df)
    X_test = DataFrame(m = m_test, pT = pT_test)
    
    # Predict probabilities
    test_probs = predict(mach, X_test)
    probs_B = pdf.(test_probs, "B")
    numeric_labels = get_numeric_labels(probs_B)

    # save_predictions("manvi_forest.txt", numeric_labels)
    # Compute counts
    N_B_pred = sum(probs_B)
    N_A_pred = length(probs_B) - N_B_pred
    
    return N_A_pred, N_B_pred
end

function qda_classify(filename1, filename2)
    
    # Load training data
    train_df = load_data(filename1)
    m_train, pT_train = parent_calc(train_df)
    labels = train_df.label
    
    A_mask = labels .== "A"
    B_mask = labels .== "B"
    
    # Class A params
    muA = [mean(m_train[A_mask]), mean(pT_train[A_mask])]
    XA = hcat(m_train[A_mask], pT_train[A_mask])
    covA = cov(XA)
    
    # Class B params
    muB = [mean(m_train[B_mask]), mean(pT_train[B_mask])]
    XB = hcat(m_train[B_mask], pT_train[B_mask])
    covB = cov(XB)
    
    # Prior probabilities
    pA = sum(A_mask) / length(labels)
    
    println("Class A: mean = ", muA)
    println("Class B: mean = ", muB)
    
    # Load test data
    test_df = load_data(filename2)
    m_test, pT_test = parent_calc(test_df)
    
    # Create distributions
    distA = MvNormal(muA, covA)
    distB = MvNormal(muB, covB)
    
    # Predict probabilities
    probs_B = zeros(length(m_test))
    for i in 1:length(m_test)
        x = [m_test[i], pT_test[i]]
        likA = pdf(distA, x) * pA
        likB = pdf(distB, x) * (1 - pA)
        probs_B[i] = likB / (likA + likB)
    end
    
    # Compute counts
    N_B_pred = sum(probs_B)
    N_A_pred = length(probs_B) - N_B_pred
    
    return N_A_pred, N_B_pred
end

df = load_data(filename1)

p1_mag, p2_mag = compute_pmag(df)

m, pT = parent_calc(df)

# Exploration
plot(p1_mag, df.E1, seriestype = :scatter, title = "E1 vs |p1|", xlabel = "|p1|", ylabel = "E1")
plot!(p2_mag, df.E2, seriestype = :scatter, title = "E2 vs |p2|", xlabel = "|p2|", ylabel = "E2")
savefig("pvsE.png")

histogram(m[df.label .== "A"], bins=50, label="Particle A", title="Invariant Mass for A and B", xlabel="Invariant Mass m", ylabel="Count")
histogram!(m[df.label .== "B"], bins=50, label="Particle B")
savefig("massperclass.png")

scatter(m, pT, group=df.label, title="Invariant Mass vs Parent pT", xlabel="Invariant Mass m", ylabel="Parent pT")
savefig("mvspt.png")

mA = m[df.label .== "A"]
mB = m[df.label .== "B"]

#Results
N_A_qda, N_B_qda = qda_classify(filename1, filename2)
N_A, N_B = random_forest_classify(filename1, filename2)
print("qda results:")
print("\n NA:", N_A_qda)
print("\n NB:", N_B_qda)
print("\n random forest results:")
print("\n NA:", N_A)
print("\n NB:", N_B)

