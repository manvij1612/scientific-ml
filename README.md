# scientific-ml

Independent exercises in scientific computing and statistical learning, mostly
built around gravitational-wave-flavoured toy problems (matched filtering,
signal-vs-noise detection, parameter estimation). Each script is self-contained
and writes its figures to the folder it is run from.

Python scripts use NumPy, SciPy, scikit-learn, PyTorch, or JAX/Flax/Optax.
Julia scripts use CSV, DataFrames, Flux, MLJ, Distributions, and Plots.
Run the Julia scripts from inside their own folder so the relative `*.csv`
paths resolve.

## `gw_signal_detection/`
Detecting a known signal buried in noise.
- `sgnl_vs_noise.jl` — matched-filter detection statistic `ρ = ⟨y, ŝ⟩`; picks
  the decision threshold and compares the signal and noise-only distributions.
- `matchedfilter_via_cnn.jl` — trains a Flux CNN to do the same detection task
  and compares its ROC curve against the analytic matched filter.

## `bayesian_estimation/`
Point estimates and posteriors from noisy data.
- `compute_mle.jl` — maximum-likelihood estimate of a satellite's orbital
  frequency from noisy angle measurements, plus a noise-variance estimate and
  the Gaussian posterior mean/variance. Reads `compute_mle.csv`.
- `find_degree.jl` — chooses the degree of a polynomial fit using train/test
  MSE and the SVD condition number of the design matrix. Reads `find_degree.csv`.

## `particle_classification/`
- `classify_particles.jl` — builds invariant-mass and transverse-momentum
  features for two-body events and classifies them with Gaussian discriminant
  analysis and a random forest (MLJ). Reads `train.csv` and `test.csv`.

## `gaussian_mixtures/`
Fitting and using Gaussian mixture models.
- `groundtruth_vs_gmm.py` — 2-D Gaussian MLE with 1σ error ellipses, fitted
  model against ground truth.
- `predict_pulsar_classes.py` — GMM on a two-component "pulsar" mixture: BIC
  model selection, label alignment, adjusted Rand index, confusion matrix.
- `classifier_auc.py` — ROC / AUC for GMM clustering versus a geometric
  decision boundary, with a train/test split.

## `ridge_regression/`
- `ridge_reg.py` — closed-form Tikhonov ridge regression checked against
  `sklearn.Ridge`, a condition-number study over polynomial degree `M` and
  penalty `λ`, and a grid search for the best `(M, λ)`.
- `sgd/batch_size_study.py`, `sgd/learning_rate_study.py` — how mini-batch size
  and learning rate affect SGD convergence on the ridge problem.
- `sgd/sgd_vs_adam.py` — SGD against Adam on the same problem.
- `sgd/nn_plus_adam.py` — a 2-layer MLP (JAX + Flax NNX + Optax) fitting
  `f(t) = e^{-3t} sin(8πt)` with Adam.

## `hamiltonian_nn/`
- `hamiltonian_nn.py` — learns the Hamiltonian `H(q, p)` of the harmonic
  oscillator with a neural network, integrates the learned dynamics with RK4,
  and tests how far the solution extrapolates beyond the training window.
