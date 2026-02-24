import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cond
from sklearn.linear_model import Ridge

def generate_data(scale_noise, N, seed=1234):
    """ Generate a noisy time series dataset {t_i, y_i}_{i=1}^N.
    Parameters
    ----------
    A : float
    Amplitude of the signal (A >= 0)
    N : int
    Number of data points. Time points uniformly sampled from [0, 2pi]
    seed : int or None, optional
    Random seed for reproducibility """

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, N)
    dt = (2*np.pi) / (N - 1)
    A=np.sqrt(np.pi/dt)
    s_hat = np.sqrt(dt / np.pi) * np.sin(2*np.pi*t) # ||s_hat||_2 == 1 exactly on this grid
    s = A * s_hat
    n = rng.normal(loc=0.0, scale=scale_noise, size=N)
    y = s + n
    return t, y, s, n, s_hat, dt

def vander_matrix(x, M):
    return np.vander(x, N=M+1, increasing=True)

def compute_omega(gamma, A, lamda, y):
    lhs_1 = A.T @ A
    lhs_2 = lamda * gamma.T @ gamma
    lhs = lhs_1 + lhs_2
    rhs = A.T @ y
    w = np.linalg.solve(lhs, rhs)
    return w

# generate testing and training data
x_data,y_data,s,n,s_hat, dt = generate_data(.1,100)
t_train = x_data[::10]
y_train = y_data[::10]
s_train = s[::10]
plt.plot(t_train,y_train,'bo')
plt.plot(t_train,s_train,'r--')
plt.plot(x_data,s,'k--')

# t_test and y_test will have 90 data points
t_test = np.delete(x_data, np.arange(0, 100, 10))
y_test = np.delete(y_data, np.arange(0, 100, 10))

Ms = np.arange(1, 10)
lamdas = [0, 1e-6, 1e-4, 1e-2, 1, 1e2]
gamma_gen = lambda M: np.diag([0] + [1]*M)
train_mse, test_mse, cond_plain, cond_ridge = {}, {}, {}, {}

for M in Ms:
    train_mse[M], test_mse[M] = [], []
    cond_plain[M], cond_ridge[M] = [], []
    A_train, A_test = vander_matrix(t_train, M), vander_matrix(t_test, M)

    for lamda in lamdas:
        gamma = gamma_gen(M)
        w = compute_omega(gamma, A_train, lamda, y_train)

        # Compute errors
        y_pred_train, y_pred_test = A_train @ w, A_test @ w
        train_mse[M].append(np.mean((y_pred_train - y_train)**2))
        test_mse[M].append(np.mean((y_pred_test - y_test)**2))
        
        # Condition numbers
        cond_plain[M].append(cond(A_train.T @ A_train))
        cond_ridge[M].append(cond(A_train.T @ A_train + lamda * (gamma.T @ gamma)))

#Plots

plt.figure(figsize=(10, 6))
for M in Ms:
    plt.plot(lamdas, test_mse[M], marker='o', label=f'M={M}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Regularization (log scale)')
plt.ylabel('Test MSE')
plt.title('Test MSE vs lambda for different M values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
plt.savefig('test_mse_vs_lambda.png')

plt.figure(figsize=(10, 6))
for M in Ms:
    plt.plot(lamdas, cond_plain[M], 'o--', label=f'Plain M={M}')
    plt.plot(lamdas, cond_ridge[M], 's-', label=f'Ridge M={M}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Regularization lamda (log scale)')
plt.ylabel('Condition Number (log scale)')
plt.title('Condition Number Before vs After Ridge Regularization')
plt.legend(ncol=2, fontsize=8)
plt.show()
plt.savefig('condition_number_vs_lambda.png')

# Using sklearn Ridge for verification
max_diffs = {}
gamma_gen_all = lambda M: np.diag([1]*(M+1))
for M in Ms:
    max_diffs[M] = []
    A_train = vander_matrix(t_train, M)
    for lamda in lamdas:
        gamma1 = gamma_gen_all(M)
        w_closed = compute_omega(gamma1, A_train, lamda, y_train)
        ridge = Ridge(alpha=lamda, fit_intercept=False, solver='auto')
        ridge.fit(A_train, y_train)
        w_sklearn = ridge.coef_

        print("Closed form w:", w_closed)
        print("scikit-learn w:", w_sklearn)
        print("Difference:", np.abs(w_closed - w_sklearn))
        max_diff =  np.max(np.abs(w_closed - w_sklearn))
        max_diffs[M].append(max_diff)

plt.figure(figsize=(10, 6))
for M in Ms:
    plt.plot(lamdas, max_diffs[M], marker='o', label=f'M={M}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Regularization $\lambda$ (log scale)')
plt.ylabel(r'Max $|w_{closed} - w_{sklearn}|$')
plt.title('Equivalence Check: Maximum Difference')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('equivalence_check.png')
plt.show()

#Grid Search for Optimal (M, lambda)
min_test_mse = float('inf')
best_M = None
best_lambda = None

for M in Ms:
    for i, lamda in enumerate(lamdas):
        current_mse = test_mse[M][i]
        if current_mse < min_test_mse:
            min_test_mse = current_mse
            best_M = M
            best_lambda = lamda

print(f"\nOptimal parameters:")
print(f"  M = {best_M}")
print(f"  Regularizarion = {best_lambda}")
print(f"  Test MSE = {min_test_mse:.6f}")

#final reflections

cases = [
    (6, 0.0001, 'Best', 'g-'),
    (7, 0, 'Ill-cond', 'r--'),
    (7, 1, 'Overly reg', 'b-.')
]

t_plot = np.linspace(0.0, 1.0, 100)

plt.figure(figsize=(10,6))

for M, lamda, label, style in cases:
    A_train = vander_matrix(t_train, M)
    A_plot = vander_matrix(t_plot, M)
    gamma = gamma_gen(M)
    w = compute_omega(gamma, A_train, lamda, y_train)
    fit = A_plot @ w
    
    plt.plot(t_plot, fit, style, lw=2, label=f'{label}: M={M}, λ={lamda}')
    print(f"{label} coefficients (M={M}, λ={lamda}):\n  {w}\n  Max |coef|: {np.max(np.abs(w)):.2e}\n")

plt.plot(x_data, s, 'k', lw=2, label='True (Noise-free) Signal')
plt.scatter(t_train, y_train, c='orange', label='Training data', zorder=3, s=15)
plt.xlabel("t")
plt.ylabel("y")
plt.title("Comparison of Fits: Best, Poorly Conditioned, Over-Regularized")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.savefig('final_reflections.png')
