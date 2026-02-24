import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
import optax
import jax
import time

def generate_data(scale_noise, N, seed=None):
    """Generate data"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, N)
    s = 1 + t + t*t
    n = rng.normal(loc=0.0, scale=scale_noise, size=N)
    y = s + n
    return t, y, s, n

def vander_matrix(t, M):
    """Produce design matrix"""
    return jnp.vander(t, N=M+1, increasing=True)

def compute_omega(M, A, lamda, y):
    gamma = jnp.eye(M+1)  
    lhs_1 = A.T @ A
    lhs_2 = lamda * gamma.T @ gamma
    lhs = lhs_1 + lhs_2
    rhs = A.T @ y
    w = jnp.linalg.solve(lhs, rhs)
    return w

def model(w, A):
    """Return model evaluation"""
    return A @ w

def loss(w, A, y, lamda):
    """Compute loss function: ||A w - y||^2 + lamda ||w||^2"""
    y_pred = model(w, A)
    res = y_pred - y
    norm_1 = jnp.linalg.norm(res)**2
    norm_2 = jnp.linalg.norm(w)**2
    loss =  norm_1 + lamda * norm_2
    return loss

def mse(w, A, y):
    """Compute mse function: 1/N ||A w - y||^2"""
    y_pred = model(w, A)
    res = y_pred - y
    return jnp.mean(res**2)

# @jax.jit
def one_step(opt_state, w, optim, A_train, y_train):
    """SGD update"""
    val, g = value_and_grad(w, A_train, y_train)
    updates, opt_state = optim.update(g, opt_state, w)
    w = optax.apply_updates(w, updates)
    return opt_state, w, val


def minibatches(key, x, y, batch_size=None):

    """Generator function that yields minibatches"""
    if batch_size is None or batch_size >= x.shape[0]:
        yield x, y
    else:
        # shuffle training data
        idx = jax.random.permutation(key, x.shape[0])
        x, y = x[idx], y[idx]
        
        for s in range(0, x.shape[0], batch_size):
            yield x[s:s+batch_size], y[s:s+batch_size]


def train_model(key, w_init, epochs, w, batch_size, target_mse, optim):
    """
    This function does the following:
    1. Calculates train and test MSE per epoch using SGD.
    2. Calculates the distance between weights obtained using SGD and closed form solution
    """
    opt_state = optim.init(w_init)
    train_mse_log = []
    test_mse_log = []
    distance_log = []
    epoch_log = []

    for e in range(epochs):
        key, subkey = jax.random.split(key)
        
        # Iterate over batches using the generator
        for A_batch, y_batch in minibatches(subkey, A_train, y_train, batch_size):
            opt_state, w_init, val = one_step(opt_state, w_init, optim, A_batch, y_batch)
    
        if e % 100 == 0 or e < 10:
            train_mse_sgd = mse(w_init, A_train, y_train)
            test_mse_sgd = mse(w_init, A_test, y_test)
            distance = jnp.linalg.norm(w_init - w)
            train_mse_log.append(train_mse_sgd)
            test_mse_log.append(test_mse_sgd)
            distance_log.append(distance)
            epoch_log.append(e)
 
    # print("SGD weights:", w_init)
    # print("Closed-form weights:", w)
    # print("L2 distance:", jnp.linalg.norm(w_init - w))

    return  train_mse_log, test_mse_log, distance_log, epoch_log

#Generate and split data
t_all, y_all, s_all, n_all = generate_data(0.0, 1000, seed=0)
t_train, t_test, y_train, y_test = train_test_split(t_all,y_all,test_size=0.2,random_state=42)

# plt.figure()
# plt.plot(t_train, y_train, 'bo', label='training data')
# plt.plot(t_test, y_test, 'r+', label='testing data')
# plt.legend(); plt.xlabel("t"); plt.ylabel("value"); plt.title("HW4 data + split")
# plt.show()

# Initializations
M  = 3
lamda = 0.0
alpha = 0.09
Ntrain = len(t_train)
B = 32
epochs = 1000
target_mse = 1e-2

#1. d Sanity check
A_train = vander_matrix(t_train, M)
A_test = vander_matrix(t_test, M)
w = compute_omega(M, A_train, lamda, y_train)
# mse_train = mse(w, A_train, y_train)
# mse_test = mse(w, A_test, y_test)
# print(f"Sanity check (noise-free data):")
# print(f"Train MSE: {mse_train}")
# print(f"Test MSE:  {mse_test}")

value_and_grad = jax.jit(jax.value_and_grad(mse))

# Optimizing with sgd
# optim = optax.sgd(alpha)

scheduler = optax.exponential_decay(
    init_value=alpha,
    transition_steps=1000,
    decay_rate=0.97
)

optimizers = {
    "SGD constant LR": optax.sgd(alpha),
    "Momentum = 0.9": optax.sgd(alpha, momentum=0.9),
    "with scheduler": optax.sgd(scheduler)
}


results = {}

for name, optim in optimizers.items():
    w_init = jnp.zeros(M+1)
    key = jax.random.PRNGKey(0)
    train_mse_log, test_mse_log, distance_log, epoch_log = train_model(key, w_init, epochs, w, B, target_mse, optim)
    results[name] = {
        'train_mse': train_mse_log,
        'test_mse': test_mse_log,
        'distance': distance_log,
        'epochs': epoch_log,
        'final_train_mse': train_mse_log[-1],
        'final_test_mse': test_mse_log[-1]
    }


plt.figure()

for name, data in results.items():
    plt.plot(data['epochs'], data['test_mse'], label=name, marker='o', markersize=3)
plt.xlabel('Epoch ')
plt.ylabel('MSE')
plt.legend()
plt.title('Test MSE vs Epoch')
plt.yscale('log') 
plt.tight_layout()
plt.savefig("1_g.png")
plt.show()
