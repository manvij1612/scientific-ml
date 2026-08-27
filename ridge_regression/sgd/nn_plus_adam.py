import numpy as np
import matplotlib.pyplot as plt
import jax.nn as jnn
from flax import nnx  # NNX = Neural Networks for jaX
import jax.numpy as jnp
import jax
import optax

def f_star(t):
    return np.exp(-3.0*t) * np.sin(8*np.pi*t)


class TwoLayerMLP(nnx.Module):
    """Class to define a 2 layer Neural Network"""
    def __init__(self, width, rngs):
        # self.d1 = nnx.Linear(1, width, rngs=rngs)   # input size =1
        # self.d2 = nnx.Linear(width, 1, rngs=rngs)   # output size=1
        self.d1 = nnx.Linear(
            1, width, 
            rngs=rngs,
            kernel_init=nnx.initializers.normal(stddev=0.05),  # small input weights
            bias_init=nnx.initializers.uniform(scale=1.0)      # uniform [-1, 1]
        )
        self.d2 = nnx.Linear(
            width, 1, 
            rngs=rngs,
            kernel_init=nnx.initializers.normal(stddev=0.05),  # small output weights
            bias_init=nnx.initializers.zeros                   # zero output bias
        )

    def __call__(self, x):
        x = self.d1(x)
        x = jnn.tanh(x)
        x = self.d2(x)
        return x.squeeze(-1)  # (N,)

# Loss (MSE) takes the model directly
def loss_fn(model_mlp,  x, y):
    preds = model_mlp(x)
    return jnp.mean((preds - y) ** 2)

@jax.jit
def one_step(opt_state, model, x_train, y_train, params):
    """optimizer update"""
    loss_val, grads = value_and_grad(model, x_train, y_train)
    updates, opt_state = optim.update(grads, opt_state, params)
    # apply updates
    params = optax.apply_updates(params, updates)
    return opt_state, params, loss_val

rng = np.random.default_rng(0)
t_train = rng.uniform(0.0, 1.0, size=128)
y_train = f_star(t_train)
t_test = np.linspace(0.0, 1.0, 2048, endpoint=True)
y_test = f_star(t_test)

x_train = jnp.array(t_train).reshape(-1, 1)
y_train = jnp.array(y_train)

x_test = jnp.array(t_test).reshape(-1, 1)
y_test = jnp.array(y_test)


# create model
m = 16
alpha = 0.0006
rngs  = nnx.Rngs(0)
model = TwoLayerMLP(width=m, rngs=rngs)

value_and_grad = nnx.value_and_grad(loss_fn)


# adam loop (full batch)

optim = optax.adam(alpha)
paramdef, params = nnx.split(model, nnx.Param)
opt_state = optim.init(params)

epochs = 200000
train_mse_log = []
test_mse_log = []
for epoch in range(epochs):

    # gradient taken over nnx.Param
    opt_state, params, loss_val = one_step(opt_state, model, x_train, y_train, params)
    model = nnx.merge(paramdef, params)
    if epoch % 10 == 0:  # Evaluate every 10 epochs
        train_mse_log.append(loss_val)
        test_pred = model(x_test)
        test_mse = jnp.mean((test_pred - y_test)**2)
        test_mse_log.append(test_mse)

print("Final train MSE:", train_mse_log[-1])
print("Final test MSE:", test_mse_log[-1])

pred_test = model(x_test)

plt.figure(figsize=(7,4))
plt.plot(t_test, y_test, label="True f*", linewidth=2)
plt.plot(t_test, pred_test, label="NN prediction", linewidth=2)
plt.scatter(t_train, y_train, s=15, c='red', label="Training pts")
plt.legend()
plt.title("Function Approximation with 2-layer NN (m=16)")
plt.xlabel("t")
plt.ylabel("value")
plt.tight_layout()
plt.savefig("2_e.png")
plt.show()


plt.figure(figsize=(6,4))
plt.plot(train_mse_log, label="Train MSE")
plt.plot(test_mse_log, label="Test MSE")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE (log scale)")
plt.title("MSE vs Epoch (m=16)")
plt.legend()
plt.tight_layout()
plt.savefig("2_e_2.png")
plt.show()
