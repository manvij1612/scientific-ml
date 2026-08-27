import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def sho_analytic(t, A=1.0, k=1.0):
    """ Analytic solution for the 1D harmonic oscillator.

    Initial conditions are
    q(0) = A
    p(0) = 0

    Parameters are
    mass m = 1
    spring constant k > 0
    """

    omega = np.sqrt(k)
    q     = A * np.cos(omega*t)
    p     = -A*omega*np.sin(omega*t)
    return q, p

def sho_rhs(q, p, k=1.0):
    """Right-hand side of the

    dq/dt = p
    dp/dt = -k q
    """

    dqdt = p
    dpdt = -k * q
    return dqdt, dpdt

def rk4_step(q, p, dt, k=1.0):
    """Single RK4 step taking us from time t to t+dt"""
    k1_q, k1_p = sho_rhs(q, p, k)
    k2_q, k2_p = sho_rhs(q + 0.5*dt*k1_q, p + 0.5*dt*k1_p, k)
    k3_q, k3_p = sho_rhs(q + 0.5*dt*k2_q, p + 0.5*dt*k2_p, k)
    k4_q, k4_p = sho_rhs(q + dt*k3_q, p + dt*k3_p, k)

    q_next = q + (dt/6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
    p_next = p + (dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
    return q_next, p_next

def sho_numerical(A=1.0, k=1.0, T=10.0, dt=0.01):
    """ Numerical solution of the harmonic oscillator. """

    t = np.arange(0.0, T + dt, dt)
    q = np.zeros_like(t)
    p = np.zeros_like(t)

    # initial conditions
    q[0] = A
    p[0] = 0.0

    for n in range(len(t) - 1):
        q[n+1], p[n+1] = rk4_step(q[n], p[n], dt, k)

    return t, q, p

class hamiltonian_nn(nn.Module):
    """
    Neural network to learn hamiltoninan H(q,p).
    """
    def __init__(self, hidden_dim=64):
        super(hamiltonian_nn, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, qp):
        # qp = self.flatten(qp)
        # logits = self.linear_relu_stack(qp)
        # return logits

        return self.net(qp)
    
def compute_hamiltonian(model, qp):
    """
    Compute Hamiltonian H(q,p) using the neural network model.
    """
    H = model(qp)

    grad_h_tuple = torch.autograd.grad(outputs = H, inputs = qp, grad_outputs = torch.ones_like(H), retain_graph=True, create_graph=True)

    grad_h = grad_h_tuple[0]  # (batch_size, 2)
    #Extract derivatives
    dH_dq = grad_h[:, 0:1]  # \partial H/\partial q
    dH_dp = grad_h[:, 1:2]  # \partial H/ \partial p
    
    # Hamilton's equations
    dqdt = dH_dp
    dpdt = -dH_dq
    
    return dqdt, dpdt


def loss_fn(model, qp, y_true):
    """
    Mean squared error loss between predicted and true derivatives.
    """
    #using a clone for gradient computation
    qp_grad = qp.clone().requires_grad_(True)

    #compute predictions
    dqdt_pred, dpdt_pred = compute_hamiltonian(model, qp_grad)

    #Concat predictions i.e (batch_size, 1) -> (batch_size, 2)
    y_pred = torch.cat([dqdt_pred, dpdt_pred], dim=1)

    #compute MSE loss
    loss = torch.mean((y_pred - y_true)**2)

    return loss

def integrate_learned_dynamics(model, q0, p0, T, dt, k=1.0):
    """
    Integrate the learned Hamiltonian dynamics forward in time using RK4.
    """
    model.eval()
    
    t = np.arange(0.0, T + dt, dt)
    q_learned = np.zeros_like(t)
    p_learned = np.zeros_like(t)
    
    #Initial conditions
    q_learned[0] = q0
    p_learned[0] = p0
    
    #Integrate using RK4 with learned dynamics
    
    for n in range(len(t) - 1):
        # Current state
        q_curr = q_learned[n]
        p_curr = p_learned[n]
        
        #RK4 integration using learned Hamiltonian
        # k1
        qp1 = torch.tensor([[q_curr, p_curr]], dtype=torch.float32, requires_grad=True)
        dqdt1, dpdt1 = compute_hamiltonian(model, qp1)
        k1_q = dqdt1.item()
        k1_p = dpdt1.item()
        
        # k2
        qp2 = torch.tensor([[q_curr + 0.5*dt*k1_q, p_curr + 0.5*dt*k1_p]], 
                            dtype=torch.float32, requires_grad=True)
        dqdt2, dpdt2 = compute_hamiltonian(model, qp2)
        k2_q = dqdt2.item()
        k2_p = dpdt2.item()
        
        # k3
        qp3 = torch.tensor([[q_curr + 0.5*dt*k2_q, p_curr + 0.5*dt*k2_p]], 
                            dtype=torch.float32, requires_grad=True)
        dqdt3, dpdt3 = compute_hamiltonian(model, qp3)
        k3_q = dqdt3.item()
        k3_p = dpdt3.item()
        
        # k4
        qp4 = torch.tensor([[q_curr + dt*k3_q, p_curr + dt*k3_p]], 
                            dtype=torch.float32, requires_grad=True)
        dqdt4, dpdt4 = compute_hamiltonian(model, qp4)
        k4_q = dqdt4.item()
        k4_p = dpdt4.item()
        
        # Update
        q_learned[n+1] = q_curr + (dt/6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
        p_learned[n+1] = p_curr + (dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)

    # Get exact solution for comparison
    q_exact, p_exact = sho_analytic(t, A=q0, k=k)
    
    return t, q_learned, p_learned, q_exact, p_exact


times, q_num, p_num = sho_numerical(A=1.0, k=1.0, T=10.0, dt=0.01)
q_exact, p_exact    = sho_analytic(times, A=1.0, k=1.0)

# plt.plot(times,p_num, 'r', label='numerical, p')
# plt.plot(times,p_exact, 'b--', label='exact, p')
plt.plot(times,q_num, 'k', label='numerical, q')
plt.plot(times,q_exact, 'r-.', label='exact, q')
plt.legend()
plt.savefig('sho_numerical_vs_analytic.png')
plt.close()

# Compute the error
error = q_num - q_exact

# Compute maximum absolute error
max_abs_error = np.max(np.abs(error))

# Plot the error
plt.figure(figsize=(10, 6))
plt.plot(times, error, 'b-', linewidth=1.5)
plt.xlabel('Time t', fontsize=12)
plt.ylabel('Error e(t) = q_num(t) - q_exact(t)', fontsize=12)
plt.title('Error in Numerical Solution vs Exact Solution', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('sho_error.png', dpi=150)
plt.close()

print(f"Maximum absolute error: max_t |e(t)| = {max_abs_error:.6e}")

#part 2 : Generate data for learning the dynamics
k =1.0
A = [0.5, 1.0, 1.5, 2.0]
T = 10.0
dt = 0.01

inputs = []
outputs = []

for j in A:
    times, q_num, p_num = sho_numerical(j, k, T + dt , dt)
    q_exact, p_exact    = sho_analytic(times, j, k)
    
    for n in range(len(times)):
        input_pair = [q_exact[n], p_exact[n]]
        dqdt = p_exact[n]
        dpdt = -k * q_exact[n]
        output_pair = [dqdt, dpdt]
        inputs.append(input_pair)
        outputs.append(output_pair)

X = np.array(inputs)
Y = np.array(outputs) 

np.random.seed(1612)
indices = np.random.permutation(len(X))

#split data 80-20
train_size = int(0.8 * len(X))
train_indices = indices[:train_size]
val_indices   = indices[train_size:]
x_train, y_train = X[train_indices], Y[train_indices]
x_val, y_val = X[val_indices], Y[val_indices]

print(f"Training set size: {len(x_train)} points")
print(f"Validation set size: {len(x_val)} points")

#part 3: Learning dynamics with a neural Hamiltonian

#Convert data into torch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

#Create dataloaders

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

#initialise model
model = hamiltonian_nn(64)
lr = 1e-4
optim = torch.optim.Adam(model.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.5)

epochs = 800
train_losses =[]
val_losses =[]

#HNN training loop

for i in range(epochs):

    #training
    model.train()
    train_loss_epoch = 0.0
    for batch_x, batch_y in train_loader:
        optim.zero_grad() #for grads to not accumulate across interations (zero before next batch)
        loss = loss_fn(model, batch_x, batch_y)
        loss.backward() # backward pass
        optim.step() #gradient descent
        train_loss_epoch += loss.item() * len(batch_x)

    train_loss_epoch /= len(x_train)
    train_losses.append(train_loss_epoch)

    #validation
    model.eval()
    val_loss_epoch = 0.0
    for batch_x, batch_y in val_loader:
        loss = loss_fn(model, batch_x, batch_y)
        val_loss_epoch += loss.item() * len(batch_x)

    val_loss_epoch /= len(x_val)
    val_losses.append(val_loss_epoch)

    scheduler.step()

#Plot training and validation losses

plt.figure(figsize=(10, 6))
plt.semilogy(train_losses, 'b-', label='Training Loss', linewidth=2)
plt.semilogy(val_losses, 'r-', label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Hamiltonian Neural Network: Training and Validation Loss', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hnn_training_loss.png', dpi=150)
plt.close()

print(f"\nFinal Training Loss: {train_losses[-1]:.6e}")
print(f"Final Validation Loss: {val_losses[-1]:.6e}")

#part 4: Using your new model

A_test = 1.25
T_test = 10.0
dt_test = 0.01

t, q_learned, p_learned, q_exact, p_exact = integrate_learned_dynamics(
    model, A_test, 0.0, T_test, dt_test, k=1.0
)

# Compute errors
error_q = q_learned - q_exact
error_p = p_learned - p_exact
mae_q = np.mean(np.abs(error_q))
mae_p = np.mean(np.abs(error_p))
max_error_q = np.max(np.abs(error_q))
max_error_p = np.max(np.abs(error_p))

print(f"\nTest Amplitude: A = {A_test}")
print(f"Integration time: T = {T_test}")
print(f"\nPosition Error (q):")
print(f"  Mean Absolute Error: {mae_q:.6e}")
print(f"  Max Absolute Error:  {max_error_q:.6e}")
print(f"\nMomentum Error (p):")
print(f"  Mean Absolute Error: {mae_p:.6e}")
print(f"  Max Absolute Error:  {max_error_p:.6e}")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

#Position trajectory
axes[0, 0].plot(t, q_exact, 'b-', label='Exact', linewidth=2)
axes[0, 0].plot(t, q_learned, 'r--', label='HNN', linewidth=2, alpha=0.8)
axes[0, 0].set_xlabel('Time t', fontsize=11)
axes[0, 0].set_ylabel('Position q(t)', fontsize=11)
axes[0, 0].set_title(f'Position Trajectory (A = {A_test})', fontsize=12)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

#Momentum trajectory
axes[0, 1].plot(t, p_exact, 'b-', label='Exact', linewidth=2)
axes[0, 1].plot(t, p_learned, 'r--', label='HNN', linewidth=2, alpha=0.8)
axes[0, 1].set_xlabel('Time t', fontsize=11)
axes[0, 1].set_ylabel('Momentum p(t)', fontsize=11)
axes[0, 1].set_title(f'Momentum Trajectory (A = {A_test})', fontsize=12)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Phase spacee
axes[1, 0].plot(q_exact, p_exact, 'b-', label='Exact', linewidth=2)
axes[1, 0].plot(q_learned, p_learned, 'r--', label='HNN', linewidth=2, alpha=0.8)
axes[1, 0].set_xlabel('Position q', fontsize=11)
axes[1, 0].set_ylabel('Momentum p', fontsize=11)
axes[1, 0].set_title('Phase Space Trajectory', fontsize=12)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axis('equal')

# Errors
axes[1, 1].plot(t, error_q, 'b-', label='Position Error', linewidth=1.5)
axes[1, 1].plot(t, error_p, 'r-', label='Momentum Error', linewidth=1.5)
axes[1, 1].set_xlabel('Time t', fontsize=11)
axes[1, 1].set_ylabel('Error', fontsize=11)
axes[1, 1].set_title('Trajectory Errors', fontsize=12)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('hnn_generalization_test.png', dpi=150)
plt.close()

#Going beyod T=10
T_extrapolate = [20.0, 50.0, 100.0]
A_test = 1.0

fig, axes = plt.subplots(len(T_extrapolate), 2, figsize=(14, 4*len(T_extrapolate)))

for i, T_ext in enumerate(T_extrapolate):
    t, q_learned, p_learned, q_exact, p_exact = integrate_learned_dynamics(
        model, A_test, 0.0, T_ext, dt_test, k=1.0
    )
    
    error_q = q_learned - q_exact
    mae_q = np.mean(np.abs(error_q))
    max_error_q = np.max(np.abs(error_q))
    
    print(f"\nExtrapolation to T = {T_ext}:")
    print(f"  Mean Absolute Error (q): {mae_q:.6e}")
    print(f"  Max Absolute Error (q):  {max_error_q:.6e}")
    
    # Trajectory
    axes[i, 0].plot(t, q_exact, 'b-', label='Exact', linewidth=2)
    axes[i, 0].plot(t, q_learned, 'r--', label='HNN', linewidth=2, alpha=0.8)
    axes[i, 0].set_xlabel('Time t', fontsize=11)
    axes[i, 0].set_ylabel('Position q(t)', fontsize=11)
    axes[i, 0].set_title(f'T = {T_ext} (A = {A_test})', fontsize=12)
    axes[i, 0].legend(fontsize=10)
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 0].axvline(x=10, color='gray', linestyle=':', linewidth=1.5, label='Training limit')
    
    # Error
    axes[i, 1].plot(t, np.abs(error_q), 'k-', linewidth=1.5)
    axes[i, 1].set_xlabel('Time t', fontsize=11)
    axes[i, 1].set_ylabel('|Error in q(t)|', fontsize=11)
    axes[i, 1].set_title(f'Absolute Error (T = {T_ext})', fontsize=12)
    axes[i, 1].set_yscale('log')
    axes[i, 1].grid(True, alpha=0.3)
    axes[i, 1].axvline(x=10, color='gray', linestyle=':', linewidth=1.5)

plt.tight_layout()
plt.savefig('hnn_extrapolation_test.png', dpi=150)
plt.close()
