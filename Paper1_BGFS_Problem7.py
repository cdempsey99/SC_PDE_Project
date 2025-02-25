import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ------------- The method of Paper 1 (modified PINN I'm gonna call it ) applied to Problem 7 from the paper ---------
# using BGFS as well

# WIP

# Define multilayer perceptron class
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.output = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))  # Using tanh for stability
        x = torch.tanh(self.hidden2(x))
        x = self.output(x)
        return x


# Initialize the model
model = MLP(input_size=2, hidden_size1=50, hidden_size2=50, output_size=1)

# Generate training points (keep your existing code for this)
num_points = 1000
x_train = np.random.rand(num_points, 1)
y_train = np.random.rand(num_points, 1)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32, requires_grad=True)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, requires_grad=True)

# Initialize L-BFGS optimizer
optimizer = optim.LBFGS(model.parameters(),
                        max_iter=20,  # max iterations per optimization step
                        max_eval=25,  # max evaluations per optimization step
                        tolerance_grad=1e-7,  # termination tolerance on first order optimality
                        tolerance_change=1e-9,  # termination tolerance on function value/parameter changes
                        history_size=50,  # update history size
                        line_search_fn='strong_wolfe')  # use strong Wolfe line search

# Part of trial solution obeying the boundary conditions by defintion
def B(x, y):
    return 2 * y * np.sin(np.pi * x)


def trial_solution(x, y, model):
    # Combine x and y into a 2D input tensor
    xy = torch.stack((x, y), dim=1).requires_grad_(True)  # Shape: (batch_size, 2)
    xy_1 = torch.stack((x, torch.ones_like(y)), dim=1).requires_grad_(True)  # Shape: (batch_size, 2)

    # Retain gradients for xy_1
    xy_1.retain_grad()

    # Model outputs
    n_xy = model(xy)  # N(x, y)
    n_x_1 = model(xy_1)  # N(x, 1)

    # Compute ∂N/∂y at (x, 1)
    n_x_1.sum().backward(retain_graph=True)  # Sum ensures proper gradient computation
    partial_derivative_y = xy_1.grad[:, 1]  # Gradient w.r.t. y (2nd input)

    # Trial solution
    trial = x * (1 - x) * y * (n_xy - n_x_1 - partial_derivative_y)
    return trial


# Source term on rhs of PDE
def f(x, y):
    return (2 - torch.pi**2 * y**2) * torch.sin(torch.pi * x)


# For visualisation
def psi_exact_np(x, y):
    return np.sin(np.pi * x) * (y**2/np.pi**2 - y**4/2*np.pi**2 + y - 1)



# Modified training loop for L-BFGS
def closure():
    optimizer.zero_grad()

    # Trial solution
    inputs = torch.cat((x_train_tensor, y_train_tensor), dim=1)  # Combine x and y
    Psi_pred = trial_solution(inputs[:, 0], inputs[:, 1], model)  # Pass x and y separately

    # Compute derivatives
    Psi_x = torch.autograd.grad(Psi_pred, x_train_tensor,
                                grad_outputs=torch.ones_like(Psi_pred),
                                create_graph=True)[0]
    Psi_xx = torch.autograd.grad(Psi_x, x_train_tensor,
                                 grad_outputs=torch.ones_like(Psi_x),
                                 create_graph=True)[0]

    Psi_y = torch.autograd.grad(Psi_pred, y_train_tensor,
                                grad_outputs=torch.ones_like(Psi_pred),
                                create_graph=True)[0]
    Psi_yy = torch.autograd.grad(Psi_y, y_train_tensor,
                                 grad_outputs=torch.ones_like(Psi_y),
                                 create_graph=True)[0]

    # Residual
    residual = Psi_xx + Psi_yy - f(x_train_tensor, y_train_tensor)
    loss = torch.mean(residual ** 2)

    # Backward pass
    loss.backward()

    return loss



# Training loop
num_epochs = 7  # Usually need fewer epochs with L-BFGS
losses = []

print("Starting training...")
for epoch in range(num_epochs):
    # The closure is called multiple times per step
    loss = optimizer.step(closure)
    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4e}')

# Visualization code (keep your existing plotting code)
x_grid = np.linspace(0, 1, 100)
y_grid = np.linspace(0, 1, 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
XY_grid = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
XY_grid_tensor = torch.tensor(XY_grid, dtype=torch.float32)

# Evaluate solution
model.eval()
with torch.no_grad():
    Psi_pred = model(XY_grid_tensor)
    Psi_pred = trial_solution(
        XY_grid_tensor[:, 0:1],
        XY_grid_tensor[:, 1:2],
        Psi_pred
    ).numpy()
    Psi_pred_grid = Psi_pred.reshape(X_grid.shape)

# Plot solution
plt.figure(figsize=(10, 8))
plt.contourf(X_grid, Y_grid, Psi_pred_grid, 20, cmap='viridis')
plt.colorbar(label='Predicted Solution')
plt.title('Solution to Poisson Equation using L-BFGS')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Compare with the exact solution
Psi_exact_grid = psi_exact_np(X_grid, Y_grid)

# Plot the exact solution
plt.figure(figsize=(8, 6))
plt.contourf(X_grid, Y_grid, Psi_exact_grid, 20, cmap='viridis')
plt.colorbar(label='Exact Solution')
plt.title('Exact Solution to Poisson Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# Plot loss history
plt.figure(figsize=(10, 6))
plt.semilogy(losses)
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.grid(True)
plt.show()