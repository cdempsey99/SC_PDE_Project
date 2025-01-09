import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ------------- The method of Paper 1 (modified PINN I'm gonna call it ) applied to the simple source function below (using Adam optim)---------

# Basic PyTorch Workflow
# 1. Define a model (MLP with sigmoid activation for hidden layers and linear output)
# 2. Define a Loss Function
# 3. Define an optimiser (Merlin, SGD)
# 4. Train the model

# In this example I will deal with the 2D Poisson eqn
# u_xx + u_yy = f(x, y)

# Let's take source f(x, y) = sin(pi * x) * sin(pi * y)
# This is handy as we have the same function as the exact solution in this case

# Define the multilayer perceptron class
class MLP(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        # Call the init method of the super class
        super(MLP, self).__init__()

        # Define the layers
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.output = nn.Linear(hidden_size2, output_size)

        # Initialize weights with larger values
        # This threw a spanner in the works
        #nn.init.xavier_uniform_(self.hidden1.weight, gain=5.0)
        #nn.init.xavier_uniform_(self.hidden2.weight, gain=5.0)
        #nn.init.xavier_uniform_(self.output.weight, gain=5.0)

    # Define forward pass
    def forward(self, x):

        # (Could try leaky relu as well)
        x = torch.tanh(self.hidden1(x)) # Apply activation function to first layer
        x = torch.tanh(self.hidden2(x)) # and second

        #x = torch.nn.functional.leaky_relu(self.hidden1(x), negative_slope=0.01)
        #x = torch.nn.functional.leaky_relu(self.hidden2(x), negative_slope=0.01)

        x = self.output(x) # Linear output layer

        return x

# Initialise the model
input_size = 2
hidden_size1 = 10
hidden_size2 = 10
output_size = 1

model = MLP(input_size, hidden_size1, hidden_size2, output_size)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# Learning scheduler
#scheduler = optim.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, verbose=True)

# Boundary-condition-obeying part of trial solution
def A(x, y):
    return 0

# Trial solution
def trial_solution(x, y, model_output):
    # I tried changing model_output to (model_output + 1) in order to bias the answer a bit more positively
    # as they are consistently too negative
    # But this had the opposite effect, so I switched it to -1
    # and this actually improved things somehow!

    # the term x*(1 - x)*y*(1 - y) has a very small maximum value on the grid so add a scaling factor
    scaling_factor = 1.0

    return A(x, y) + x * (1 - x) * y * (1 - y) * scaling_factor * (model_output)

# Source term f(x, y)
def f(x, y):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

# Exact solution function (designed to use tensors)
def psi_exact(x, y):
    return (1/np.pi**2) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

# Exact solution function (designed to use numpy arrays, for visualisation)
def psi_exact_np(x, y):
    return (-1/np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y)




# Define the loss function
def compute_loss(model, X, Y, f):

    X.requires_grad = True
    Y.requires_grad = True
    # Concatenate tensors
    inputs = torch.cat((X, Y), dim=1) # Shape: (N, 2)

    # Forward pass
    model_output = model(inputs)

    # Insert the models output into the trial solution
    Psi_pred = trial_solution(X, Y, model_output)

    # Compute the second derivatives using autograd
    psi_x = torch.autograd.grad(Psi_pred, X, grad_outputs=torch.ones_like(Psi_pred), create_graph=True)[0]
    psi_xx = torch.autograd.grad(psi_x, X, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]

    psi_y = torch.autograd.grad(Psi_pred, Y, grad_outputs=torch.ones_like(Psi_pred), create_graph=True)[0]
    psi_yy = torch.autograd.grad(psi_y, Y, grad_outputs=torch.ones_like(psi_y), create_graph=True)[0]

    # Residual
    residual = psi_xx + psi_yy - f(inputs[:, 0], inputs[:, 1])

    # Loss is the mean squared residual
    return torch.mean(residual ** 2)


# Generate training points
#num_points = 500
#x_train = np.random.rand(num_points, 1)   # x in [0, 1]
#y_train = np.random.rand(num_points, 1)   # y in [0, 1]
#XY_train = np.hstack((x_train, y_train))

# Generate more training points with slight bias towards the center
num_points = 1000
# Generate points with slight bias towards center
r = np.random.rand(num_points, 1) ** 0.5  # This creates more points near the center
theta = 2 * np.pi * np.random.rand(num_points, 1)
x_train = 0.5 + 0.5 * r * np.cos(theta)
y_train = 0.5 + 0.5 * r * np.sin(theta)

XY_train = np.hstack((x_train, y_train))

# Convert to PyTorch tensors
XY_train_tensor = torch.tensor(XY_train, dtype=torch.float32)
x_train_tensor = XY_train_tensor[:, 0:1]
y_train_tensor = XY_train_tensor[:, 1:]

# Find exact solution at these points
XY_train_exact = psi_exact(x_train_tensor, y_train_tensor)
XY_train_exact_tensor = torch.tensor(XY_train_exact, dtype=torch.float32).view(-1, 1)


losses = []

# Training loop
num_epochs = 100
for epoch in range(num_epochs):

    model.train()

    # Compute the loss
    loss = compute_loss(model, x_train_tensor, y_train_tensor, f)

    # Backward pass and optimisation
    optimizer.zero_grad()   # Zero the gradients before the backward pass
    loss.backward()    # Compute the gradients

    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()   # Update the model parameters

    # Keep track of losses for visualisation
    losses.append(loss.item())

    # Print the loss every 100 epochs for monitoring
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4e}')



# Visualise the result
x_grid = np.linspace(0, 1, 100)
y_grid = np.linspace(0, 1, 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
XY_grid = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
XY_grid_tensor = torch.tensor(XY_grid, dtype=torch.float32)

# Predict solution with the trained model
model.eval()
Psi_pred = model(XY_grid_tensor).detach().numpy()

# Reshape
Psi_pred_grid = Psi_pred.reshape(X_grid.shape)

# Plot the predicted solution
plt.figure(figsize=(8, 6))
plt.contourf(X_grid, Y_grid, Psi_pred_grid, 20, cmap='viridis')
plt.colorbar(label='Predicted Solution')
plt.title('Predicted Solution to Poisson Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Compare with the exact solution
Psi_exact_grid = psi_exact_np(X_grid, Y_grid)


# Check relative error ?

# Plot the exact solution
plt.figure(figsize=(8, 6))
plt.contourf(X_grid, Y_grid, Psi_exact_grid, 20, cmap='viridis')
plt.colorbar(label='Exact Solution')
plt.title('Exact Solution to Poisson Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# Plot the losses versus epochs
plt.plot(losses, label='Losses Over Epochs')
plt.title("Losses over Epochs")
plt.xlabel("Epoch number")
plt.ylabel("Loss")
plt.legend()
plt.show()



# Ok this is producing results, but poor ones. Ideas to improve the model:
# 1. Increase complexity of model, there are very few weights and biases currently
# 2. Change activation function, to RelU maybe?
# 3. Change the learning rate
# 4. Change the optimiser
# 5. Use more grid points, potentially not uniformly chosen as well
# 6. Visualise the residual

