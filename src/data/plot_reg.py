#%%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")
# -----------------------------
# 1. Generate dataset
# -----------------------------
n = 500
x1 = np.random.randn(n)
x2 = np.random.randn(n)
x3 = (x1 + x2) / 2

X = np.vstack([x1, x2, x3]).T

# True model
y = x1 + 2 * x2

# -----------------------------
# 2. Define evaluation functions
# -----------------------------
def predict(X, w):
    return X @ w

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def l1(w):
    return np.sum(np.abs(w))

def l2(w):
    return np.sum(w**2)

# -----------------------------
# 3. Sweep over weights
# -----------------------------
w1_vals = np.linspace(-1, 1, 80)
w2_vals = np.linspace(-1, 2, 80)
w3_fixed = 0.0  # you can vary this too

MSE_grid = np.zeros((len(w1_vals), len(w2_vals)))
L1_grid  = np.zeros_like(MSE_grid)
L2_grid  = np.zeros_like(MSE_grid)

for i, w1 in enumerate(w1_vals):
    for j, w2 in enumerate(w2_vals):
        w = np.array([w1, w2, w3_fixed])
        y_pred = predict(X, w)
        MSE_grid[i, j] = mse(y, y_pred)
        L1_grid[i, j]  = l1(w)
        L2_grid[i, j]  = l2(w)

# -----------------------------
# 4. Plot error surface
# -----------------------------
W1, W2 = np.meshgrid(w1_vals, w2_vals)

fig = plt.figure(figsize=(14, 5))

# --- MSE ---
ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(W1, W2, MSE_grid.T, cmap='viridis')
ax.set_title("MSE landscape")
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_zlabel("MSE")

# --- L1 ---
ax = fig.add_subplot(132, projection='3d')
ax.plot_surface(W1, W2, L1_grid.T, cmap='magma')
ax.set_title("L1 regularization")
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_zlabel("L1")

# --- L2 ---
ax = fig.add_subplot(133, projection='3d')
ax.plot_surface(W1, W2, L2_grid.T, cmap='plasma')
ax.set_title("L2 regularization")
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_zlabel("L2")

plt.tight_layout()
plt.show()

# -----------------------------
# 5. Print perfect weights
# -----------------------------
w_star = np.array([1, 2, 0])
print("Perfect weights:", w_star)
print("L1(w*) =", l1(w_star))
print("L2(w*) =", l2(w_star))

# %%
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Create a simple 2D regression problem
# ----------------------------------------------------
np.random.seed(0)
n = 200

x1 = np.random.randn(n)
x2 = np.random.randn(n)
X = np.vstack([x1, x2]).T

# True model
w_true = np.array([1.0, 2.0])
y = X @ w_true + 0.1 * np.random.randn(n)

# ----------------------------------------------------
# 2. Define loss function
# ----------------------------------------------------
def loss(w):
    y_pred = X @ w
    return np.mean((y - y_pred)**2)

# ----------------------------------------------------
# 3. Grid of weights
# ----------------------------------------------------
w1_vals = np.linspace(0, 3, 200)
w2_vals = np.linspace(0, 3, 200)

W1, W2 = np.meshgrid(w1_vals, w2_vals)
Loss = np.zeros_like(W1)

for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w = np.array([W1[i, j], W2[i, j]])
        Loss[i, j] = loss(w)

# ----------------------------------------------------
# 4. Plot contours + regularization ball
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))

# Loss contours
contours = ax.contour(W1, W2, Loss, levels=10, cmap="viridis")
ax.clabel(contours, inline=True, fontsize=8)

# L2 ball
radius = 1.5
theta = np.linspace(0, 2*np.pi, 300)
ax.plot(radius*np.cos(theta), radius*np.sin(theta), 'r', label="L2 ball")

# L1 ball (diamond)
l1_r = 1.5
ax.plot([0, l1_r, 0, -l1_r, 0],
        [l1_r, 0, -l1_r, 0, l1_r],
        'm--', label="L1 ball")

# True optimum
ax.scatter([w_true[0]], [w_true[1]], color='black', s=60, label="True w*")

ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_title("Loss contours + Regularization balls")
ax.set_aspect("equal")
ax.legend()
ax.grid(False)
plt.savefig("balls.pdf")
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Simulated training dynamics
np.random.seed(0)
epochs = 40

# Training error decreases smoothly with small noise
train_error = np.exp(-np.linspace(0, 3, epochs)) + 0.02 * np.random.randn(epochs)

# Validation error decreases, then increases (overfitting)
val_error = np.exp(-np.linspace(0, 2, epochs)) + 0.1 * np.random.randn(epochs)
val_error[20:] += 0.03 * np.arange(epochs - 20)  # induce overfitting

# Early stopping point = epoch with minimum validation error
early_stop_epoch = np.argmin(val_error)

plt.figure(figsize=(10, 6))
plt.plot(train_error, label="Training Error", linewidth=2)
plt.plot(val_error, label="Validation Error", linewidth=2)

# Mark early stopping
#plt.axvline(early_stop_epoch, color="red", linestyle="--", label=f"Early Stop (epoch {early_stop_epoch})")

plt.title("Training vs Validation Error with Early Stopping")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig("early_stopping_no_line.pdf")

    # %%
