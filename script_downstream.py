#%%
import torch
from rdkit import RDLogger
import argparse
RDLogger.DisableLog('rdApp.error')
#%%
from src.model.builder import ModelBuilder
from src.model.checkpoint import ModelCheckpoint
from src.data.treat_data import load_data, make_loader
import os
import random
import numpy as np
import torch.nn as nn

import optuna

import torch
from torch.utils.data import TensorDataset, DataLoader

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.utils.evaluations import get_all_embeddings_downstream

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
parser = argparse.ArgumentParser(description="Downstream evaluation script for molecular graph generation models.")
parser.add_argument(
    "--model_name",
    type=str,
    default="vae_PubChem16_batch64_arom1_hfc1_lat512_enc512_128_4_5_dec5_0.0_reinj1_regw0.0_attr1_lr0.0001_0.1_wd0.0_b0.0001_200_cycle200_drop0.0_rne4.0_fc1_gl0_weloss1.25",
    help="Name of the trained model to evaluate (must correspond to a file in saved_models/)",
)

parser.add_argument(
    "--idx_target",
    type=int,
    default=0,
    help="Index of the target variable to predict: 0 for mu, 4 for Homo-Lumo gap, 7 for U0 (0=BIC for BayesianNetAsia)"
)

args = parser.parse_args()
model_name = args.model_name
model_path = os.path.join("saved_models", model_name)
checkpoint = ModelCheckpoint.load(model_path)
model = ModelBuilder.build_from_checkpoint(model_path)
model = model.to(device)
model = model.eval()
checkpoint_dataset = checkpoint.config_state_dict.get("dataset", "QM9NoHydro")
dataset = checkpoint_dataset if checkpoint_dataset.startswith("BayesianNet") else "QM9NoHydro"
splits = [0.8, 0.1, 0.1]

#%%
with_aromatic = ("arom1" in model_name)
train_ds, val_ds, test_ds = load_data(dataset, splits=splits, data_root="data", with_aromatic=with_aromatic)

#%%
test_loader = make_loader(test_ds, batch_size=256, shuffle=False, workers=0)
val_loader = make_loader(val_ds, batch_size=256, shuffle=False, workers=0)
train_loader = make_loader(train_ds, batch_size=256, shuffle=False, workers=0)

#%%
embeddings_tr, _, _, targets_tr = get_all_embeddings_downstream(
    num_batches=None,
    loader=train_loader,
    model_name=model_name,
    model=model,
    save_targets=True,
    force_recompute=False,
    splits="train",
    dataset=dataset,
)

embeddings_val, _, _, targets_val = get_all_embeddings_downstream(
    num_batches=None,
    loader=val_loader,
    model_name=model_name,
    model=model,
    save_targets=True,
    force_recompute=False,
    splits="val",
    dataset=dataset,
)

embeddings_test, _, _, targets_test = get_all_embeddings_downstream(
    num_batches=None,
    loader=test_loader,
    model_name=model_name,
    model=model,
    save_targets=True,
    force_recompute=False,
    splits="test",
    dataset=dataset,
)

idx_target = args.idx_target
targets_tr = targets_tr[:, idx_target].unsqueeze(1)
targets_val = targets_val[:, idx_target].unsqueeze(1)
targets_test = targets_test[:, idx_target].unsqueeze(1)

eps = 1e-8
mu_tr = targets_tr.mean(dim=0)
std_tr = targets_tr.std(dim=0) + eps
targets_tr = ((targets_tr - mu_tr) / std_tr)
targets_val = ((targets_val - mu_tr) / std_tr)
targets_test = ((targets_test - mu_tr) / std_tr)

mu_emb = embeddings_tr.mean(dim=0)
std_emb = embeddings_tr.std(dim=0) + eps
embeddings_tr = (embeddings_tr - mu_emb) / std_emb
embeddings_val = (embeddings_val - mu_emb) / std_emb
embeddings_test = (embeddings_test - mu_emb) / std_emb

print("Statistics:")
print(f"Training embeddings shape: {embeddings_tr.shape}, Targets shape: {targets_tr.shape}")
print(f"Validate embeddings shape: {embeddings_val.shape}, Targets shape: {targets_val.shape}")
print(f"Testing  embeddings shape: {embeddings_test.shape}, Targets shape: {targets_test.shape}")

print(f"Target mean before normalizing (train): {mu_tr.item():.4f}, Target std (train): {std_tr.item():.4f}")
print(f"Target mean after normalizing (train): {targets_tr.mean().item():.4f}, Target std (train): {targets_tr.std().item():.4f}")
print(f"Target mean after normalizing (val): {targets_val.mean().item():.4f}, Target std (val): {targets_val.std().item():.4f}")
print(f"Target mean after normalizing (test): {targets_test.mean().item():.4f}, Target std (test): {targets_test.std().item():.4f}")

print(f"Embedding mean before normalizing (train): {mu_emb.mean().item():.4f}, Embedding std (train): {std_emb.mean().item():.4f}")
print(f"Embedding mean after normalizing (train): {embeddings_tr.mean().item():.4f}, Embedding std (train): {embeddings_tr.std().item():.4f}")
print(f"Embedding mean after normalizing (val): {embeddings_val.mean().item():.4f}, Embedding std (val): {embeddings_val.std().item():.4f}")
print(f"Embedding mean after normalizing (test): {embeddings_test.mean().item():.4f}, Embedding std (test): {embeddings_test.std().item():.4f}")

dataset_tr = TensorDataset(embeddings_tr, targets_tr)
dataset_val = TensorDataset(embeddings_val, targets_val)
dataset_test = TensorDataset(embeddings_test, targets_test)

train_loader_reg = DataLoader(dataset_tr, batch_size=256, shuffle=True)
val_loader_reg = DataLoader(dataset_val, batch_size=256, shuffle=False)
test_loader_reg = DataLoader(dataset_test, batch_size=256, shuffle=False)
# %%

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_lays=2, dropout = 0):
        super().__init__()
        if num_lays == 1:
            self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim))
        else:
            self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                                    nn.Linear(hidden_dim, out_dim)
                                )

    def forward(self, x):
        return self.mlp(x)

#training loop
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.save = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save = True
        else:
            self.save = False
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

#%%
def _collect_predictions(model, loader):
    device = next(model.parameters()).device
    model.eval()
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds_all.append(preds.detach().cpu())
            targets_all.append(y.detach().cpu())
    return torch.cat(preds_all, dim=0), torch.cat(targets_all, dim=0)


def _pearson_r(preds, targets):
    preds_m = preds - torch.mean(preds)
    targets_m = targets - torch.mean(targets)
    num = torch.sum(preds_m * targets_m)
    denom = torch.sqrt(torch.sum(preds_m**2) * torch.sum(targets_m**2)).clamp_min(1e-8)
    return num / denom


def evaluate_regression(model, loader):
    """
    Compute the MSE, RMSE and Pearson R of `model` over all batches in `loader`.
    """
    preds, targets = _collect_predictions(model, loader)
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    mse = torch.mean((preds - targets) ** 2)
    rmse = torch.sqrt(mse)
    pearson_r = _pearson_r(preds, targets)

    return mse.item(), rmse.item(), pearson_r.item()


def evaluate_mse(model, loader):
    """
    Compute the mean squared error of `model` over all batches in `loader`.
    """
    return evaluate_regression(model, loader)[0]

def train_one_batch(model, batch, criterion, optimizer):
    x_batch, y_batch = batch
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    return loss

def objective(trial):
    # --- Hyperparameters to search ---
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
    dropout = trial.suggest_float("dropout", 0.0, 0.9)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    num_lays = 2
    # --- Build model ---
    model = MLP(mu_emb.shape[-1], hidden_dim, 1, num_lays, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold=1e-3)
    print(f"Trial {trial.number}: hidden_dim={hidden_dim}, dropout={dropout:.2f}, lr={lr:.2e}, weight_decay={weight_decay:.2e}")
    # --- Short training loop ---
    max_epochs = 1000
    early_stopper = EarlyStopping(patience=5, min_delta=1e-3)
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader_reg:
            loss = train_one_batch(model, batch, criterion, optimizer)

        # Always compute val_loss for Optuna
        val_loss = evaluate_mse(model, val_loader_reg)

        # Only apply scheduler + early stopping every 5 epochs
        if epoch % 5 == 0:
            scheduler.step(val_loss)
            early_stopper(val_loss)
            if early_stopper.save:
                torch.save(model.state_dict(), f"best_model_trial_{trial.number}_{model_name}_{idx_target}.pt")
            if early_stopper.early_stop:
                break

        # Optuna needs a metric every epoch
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    train_loss = evaluate_mse(model, train_loader_reg)
    print(f"Trial {trial.number} finished with val_loss={val_loss:.4f} and train_loss={train_loss:.4f}")
    return val_loss


# --- Run the study ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("Best trial:")
print(study.best_trial.params)

# %%
# --- Final evaluation on test set ---
best_trial = study.best_trial
best_model = MLP(mu_emb.shape[-1], best_trial.params["hidden_dim"], 1, num_lays=2, dropout=best_trial.params["dropout"]).to(device)
best_model.load_state_dict(torch.load(f"best_model_trial_{best_trial.number}_{model_name}_{idx_target}.pt"))

# 4. Evaluate on test set
if not dataset.startswith("BayesianNet"):
    test_mse = evaluate_mse(best_model, test_loader_reg)
    print(f"\nFinal Test MSE for model {model_name} and target {idx_target}: {test_mse:.4f}")

else:
    test_mse, test_rmse, test_pearson_r = evaluate_regression(best_model, test_loader_reg)
    print(
        f"\nFinal Test metrics for model {model_name} and target {idx_target}: "
        f"MSE={test_mse:.4f}, "
        f"RMSE={test_rmse:.4f}, "
        f"Pearson r={test_pearson_r:.4f}"
    )
