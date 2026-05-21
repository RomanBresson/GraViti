# This code was obtained from the official Catflow submission: https://openreview.net/forum?id=UahrHR5HQh&noteId=BoBuVw1Bmx

import torch
import torch.nn as nn
from torch_geometric.nn.pool import global_add_pool, global_max_pool, global_mean_pool

class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        """ X: bs, n, dx. """
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.cat((m, mi, ma, std))
        out = self.lin(z)
        return out

class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.cat((m, mi, ma, std))
        out = self.lin(z)
        return out

class Xtoy_masked(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, mask):
        """ X: bs, n, dx. """
        bs,n,dx = X.shape
        X_flat = X.reshape(-1, dx)
        mask_flat = mask.reshape(-1).bool()
        batch_indices = torch.arange(bs).unsqueeze(1).expand(-1, n).reshape(-1).to(X.device)
        if mask_flat.sum() == 0:
            # no valid items at all -> return zeros
            return self.lin(torch.zeros(bs, 4 * dx, device=X.device, dtype=X.dtype))

        # Step 3: Select unmasked entries
        valid_features = X_flat[mask_flat]        # (k, f)
        valid_batch = batch_indices[mask_flat]           # (k,)
        m = global_mean_pool(valid_features, valid_batch)
        ma = global_max_pool(valid_features, valid_batch)
        mi = -global_max_pool(-valid_features, valid_batch)
        std_int = valid_features-m[valid_batch]
        std_int = std_int**2
        std = global_mean_pool(std_int, valid_batch)
        z = torch.cat((m, mi, ma, std), dim=-1)
        out = self.lin(z)
        return out

class Etoy_masked(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E, mask):
        """ E: (b, n, n, f), mask: (b, n) """
        b, n, _, f = E.shape

        # Step 1: Create joint mask for valid (i, j) pairs
        mask_bool = mask.bool()  # (b, n)
        mask_row = mask_bool.unsqueeze(2)         # (b, n, 1)
        mask_col = mask_bool.unsqueeze(1)         # (b, 1, n)
        joint_mask = mask_row & mask_col     # (b, n, n) — True where both i and j are unmasked

        # Step 2: Flatten E and joint_mask
        E_flat = E.reshape(b * n * n, f)        # (b*n*n, f)
        joint_mask_flat = joint_mask.reshape(-1)  # (b*n*n,)

        # Step 3: Compute batch indices
        batch_indices = torch.arange(b).reshape(b, 1, 1).expand(-1, n, n).reshape(-1).to(E.device)  # (b*n*n,)
        if joint_mask_flat.sum() == 0:
            return self.lin(torch.zeros(b, 4 * f, device=E.device, dtype=E.dtype))

        # Step 4: Select valid entries
        valid_E = E_flat[joint_mask_flat]           # (k, f)
        valid_batch = batch_indices[joint_mask_flat]  # (k,)

        # Step 5: Pooling
        m = global_mean_pool(valid_E, valid_batch)
        ma = global_max_pool(valid_E, valid_batch)
        mi = -global_max_pool(-valid_E, valid_batch)

        # Step 6: Standard deviation
        std_int = (valid_E - m[valid_batch]) ** 2
        std = global_mean_pool(std_int, valid_batch)

        # Step 7: Concatenate and project
        z = torch.cat((m, mi, ma, std), dim=-1)
        out = self.lin(z)
        return out

def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)