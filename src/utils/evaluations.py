# %%
import random
import torch
from rdkit import Chem
from src.data.pubchem import _process_molecule_worker
from src.data.constants import get_valid_atoms, get_valid_bonds
from src.data.processing import PreTransformPubChem, TransformGraph
from src.data.treat_data import make_loader
from torch_geometric.data import InMemoryDataset, Data
from rdkit.Chem import Draw, Descriptors, MolToSmiles
import cma
import os

from src.model.checkpoint import ModelCheckpoint
from src.model.builder import ModelBuilder
from src.data.treat_data import treat_batch,treat_qm9_as_pc
from sklearn.metrics import confusion_matrix
from src.utils.misc import mask_E
from src.utils.chem import make_molecules_from_outputs, make_molecules_from_batch

from src.utils.coloring import (
    make_nx_graphs_from_outputs as make_coloring_graphs_from_outputs,
    plot_nx_graphs_grid as plot_coloring_graphs_grid,
)
from src.utils.bayesian_net import (
    make_nx_graphs_from_outputs as make_bayesian_net_graphs_from_outputs,
    make_nx_graphs_from_batch as make_bayesian_net_graphs_from_batch,
    plot_nx_graphs_grid as plot_bayesian_net_graphs_grid,
)
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt

def get_non_node_index(dataset_name):
    NO_NODE_INDEX = {
        'QM9NoHydro': 4,
        'QM9WithHydro': 5,
        'ZINC250k': 9,
        'PubChem16': 31,
        'PubChem16S': 31,
        'PubChem32': 31,
        'PubChem32S': 31,}[dataset_name]
    return NO_NODE_INDEX
# %%
def confusion_matrix_nodes(batch, pred_X, node_mask=None):
    """
    Compute confusion matrix for node existence prediction.
    
    Args:
        batch: Original batch with ground truth data.
        pred_X: Predicted node features (logits or probabilities).
        node_mask (torch.Tensor, optional): Boolean mask of shape (B, N), where True indicates valid node.
    """
    if node_mask is None:
        real_nodes = batch['node_mask'].flatten()
    else:
        real_nodes = node_mask.flatten()
    pred_nodes = pred_X.argmax(-1).flatten()[real_nodes]
    gt_nodes = batch['X'].flatten()[real_nodes]
    cm = confusion_matrix(gt_nodes.cpu(), pred_nodes.cpu(), labels=list(range(pred_X.size(-1))))
    return cm

def confusion_matrix_edges(batch, pred_E, node_mask=None):
    """
    Compute confusion matrix for edge existence prediction.
    
    Args:
        batch: Original batch with ground truth data.
        pred_E: Predicted edge features (logits or probabilities).
        node_mask (torch.Tensor, optional): Boolean mask of shape (B, N), where True indicates valid node.
    """
    edge_mask = mask_E(node_mask if node_mask is not None else batch['node_mask'])
    real_edges = edge_mask.flatten()
    pred_edges = pred_E.argmax(-1).flatten()[real_edges]
    gt_edges = batch['E'].flatten()[real_edges]
    cm = confusion_matrix(gt_edges.cpu(), pred_edges.cpu(), labels=list(range(pred_E.size(-1))))
    return cm

def confusion_matrices_loader(model, data_loader):
    model.eval()
    total_cm_nodes_with_gt_mask = None
    total_cm_edges_with_gt_mask = None
    total_cm_nodes_with_pred_mask = None
    total_cm_edges_with_pred_mask = None
    with torch.no_grad():
        for batch in data_loader:
            b = treat_batch(batch, batch.x.device)
            recon_batch = model(b)
            pred_X, pred_E, _, _, _ = recon_batch
            cm_nodes_with_gt_mask = confusion_matrix_nodes(b, pred_X, node_mask=b['node_mask'])
            cm_edges_with_gt_mask = confusion_matrix_edges(b, pred_E, node_mask=b['node_mask'])
            cm_nodes_with_pred_mask = confusion_matrix_nodes(b, pred_X, node_mask=b['node_mask'])
            cm_edges_with_pred_mask = confusion_matrix_edges(b, pred_E, node_mask=b['node_mask'])
            if total_cm_nodes_with_gt_mask is None:
                total_cm_nodes_with_gt_mask = cm_nodes_with_gt_mask
                total_cm_edges_with_gt_mask = cm_edges_with_gt_mask
                total_cm_nodes_with_pred_mask = cm_nodes_with_pred_mask
                total_cm_edges_with_pred_mask = cm_edges_with_pred_mask
            else:
                total_cm_nodes_with_gt_mask += cm_nodes_with_gt_mask
                total_cm_edges_with_gt_mask += cm_edges_with_gt_mask
                total_cm_nodes_with_pred_mask += cm_nodes_with_pred_mask
                total_cm_edges_with_pred_mask += cm_edges_with_pred_mask
    return total_cm_nodes_with_gt_mask, total_cm_edges_with_gt_mask, total_cm_nodes_with_pred_mask, total_cm_edges_with_pred_mask

def accuracy_from_confusion_matrix(cm):
    correct = np.trace(cm)
    total = np.sum(cm)
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def load_embeddings_and_attributes(model_name, splits):
    """Load embeddings and attributes if they already exist."""
    base = os.path.join("results", model_name)

    def load_tensor(name):
        path = os.path.join(base, name)
        return torch.load(path) if os.path.exists(path) else None

    embeddings = load_tensor(f"embeddings_{splits}.pt")
    mws = load_tensor(f"mw_{splits}.pt")
    logps = load_tensor(f"logp_{splits}.pt")
    targets = load_tensor(f"targets_{splits}.pt")

    # If embeddings exist, assume the rest are consistent
    if embeddings is not None:
        return embeddings, mws, logps, targets
    return None, None, None, None


def get_all_embeddings_downstream(
    num_batches, loader, model, model_name, save_targets=False, force_recompute=False, splits="any", dataset=None
):
    # Try loading first
    loaded = load_embeddings_and_attributes(model_name, splits)
    if (loaded[0] is not None and (not force_recompute) and ((not save_targets) or loaded[3] is not None)):
        return loaded  # Already computed
    # Otherwise compute
    device = next(iter(model.parameters())).device
    it = iter(loader)
    emb_list, mw_list, logp_list = [], [], []
    tgt_list = [] if save_targets else None
    if num_batches is None:
        num_batches = len(loader)
    print(f"Computing embeddings for {num_batches} batches...")
    for i in range(num_batches):
        print(f"Processing batch {i+1}/{num_batches}")
        try:
            batch = next(it)
        except StopIteration:
            break
        if dataset is None or dataset.startswith("QM9"):
            batch = treat_qm9_as_pc(batch, device)
        else:
            batch = treat_batch(batch, device)
        with torch.no_grad():
            emb = model.encode(batch)[0].cpu()
        emb_list.append(emb)
        chem = batch.get("chem_properties")
        if chem is not None:
            mw_list.append(chem[:, 0].cpu() if chem is not None else None)
            logp_list.append(chem[:, 1].cpu() if chem is not None else None)

        if save_targets:
            tgt = batch.get("reg_target")
            tgt_list.append(tgt.cpu() if tgt is not None else None)

    targets = None
    if save_targets:
        targets = (
            torch.cat([x for x in tgt_list], dim=0)
        )
    # Concatenate
    embeddings = torch.cat(emb_list, dim=0)
    mws = torch.cat(mw_list, dim=0) if len(mw_list) > 0 else None
    logps = torch.cat(logp_list, dim=0) if len(logp_list) > 0 else None

    # Save
    save_dir = os.path.join("results", model_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(embeddings, os.path.join(save_dir, f"embeddings_{splits}.pt"))
    if mws is not None:
        torch.save(mws, os.path.join(save_dir, f"mw_{splits}.pt"))
    if logps is not None:
        torch.save(logps, os.path.join(save_dir, f"logp_{splits}.pt"))
    if save_targets:
        torch.save(targets, os.path.join(save_dir, f"targets_{splits}.pt"))
    return embeddings, mws, logps, targets


def plot_embeddings(embeddings, value_to_color, color_labels=None):
    # Compute 2D UMAP projection
    reducer = UMAP(n_components=2, n_neighbors=10)
    embedding_2d = reducer.fit_transform(embeddings)
    figs = []
    num_plots = len(value_to_color)

    # Default labels if none provided
    if color_labels is None:
        color_labels = [f"Feature {i}" for i in range(num_plots)]

    for fi, label in zip(value_to_color, color_labels):
        fig, ax = plt.subplots(figsize=(10, 10))

        scatter = ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=fi,
            cmap='viridis',
            s=5
        )

        fig.colorbar(scatter, ax=ax, label=label)
        ax.set_title(f'2D UMAP Projection Colored by {label}')
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')

        figs.append(fig)

    return figs


#%%
def decode_with_size(model, codes, num_nodes):
    model.eval()
    batch_size = codes.size(0)
    node_filter = torch.ones((batch_size, num_nodes), device=next(iter(model.parameters())).device).bool()
    with torch.no_grad():
        generated_graphs = model.decoder(codes, override_size=num_nodes, node_filter=node_filter)
    return generated_graphs

def truncate_graphs(X, E, num_nodes):
    x = X[:num_nodes]
    e = E[:num_nodes, :num_nodes]
    return x,e

def sample_latent(model, num_graphs=100):
    model.eval()
    latent_dim = model.latent_dim
    codes = torch.randn((num_graphs, latent_dim), device=next(iter(model.parameters())).device)
    return codes

def sample_graphs(model, num_graphs=100, forced_size=None):
    device = next(model.parameters()).device

    codes = sample_latent(model, num_graphs=num_graphs).to(device)
    if forced_size is not None:
        forced_size = forced_size * torch.ones(num_graphs, device=device).int()

    with torch.no_grad():
        outputs = model.decoder(codes, force_size_graph=forced_size)
    return outputs

def molecules_from_outputs(outputs, dataset_name):
    # Coloring dataset - Create NetworkX graphs
    if dataset_name.startswith("Coloring"):
        return make_coloring_graphs_from_outputs(
            batch=outputs,
            dataset_name=dataset_name,
        )

    # Bayesian-network dataset - Create NetworkX DiGraphs
    if dataset_name.startswith("BayesianNet"):
        return make_bayesian_net_graphs_from_outputs(
            batch=outputs,
            dataset_name=dataset_name,
        )
    
    # Molecular datasets - Create RDKit Molecules
    with_aromatic = (outputs['E'].shape[-1]==5)
    return make_molecules_from_outputs(
        batch=outputs,
        dataset_name=dataset_name,
        with_aromatic=with_aromatic,
    )

def validity(molecules, dataset_name):
    valid_count = 0
    valid_molecules = []
    is_valid_fn = lambda data: (
        data.graph.get("valid", False)
        if dataset_name.startswith(("Coloring", "BayesianNet"))
        else data is not None
    )
    for mol in molecules:
        if is_valid_fn(mol):
            valid_count += 1
            valid_molecules.append(mol)
    validity_ratio = valid_count / len(molecules) if len(molecules) > 0 else 0.0
    return validity_ratio, valid_molecules

def uniqueness(generated_smiles):
    unique_mols = set(generated_smiles)
    uniqueness_ratio = len(unique_mols) / len(generated_smiles) if len(generated_smiles) > 0 else 0.0
    return uniqueness_ratio, unique_mols

def novelty(generated_smiles, training_smiles):
    training_smiles_set = set(training_smiles)
    generated_smiles_set = set(generated_smiles)
    all_smiles = training_smiles_set.union(generated_smiles_set)
    number_novel = len(all_smiles)-len(training_smiles_set)
    novelty_ratio = number_novel/len(generated_smiles_set) if len(generated_smiles_set) > 0 else 0.0
    return novelty_ratio

def evaluate_generation(model,
                        training_smiles,
                        dataset_name,
                        num_graphs=1024,
                        forced_size=None,
                        batch_size=64):
    # forced_size is an int or None
    num_batches = num_graphs//batch_size
    generated_molecules = []
    for _ in range(num_batches):
        outputs = sample_graphs(model, num_graphs=batch_size, forced_size=forced_size)
        generated_molecules += molecules_from_outputs(outputs, dataset_name=dataset_name)
    validity_ratio, valid_molecules = validity(generated_molecules, dataset_name)
    generated_smiles = (
        [G.graph.get("id", "") for G in valid_molecules]
        if dataset_name.startswith(("Coloring", "BayesianNet"))
        else [Chem.MolToSmiles(mol) for mol in valid_molecules]
    )
    uniqueness_ratio, unique_smiles = uniqueness(generated_smiles)
    novelty_ratio = novelty(unique_smiles, training_smiles) 
    return validity_ratio, uniqueness_ratio, novelty_ratio, generated_molecules, unique_smiles
# %%
def get_smiles_loader(loader):
    smiles = []
    for batch in loader:
        treated_batch = treat_batch(batch, batch.x.device)
        mols = make_molecules_from_batch(treated_batch)
        smiles += [Chem.MolToSmiles(mol) for mol in mols]
    return smiles

def get_first_embeddings(num_batches, loader, model):
    l = iter(loader)
    device = next(iter(model.parameters())).device
    first_embeddings = []
    if num_batches is None:
        num_batches = len(loader)
    for b in l:
        if len(first_embeddings)>=num_batches:
            break
        print(len(first_embeddings))
        with torch.no_grad():
            try:
                bat = treat_batch(b, device)
                first_embeddings.append(model.encode(bat)[0])
            except:
                continue
    return first_embeddings

def interpolate(mol1_emb, mol2_emb, model, dataset, steps=100, interpolation_mode='rot'):
    """
    Batched interpolation between two molecule embeddings.

    - 'rot': SLERP on direction + linear interpolation on norm
    - 'lin': standard linear interpolation

    Decodes all interpolated embeddings in one forward pass.
    Keeps only molecules that change (based on canonical SMILES).
    """
    device = mol1_emb.device
    t_values = torch.linspace(0, 1, steps, device=device)

    if interpolation_mode == 'rot':
        # --- Decompose into norm and direction ---
        norm1 = torch.norm(mol1_emb)
        norm2 = torch.norm(mol2_emb)

        # Avoid division by zero
        eps = 1e-8
        v1 = mol1_emb / (norm1 + eps)
        v2 = mol2_emb / (norm2 + eps)

        # --- Angle between directions ---
        cos_theta = torch.clamp(torch.dot(v1, v2), -1.0, 1.0)
        theta = torch.acos(cos_theta)

        # --- Interpolate norms linearly ---
        interp_norms = (1 - t_values) * norm1 + t_values * norm2
        interp_norms = interp_norms[:, None]  # (steps, 1)

        if torch.isclose(theta, torch.tensor(0.0, device=device)):
            # Directions are (almost) identical → linear direction interpolation
            directions = (
                (1 - t_values[:, None]) * v1[None, :]
                + t_values[:, None] * v2[None, :]
            )
        else:
            sin_theta = torch.sin(theta)
            t = t_values[:, None]

            directions = (
                torch.sin((1 - t) * theta) / sin_theta * v1[None, :]
                + torch.sin(t * theta) / sin_theta * v2[None, :]
            )

        # --- Recompose: direction × interpolated norm ---
        emb_batch = directions * interp_norms

    elif interpolation_mode == 'lin':
        emb_batch = (
            (1 - t_values[:, None]) * mol1_emb[None, :]
            + t_values[:, None] * mol2_emb[None, :]
        )

    else:
        raise ValueError(f"Unknown interpolation_mode: {interpolation_mode}")

    # --- Decode ---
    with torch.no_grad():
        out_batch = model.decode(emb_batch)

    mols = molecules_from_outputs(out_batch, dataset)

    molecules = []
    times = []
    prev_smiles = None

    for t, mol in zip(t_values, mols):
        if dataset.startswith(("Coloring", "BayesianNet")):
            if not mol.graph.get("valid", False):
                continue
            smi = mol.graph.get("id", "")
        else:
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol)

        if smi != prev_smiles:
            molecules.append(mol)
            times.append(float(t))
            prev_smiles = smi

    return molecules, times


# %%
def plot_interpolation(molecules, times, dataset, mols_per_row=5, figsize=(12, 6)):
    """
    Plot the unique molecules obtained along the interpolation path.
    """
    if len(molecules) == 0:
        raise ValueError("No molecules/graphs to plot.")

    # Convert times to labels
    labels = None
    #labels = [f"t={t:.2f}" for t in times]
    if dataset is not None and dataset.startswith("Coloring"):
        return plot_coloring_graphs_grid(
            molecules,
            legends=labels,
            graphs_per_row=mols_per_row,
            sub_img_size=(250, 250),
            with_labels=True,
        )

    if dataset is not None and dataset.startswith("BayesianNet"):
        return plot_bayesian_net_graphs_grid(
            molecules,
            legends=labels,
            graphs_per_row=mols_per_row,
            sub_img_size=(250, 250),
            with_labels=True,
        )

    # RDKit grid image
    return Chem.Draw.MolsToGridImage(
        molecules,
        molsPerRow=mols_per_row,
        subImgSize=(250, 250),
        legends=labels,
        useSVG=False,
        returnPNG=False
    )

def plot_logp_evolution(molecules, scores, mols_per_row=5, figsize=(14, 8)):
    """
    Keep only unique molecules (by canonical SMILES) and return a PIL grid image.
    """
    unique_mols = []
    unique_scores = []
    seen = set()
    for mol, s in zip(molecules, scores):
        if mol is None:
            continue
        smi = MolToSmiles(mol)
        if smi not in seen:
            seen.add(smi)
            unique_mols.append(mol)
            unique_scores.append(s)
    if len(unique_mols) == 0:
        raise ValueError("No valid molecules to plot.")
    labels = [f"logP={s:.2f}" for s in unique_scores]
    opts = Draw.MolDrawOptions()
    opts.legendFontSize = 30
    opts.legendFraction = 0.20
    pil_img = Draw.MolsToGridImage(
        unique_mols,
        molsPerRow=mols_per_row,
        subImgSize=(200, 200),
        legends=labels,
        useSVG=False,
        returnPNG=False,
        drawOptions=opts
    )
    return pil_img


def is_valid_emb(emb, model, dataset):
    """Check whether an embedding decodes to a valid molecule/coloring graph."""
    with torch.no_grad():
        out = model.decode(emb.unsqueeze(0))
    decoded = molecules_from_outputs(out, dataset)
    if len(decoded) == 0:
        return False
    mol = decoded[0]
    if dataset.startswith(("Coloring", "BayesianNet")):
        return mol.graph.get("valid", False)
    return mol is not None


def interpolate_random_pairs(S, model, dataset, num_pairs=10, steps=100, interpolation_mode='lin', seed=42):
    """
    Selects random valid pairs of molecule embeddings from S,
    runs interpolation on each pair, and returns the results.
    """
    results = []
    # Flatten S into a list of embeddings
    all_embs = [emb[i] for emb in S for i in range(emb.shape[0])]
    # Pre-filter valid embeddings
    valid_embs = []
    for emb in all_embs:
        if is_valid_emb(emb, model, dataset):
            valid_embs.append(emb)
    print(f"Found {len(valid_embs)} valid embeddings.")
    if len(valid_embs) < 2:
        print("Not enough valid molecules to interpolate.")
        return results
    # Sample random pairs
    random.seed(seed)
    for k in range(num_pairs):
        mol1_emb, mol2_emb = random.sample(valid_embs, 2)
        print(f"Interpolating pair {k+1}/{num_pairs}...")
        mols, ts = interpolate(
            mol1_emb, mol2_emb,
            model=model,
            dataset=dataset,
            steps=steps,
            interpolation_mode=interpolation_mode
        )
        results.append({
            "pair_index": k,
            "mol1_emb": mol1_emb,
            "mol2_emb": mol2_emb,
            "molecules": mols,
            "times": ts
        })
    return results
# %%
def interpolation_wrapper(first_batches, model, model_name, dataset, num_pairs=10, steps=100, interp_mode='lin'):    
    output_dir = os.path.join("results", model_name)
    os.makedirs(output_dir, exist_ok=True)
    results = interpolate_random_pairs(first_batches, model, dataset, num_pairs, steps, interpolation_mode=interp_mode)
    for i,res in enumerate(results):
        if len(res["molecules"]) == 0:
            continue
        img = plot_interpolation(molecules=res['molecules'], times=res['times'], dataset=dataset)
        img.save(os.path.join(output_dir, f"interp_{i}_{interp_mode}.pdf"))
# %%
def optimize_logp_cma(
    z0,
    model,
    dataset,
    sigma0=0.5,
    n_iter=30,
    popsize=32,
):
    """
    Simple CMA-ES optimization in latent space to maximize logP.

    Args:
        z0: initial latent vector (torch.Tensor, shape [D])
        model: generative model with .decode(latents)
        dataset: dataset object for molecules_from_outputs
        sigma0: initial CMA-ES step size
        n_iter: number of CMA-ES iterations
        popsize: population size per iteration

    Returns:
        best_mols: list of best molecules over time
        best_scores: list of best logP scores
        zs: list of latent vectors for best solutions
    """
    device = z0.device
    D = z0.shape[-1]
    # --- Decode initial molecule ---
    with torch.no_grad():
        out0 = model.decode(z0.unsqueeze(0))
    mol0 = molecules_from_outputs(out0, dataset)[0]
    init_logp = Descriptors.MolLogP(mol0) if mol0 is not None else -1e6
    # --- CMA-ES setup ---
    x0 = z0.detach().cpu().numpy()
    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "popsize": popsize,
            "verb_disp": 0,
            "verb_log": 0,
        },
    )
    best_mols = [mol0]
    best_scores = [init_logp]
    zs = [z0.clone()]
    global_best_score = init_logp
    global_best_mol = mol0
    global_best_z = z0.clone()
    for ite in range(n_iter):
        # --- Ask CMA-ES for candidate latent vectors ---
        xs = es.ask()  # list of np arrays, shape [popsize, D]
        z_batch = torch.tensor(xs, dtype=torch.float32, device=device)
        # --- Decode all candidates ---
        with torch.no_grad():
            out = model.decode(z_batch)
        mols = molecules_from_outputs(out, dataset)
        # --- Compute logP scores ---
        scores = []
        for mol in mols:
            if mol is None:
                scores.append(-1e6)
            else:
                try:
                    scores.append(Descriptors.MolLogP(mol))
                except Exception:
                    scores.append(-1e6)
        scores = np.array(scores, dtype=float)
        # CMA-ES minimizes → pass negative objective
        es.tell(xs, (-scores).tolist())
        # --- Track best candidate ---
        idx = int(np.argmax(scores))
        iter_best_score = scores[idx]
        iter_best_mol = mols[idx]
        iter_best_z = z_batch[idx]
        if iter_best_score > global_best_score:
            global_best_score = iter_best_score
            global_best_mol = iter_best_mol
            global_best_z = iter_best_z.clone()
        best_mols.append(global_best_mol)
        best_scores.append(global_best_score)
        zs.append(global_best_z.clone())

        if es.stop():
            break

    return best_mols, best_scores, zs

# %%
def logp_optim_wrapper_cma(first_embeddings, model, model_name, dataset, with_aromatic=True):
    output_dir = os.path.join("results", model_name)
    os.makedirs(output_dir, exist_ok=True)
    i = 0
    for batch_emb in first_embeddings:
        # --- Decode batch molecules ---
        with torch.no_grad():
            mols = make_molecules_from_outputs(model.decode(batch_emb), with_aromatic=with_aromatic, dataset_name=dataset)  # list of RDKit Mol or None

        valid = []
        logps = []

        # --- Compute logP for valid molecules ---
        for idx, mol in enumerate(mols):
            if mol is None:
                continue
            try:
                lp = Descriptors.MolLogP(mol)
            except Exception:
                continue
            valid.append(idx)
            logps.append(lp)

        if len(valid) == 0:
            continue  # skip batch with no valid molecules

        # --- Choose two starting points: lowest logP and random valid ---
        min_idx = valid[np.argmin(logps)]
        rand_idx = random.choice(valid)

        for chosen_idx in [min_idx, rand_idx]:
            try:
                best_mols, best_scores, zs = optimize_logp_cma(
                    z0=batch_emb[chosen_idx],
                    model=model,
                    dataset=dataset,
                    n_iter=100,
                    popsize=64,
                    sigma0=0.1,
                )

                i += 1
                img = plot_logp_evolution(best_mols, best_scores)
                img.save(f"results/{model_name}/mols_logp_{i}.pdf")

            except Exception as e:
                continue

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_generation_trials(
    model,
    training_smiles,
    dataset,
    num_graphs=1024,
    batch_size=64,
    num_trials=10,
    base_seed=42,
    forced_size=None,
):
    metrics = []

    for t in range(num_trials):
        seed = base_seed + t
        set_seed(seed)

        validity, uniqueness, novelty, gen_mols, gen_smiles = evaluate_generation(
            model=model,
            training_smiles=training_smiles,
            dataset_name=dataset,
            num_graphs=num_graphs,
            batch_size=batch_size,
            forced_size=forced_size
        )

        metrics.append({
            "seed": seed,
            "validity": validity,
            "uniqueness": uniqueness,
            "novelty": novelty,
            "count": len(gen_mols)
        })

    return metrics

class MoleculeDataset(InMemoryDataset):
    def __init__(self, data_list, transform=None):
        super().__init__('.', transform)
        self.data, self.slices = self.collate(data_list)

def make_molecule_from_smiles(smiles, with_aromatic=True, drop_fragments=True):
    molecules = []
    for smile in smiles:
        mol = _process_molecule_worker(
            smile,
            get_valid_atoms("PubChem16"),
            get_valid_bonds(with_aromatic),
            drop_fragments=drop_fragments,
            pre_transform=PreTransformPubChem(with_aromatic)
        )
        molecules.append(mol[1])

    dataset = MoleculeDataset(
        molecules,
        transform=TransformGraph(dataset='PubChem16', with_aromatic=with_aromatic)
    )
    return dataset

# %%
def interpolation_forced(smiles_start, smiles_end, model, dataset, steps=100):
    d1 = make_loader(make_molecule_from_smiles(smiles_start), batch_size=64)
    d2 = make_loader(make_molecule_from_smiles(smiles_end), batch_size=64)
    batch1 = next(iter(d1))
    batch2 = next(iter(d2))
    device = next(model.parameters()).device
    batch1 = treat_batch(batch1, device)
    batch2 = treat_batch(batch2, device)
    all_mols = []
    all_times = []
    with torch.no_grad():
        emb1 = model.encode(batch1)[0]
        emb2 = model.encode(batch2)[0]
        for e1, e2 in zip(emb1, emb2):
            mols, times = interpolate(e1, e2, model, dataset, steps=steps)
            all_mols.append(mols)
            all_times.append(times)
    return all_mols, all_times

def permute_nodes(perm, node_level_info):
    B, n = perm.shape
    batch_idx = torch.arange(B, device=perm.device).unsqueeze(1)
    return node_level_info[batch_idx, perm]

def permute_edges(perm, edge_labels):
    B, N = perm.shape
    batch_idx = torch.arange(B, device=perm.device).view(-1, 1)
    out = edge_labels[batch_idx, perm]
    out = out.transpose(1, 2)[batch_idx, perm].transpose(1, 2)    
    return out

def unpermute_graph(outputs):
    """
    Apply the predicted permutation matrices to reorder nodes and edges.

    Expected:
        outputs['predicted_perm_matrix']: (B, n, n) permutation matrices
    """

    # Convert permutation matrix -> permutation list
    perm = outputs['predicted_perm_matrix'].argmax(-1)

    # Safety check
    assert perm.dim() == 2, (
        f"Permutation must be (B, n, n) -> argmax -> (B, n). "
        f"Got shape {tuple(perm.shape)} instead."
    )

    new_outputs = dict(outputs)

    # Apply permutation to nodes
    new_outputs['X'] = permute_nodes(perm, outputs['X'])

    # Apply permutation to edges
    new_outputs['E'] = permute_edges(perm, outputs['E'])

    # Optional node-level predictions
    for key in ['used_node_filter', 'predicted_hydrogens', 'predicted_formal_charges']:
        if outputs.get(key) is not None:
            new_outputs[key] = permute_nodes(perm, outputs[key])

    return new_outputs
