# This code is adapted from the official GRALE repository:
# https://github.com/KrzakalaPaul/GRALE
# Please cite the original GRALE paper if you use this code:

#@inproceedings{grale2023,
#  title={GRALE: Graph Representation Learning with Explicit Link Encoding},
#  author={Paul Krzakala and Florent Krzakala and others},
#  booktitle={Proceedings of the 40th International Conference on Machine Learning},
#  year={2023}
#}

import torch
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
import torch
import numpy as np
from src.model.grale_losses.grale_losses import pairwise_L1_norm

def permutations_list_to_matrices(permutations_list, device = 'cuda'):
    B,Mmax = len(permutations_list),len(permutations_list[0])
    permutations_matrices = torch.stack([torch.eye(Mmax, device=device)[:,permutation] for permutation in permutations_list]) 
    return permutations_matrices                                    

def permutations_matrices_to_list(permutations_matrices):
    permutations_list = permutations_matrices.argmax(dim=1)
    permutations_list = permutations_list.cpu().detach().numpy()
    permutations_list = [permutation for permutation in permutations_list]
    # Check that it is a permutation
    for i,permutation in enumerate(permutations_list):
        assert len(permutation) == len(set(permutation)), f'Permutation matrix cannot be converted to list, try to approximate with Hungarian'
    return permutations_list

######################### HUNGARIAN #########################
def hungarian_solver(cost):
    _, permutation = linear_sum_assignment(cost.T)
    return permutation

def batched_hungarian_solver(cost, use_joblib = False):
    '''
    Compute optimal permutation matrices for each batch element
    '''

    cost = cost.detach().cpu().numpy()
    if use_joblib:
        permutations_list = Parallel(n_jobs = 8, backend = 'threading')(delayed(hungarian_solver)(cost_i) for cost_i in cost)   
    else:
        permutations_list = [hungarian_solver(cost_i) for cost_i in cost]
    log_solver = {} # Placeholder for now
    
    return permutations_list, log_solver

def batched_hungarian_projection(K, metric = 'KL'):
    '''
    Project a batch of positive matrices onto the set of permutations
    If metric = 'KL': return argmin_P KL(P||K) = argmin_P -<P,log(K)>
    If metric = 'F': return argmin_P ||P-K||_F = argmin_P -<P,K>
    '''
    if metric == 'KL':
        K = torch.clamp(K,1e-10,1)
        return batched_hungarian_solver(-K.log())
    elif metric == 'F':
        return batched_hungarian_solver(-K)

######################### SINKHORN #########################

def batched_sinkhorn_projection(K, max_iter = 10000, tol = 1e-5, last_iter_grad = False, fixed_n_iters = False):
    '''
    Project a batch of positive matrices onto the set of doubly stochastic matrices using Sinkhorn algorithm
    return argmin_T KL(T||K) = argmin_T -<T,log(K)>
    '''

    B, m, n = K.shape
    
    assert m == n, "Cost matrix must be square"

    # Weights [n,] and [m,]
    a = torch.ones(B,n, device = K.device, dtype=K.dtype)
    b = torch.ones(B,n, device = K.device, dtype=K.dtype)

    # Initialize the iteration with the change of variable
    f = torch.ones_like(a)
    g = torch.ones_like(b)
    
    with torch.set_grad_enabled(not last_iter_grad):
        
        n_iters = 0
        for _ in range(max_iter):
            
            f_prev = f
            g_prev = g
            
            summand_f = (K*g[:,None,:]).sum(dim=2)
            f = a / summand_f
            
            summand_g = (K*f[:,:,None]).sum(dim=1)
            g = b / summand_g
            
            n_iters += 1
            
            if not fixed_n_iters:
                max_err_u = torch.max(torch.abs(f_prev-f))
                max_err_v = torch.max(torch.abs(g_prev-g))
                if max_err_u < tol and max_err_v < tol:
                    break
            
    if last_iter_grad:
        summand_f = (K*g[:,None,:]).sum(dim=2)
        f = a / summand_f
        
        summand_g = (K*f[:,:,None]).sum(dim=1)
        g = b / summand_g

    P = (K * f[:,:,None] * g[:,None,:])
    summand_f = P.sum(dim=2)
    summand_g = P.sum(dim=1)
    max_err_a = torch.amax(torch.abs(a - summand_f), dim=-1).mean()
    max_err_b = torch.amax(torch.abs(b - summand_g), dim=-1).mean()
    log_solver = {'n sinkhorn iters': n_iters, 'sinkhorn marginal error (rows)': max_err_a, 'sinkhorn marginal error (cols)': max_err_b} 

    return P, log_solver
    

def batched_log_sinkhorn_projection(K, max_iter = 10000, tol = 1e-5, last_iter_grad = False, fixed_n_iters = False):
    '''
    Project exp(K) onto the set of doubly stochastic matrices, using log-sum-exp trick and Sinkhorn algorithm
    return argmin_T KL(T||exp(K)) = argmin_T -<T,exp(K)>
    '''

    B, m, n = K.shape
    
    assert m == n, "Cost matrix must be square"

    # Weights [n,] and [m,]
    a = torch.ones(B,n, device = K.device, dtype=K.dtype)
    b = torch.ones(B,n, device = K.device, dtype=K.dtype)

    log_a = torch.log(a)  # [n]
    log_b = torch.log(b)  # [m]

    # Initialize the iteration with the change of variable
    u = torch.zeros_like(a)
    v = torch.zeros_like(b)
    
    with torch.set_grad_enabled(not last_iter_grad):
        
        n_iters = 0
        for _ in range(max_iter):
            
            u_prev = u
            v_prev = v

            summand_u = (K + v[:,None,:]).logsumexp(dim=2).squeeze() 
            u = (log_a - summand_u)

            summand_v = (K + u[:,:,None]).logsumexp(dim=1).squeeze()
            v = (log_b - summand_v)
            
            n_iters += 1
            
            if not fixed_n_iters:
                max_err_u = torch.max(torch.abs(u_prev-u))
                max_err_v = torch.max(torch.abs(v_prev-v))
                if max_err_u < tol and max_err_v < tol:
                    break
            
    if last_iter_grad:
        summand_u = (K + v[:,None,:]) 
        u = (log_a - summand_u.logsumexp(dim=2).squeeze())

        summand_v = (K + u[:,:,None]) 
        v = (log_b - summand_v.logsumexp(dim=1).squeeze())

    log_P = (K + u[:,:,None] + v[:,None,:])
    P = log_P.exp()
    
    summand_f = P.sum(dim=2)
    summand_g = P.sum(dim=1)
    max_err_a = torch.amax(torch.abs(a - summand_f), dim=-1).mean()
    max_err_b = torch.amax(torch.abs(b - summand_g), dim=-1).mean()
    log_solver = {'n sinkhorn iters': n_iters, 'sinkhorn marginal error (rows)': max_err_a.item(), 'sinkhorn marginal error (cols)': max_err_b.item()} 

    return P, log_P, log_solver

class AbstractMatcher(torch.nn.Module):
    '''
    Abstract class for Matcher part of the Graph Autoencoder.
    '''
    def forward(self, node_embeddings_inputs: torch.Tensor, node_embeddings_outputs: torch.Tensor, hard = False):
        '''
        Input: 
            node_masks_inputs of shape (batch_size, n_nodes_max)
            node_embeddings_inputs of shape (batch_size, n_nodes_max, node_model_dim)   
            node_embeddings_outputs of shape (batch_size, n_nodes_max, node_model_dim)
        Outputs:
            permutations to apply to the outputs to match the orderering of the inputs i.e.
                permutation_list: list of numpy array (n_nodes_max,) if hard 
                permutation_matrices: Tensor (batch_size, n_nodes_max, n_nodes_max) else
            log_solver: dict of additional information to log
        '''
        if hard:
            permutation_list = ...
            log_solver = ...
            return permutation_list, log_solver
        else:
            permutation_matrices = ...
            log_solver = ...
            return permutation_matrices, log_solver

class SinkhornMatcher(AbstractMatcher):
    '''
    a_i = Linear(node_embeddings_outputs_i)
    b_j = Linear(node_embeddings_inputs_j)
    aff_ij = <a_i, b_j>
    K = 0.5 softmax(aff_ij, dim = 0) + 0.5 softmax(aff_ij, dim = 1) (approximate marginals)
    P = Sinkhorn(K) (K iterations of Sinkhorn projections, marginal might not be fully respected if it didn't converge)
    Cost is quadratic.
    '''
    def __init__(self, 
                 node_model_dim,
                 matcher_dim,
                 n_nodes_max,
                 max_iter_sinkhorn = 100,
                 tol_sinkhorn = 1e-3,
                 fixed_n_iters_sinkhorn = False,
                 normalize_cost_matrix = True,
                 epsilon = 1e-4
                 ):
        
        super().__init__()
        
        self.node_padding_features = torch.nn.Parameter(torch.randn(1,1,node_model_dim))
        torch.nn.init.xavier_uniform_(self.node_padding_features)
        self.positionnal_encoding_outputs = torch.nn.Parameter(torch.randn(1,n_nodes_max,node_model_dim))
        torch.nn.init.xavier_uniform_(self.positionnal_encoding_outputs)
        self.linear_inputs = torch.nn.Linear(node_model_dim,matcher_dim)
        self.linear_outputs = torch.nn.Linear(node_model_dim,matcher_dim)
    
        self.max_iter = max_iter_sinkhorn
        self.tol = tol_sinkhorn
        self.fixed_n_iters = fixed_n_iters_sinkhorn
        self.normalize_cost_matrix = normalize_cost_matrix
        self.epsilon = epsilon
        
    def forward(self, node_embeddings_inputs: torch.Tensor, node_masks_inputs: torch.Tensor, node_embeddings_outputs: torch.Tensor, hard = False):
        batchsize, n_nodes_max, _ = node_embeddings_inputs.shape
        # Add padding features to node_embeddings_inputs
        padding = self.node_padding_features.expand(batchsize,n_nodes_max,-1)
        mask = node_masks_inputs.unsqueeze(-1)
        node_embeddings_inputs = (~mask)*node_embeddings_inputs + mask*padding
        # Add positionnal encoding to outputs
        node_embeddings_outputs = node_embeddings_outputs + self.positionnal_encoding_outputs
        # Compute affinity
        a = self.linear_outputs(node_embeddings_outputs)
        b = self.linear_inputs(node_embeddings_inputs)
        C = pairwise_L1_norm(a,b)
        if self.normalize_cost_matrix:
            C = C/C.sum(dim=(1,2),keepdim=True)
        C = C/self.epsilon
        log_K = -C
        if hard:
            list_of_perms, log_solver = batched_hungarian_projection(log_K, metric='F')
            list_of_perms = np.array(list_of_perms)
            perms = torch.from_numpy(list_of_perms).long().to(node_embeddings_inputs.device)
            I = torch.eye(perms.size(1), dtype=torch.float32, device=node_embeddings_inputs.device)
            return I[perms], log_solver
        else:
            P, log_P, log_solver = batched_log_sinkhorn_projection(log_K,max_iter=self.max_iter,tol=self.tol, fixed_n_iters = self.fixed_n_iters)
            return P, log_solver