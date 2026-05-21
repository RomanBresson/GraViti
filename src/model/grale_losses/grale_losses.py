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
from torch.nn.functional import logsigmoid
from torch.special import entr as entr_torch
from math import sqrt
from torch.nn.functional import logsigmoid

def bmv(Matrices,Vectors):
    return torch.einsum('bij,bj->bi', Matrices, Vectors)

def bop(vectors1,vectors2):
    return torch.einsum('bi,bj->bij', vectors1, vectors2)

###### ------------------ Usefull functions to build cost matrices ------------------ ######

def squared_norm(X1,X2):
    dim = X1.shape[-1]
    return torch.sum((X1-X2)**2,dim=-1)/sqrt(dim)

def pairwise_squared_norm(X1,X2):
    dim = X1.shape[-1]
    norm1 = torch.sum(X1**2,dim=2)
    norm2 = torch.sum(X2**2,dim=2)
    dot = torch.bmm(X1,X2.permute(0,2,1))
    loss = norm1[:,:,None] + norm2[:,None,:] - 2*dot
    loss = loss/sqrt(dim)
    return loss

def pairwise_L1_norm(X1,X2):
    dim = X1.shape[-1]
    return torch.sum(torch.abs(X1[:,:,None]-X2[:,None,:]),dim=-1)/sqrt(dim)

def pairwise_BCE_loss(logits,targets):
    batchsize, M = logits.shape
    logits = logits.unsqueeze(-1).expand(-1,-1,M)
    targets = targets.unsqueeze(-2).expand(-1,M,-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    return loss

def pairwise_CE_loss(logits,targets):
    return -torch.log_softmax(logits,dim=-1)@torch.permute(targets,(0,2,1)) 

def softmax_to_one_hot(logits):
    classes = torch.argmax(logits,dim=-1,keepdim=True)
    one_hot = torch.zeros_like(logits).scatter_(-1, classes, 1)
    return one_hot 

class LinearLoss():
    '''
    Compute the loss L(T,C1,C2) = sum_ik T_ik L(F_i, F_k)
    For C1 of shape (B,N1,D), C2 of shape (B,N2,D2) and T of shape (B,N1,N2)

    It is also possible to add weights to the loss:
    L(T,C1,C2) = sum_ik T_ik L(F_i, F_k) w1_i w2_k
    '''

    def pairwise_loss(self, F1, F2):
        '''
        Build cost matrix C_ij = L(F1_i, F2_j)
        '''
        return ...

    def pointwise_loss(self, F1, F2):
        '''
        Compute pointwise loss C_i = L(F1_i, F2_i)
        '''
        return ...

    def forward(self, T, F1, F2, weight_1=None, weight_2=None):
        cost_matrix = self.pairwise_loss(F1,F2)
        if weight_1 is not None:
            cost_matrix = cost_matrix * weight_1[:,:,None]
        if weight_2 is not None:
            cost_matrix = cost_matrix * weight_2[:,None,:]
        return torch.sum(T*cost_matrix, dim=(1,2))

    def forward_aligned(self, F1, F2, weight = None):
        cost = self.pointwise_loss(F1,F2)
        cost = cost * weight if weight is not None else cost
        return torch.sum(cost, dim=1)

class LinearL2(LinearLoss):

    def pairwise_loss(self, F1, F2):
        '''
        Build cost matrix C_ij = L(F1_i, F2_j)
        '''
        return pairwise_squared_norm(F1,F2)

    def pointwise_loss(self, F1, F2):
        '''
        Compute pointwise loss C_i = L(F1_i, F2_i)
        '''
        return squared_norm(F1,F2)

class LinearBCE(LinearLoss):

    def pairwise_loss(self, logits, targets):
        return pairwise_BCE_loss(logits,targets)

    def pointwise_loss(self, logits, targets):
        return torch.nn.BCEWithLogitsLoss(reduction='none')(logits,targets) 

class LinearBinaryAccuracy(LinearLoss):

    def pairwise_loss(self, predictions, targets):
        preds = torch.where(predictions>0.5,1,0)
        cost = torch.where(preds[:,:,None]==targets[:,None,:],0,1)
        return cost
 
    def pointwise_loss(self, logits, targets):
        preds = torch.where(logits>0.5,1,0)
        cost = torch.where(preds == targets, 0 , 1)
        return cost
 
class LinearCE(LinearLoss):

    def pairwise_loss(self, logits, targets):
        return pairwise_CE_loss(logits,targets)

    def pointwise_loss(self, logits, targets):
        return -(torch.log_softmax(logits,dim=-1)*targets).sum(dim=-1)

class LinearAccuracy(LinearLoss):

    def pairwise_loss(self, predictions, targets):
        preds = softmax_to_one_hot(predictions)
        cost = 1 - torch.bmm(preds, targets.transpose(1,2))
        return cost

    def pointwise_loss(self, predictions, targets):
        preds = softmax_to_one_hot(predictions)
        cost = 1 - (preds*targets).sum(-1)
        return cost


class QuadraticLoss():
    '''
    Compute the loss L(T,C1,C2) = sum_ijkl T_ik T_jl L(C1_ij, C2_kl)
    For C1 of shape (B,N1,N1,D), C2 of shape (B,N2,N2,D2) and T of shape (B,N1,N2)
    Assumes that L(a,b) = f1(a) + f2(b) - < h1(a), h2(b) >

    It is also possible to add weights to the loss:
    L(T,C1,C2) = sum_ijkl T_ik T_jl L(C1_ij, C2_kl) W1_ij W2_kl

    For instance, setting mask_self_loops = True will set:
         W1 = 1 - diag(1) 
         W2 = 1 - diag(1)
    Providing weight_1 and weight_2 of shape (B,N1) and (B,N2) will set:
        W1_ij = weight_1_i weight_1_j 
        W2_kl = weight_2_k weight_2_l
    '''

    def __init__(self, mask_self_loops = False):
        self.mask_self_loops = mask_self_loops

    def f1(self, C1):
        return ...

    def f2(self, C2):
        return ...

    def h1(self, C1):
        return ...

    def h2(self, C2):
        return ...

    def forward(self, T, C1, C2, weight_1=None, weight_2=None):
        L = self.tensor_product(T,self.f1(C1),self.f2(C2),self.h1(C1),self.h2(C2),weight_1,weight_2,self.mask_self_loops)
        return torch.sum(L*T, dim=(1,2))

    def forward_aligned(self, C1, C2, weight=None):
        L = self.f1(C1) + self.f2(C2) - torch.einsum('bijd,bijd->bij', self.h1(C1), self.h2(C2))
        if weight is not None:
            L = L * weight[:,:,None] * weight[:,None,:]
        if self.mask_self_loops:
            L = L * (1 - torch.eye(L.shape[1], device = L.device)).unsqueeze(0)
        return torch.sum(L, dim=(1,2))

    def tensor_product(self,T,f1,f2,h1,h2,weight_1=None,weight_2=None,mask_self_loops=False):
        '''
        Compute Tensor_ik = sum_jl T_jl L(C1_ij, C2_kl) W1_ij W2_kl
        where:
            L is the lost function that decomposes as L(a,b) = f1(a) + f2(b) - < h1(a), h2(b) >
            W1 and W2 are masks defined as:
                W = mask_self_loops * weight
                mask_self_loops = 1 - diag(1) if mask_self_loops = True else 1
                mask_nodes = mask * mask.t() if mask is not None else 1
        '''

        B, N1, N2 = T.shape

        w1 = weight_1 if weight_1 is not None else torch.ones(B,N1, device = T.device)
        w2 = weight_2 if weight_2 is not None else torch.ones(B,N2, device = T.device)

        # Initialize W1 with as few operations as possible (try to avoid multipliLation by 1)
        if mask_self_loops and weight_1 is not None:
            W1 = 1 - torch.eye(N1, device = T.device).unsqueeze(0).repeat(B,1,1)
            W1 = W1 * w1[:,None,:] * w1[:,:,None]
        elif mask_self_loops:
            W1 = 1 - torch.eye(N1, device = T.device).unsqueeze(0).repeat(B,1,1)
        elif weight_1 is not None:
            W1 = w1[:,None,:] * w1[:,:,None]

        # Initialize W2 with as few operations as possible (try to avoid multipliLation by 1)
        if mask_self_loops and weight_2 is not None:
            W2 = 1 - torch.eye(N2, device = T.device).unsqueeze(0).repeat(B,1,1)
            W2 = W2 * w2[:,None,:] * w2[:,:,None]
        elif mask_self_loops:
            W2 = 1 - torch.eye(N2, device = T.device).unsqueeze(0).repeat(B,1,1)
        elif weight_2 is not None:
            W2 = w2[:,None,:] * w2[:,:,None]

        # Compute U1 = f1 * W1 and V1 = h1 * W1
        U1 = f1
        V1 = h1
        if mask_self_loops or weight_1 is not None:
            U1 = U1 * W1
            V1 = V1 * W1.unsqueeze(-1)

        # Compute U2 = f2 * W2 and V2 = h2 * W2
        U2 = f2
        V2 = h2
        if mask_self_loops or weight_2 is not None:
            U2 = U2 * W2
            V2 = V2 * W2.unsqueeze(-1)

        # Compute La = U1@T@W2^T, this can be done faster if W2 = w2 w2^T
        if mask_self_loops: # No acceleration possible
            La = torch.bmm(U1, torch.bmm(T, W2.transpose(1,2)))
        else: # compute only matrix/vector products + an outer product
            La = bmv(T,w2)
            La = bmv(U1,La)
            La = bop(La,w2)

        # Compute Lb = W1@T@U2^T, this can be done faster if W1 = w1 w1^T
        if mask_self_loops:
            Lb = torch.bmm(W1, torch.bmm(T, U2.transpose(1,2)))
        else:
            Lb = bmv(T.transpose(1,2),w1)
            Lb = bmv(U2,Lb)
            Lb = bop(w1,Lb)

        # Compute Lc = sum_d (V1@T@V2^T)_d, no acceleration possible
        Lc = torch.einsum('bijd,bjl,bkld->bik', V1, T, V2)

        L = La + Lb - Lc

        return L

class QuadraticL2(QuadraticLoss):

    def f1(self, C1):
        if C1.ndim == 4:
            return torch.sum(C1**2,dim=-1)
        else:
            return C1**2

    def f2(self, C2):
        if C2.ndim == 4:
            return torch.sum(C2**2,dim=-1)
        else:
            return C2**2

    def h1(self, C1):
        if C1.ndim == 3:
            C1 = C1.unsqueeze(-1)
        return 2*C1

    def h2(self, C2):
        if C2.ndim == 3:
            C2 = C2.unsqueeze(-1)
        return C2

class QuadraticBCE(QuadraticLoss):
    '''
    L(a,b) = KL(sigmoid(a),b) + KL(1-sigmoid(a),1-b)
    Expect C1 to be given in logits (pre-sigmoid) and C2 to be given in probabilities
    '''

    def f1(self, logits):
        return -logsigmoid(logits)

    def f2(self, targets):
        return-entr_torch(targets)-entr_torch(1-targets)

    def h1(self, logits):
        return -logits.unsqueeze(-1)

    def h2(self, targets):
        return (1-targets).unsqueeze(-1)

class QuadraticCE(QuadraticLoss):
    '''
    L(a,b) = KL(softmax(a),b)
    Expect C1 to be given in logits (pre-softmax) and C2 to be given in probabilities
    '''

    def f1(self, logits):
        return torch.logsumexp(logits,dim=-1)

    def f2(self, targets):
        return -entr_torch(targets).sum(dim=-1)

    def h1(self, logits):
        return logits

    def h2(self, targets):
        return targets

class QuadraticAccuracy(QuadraticLoss):
    '''
    L(a,b) = 1[argmax(a)!=b] = 1 - <one_hot(a),b>
    Expect C1 to be a softmax and C2 to be a one hot encoded target
    '''

    def f1(self, logits):
        B, N, _, D = logits.shape
        return torch.ones((B,N,N), device=logits.device) 

    def f2(self, targets):
        B, N, _, D = targets.shape
        return torch.zeros((B,N,N), device=targets.device) 

    def h1(self, logits):
        return softmax_to_one_hot(logits)

    def h2(self, targets):
        return targets

class QuadraticBinaryAccuracy(QuadraticAccuracy):
    '''
    L(a,b) = 1[ (a>0.5) != b ]
    Can be computed by first defining A = [a, 1-a] and B = [b, 1-b] then
    L(a,b) = Accuracy(A,B)
    '''

    def transform(self, C1,C2):
        C1 = torch.stack([C1,1-C1], dim=-1)
        C2 = torch.stack([C2,1-C2], dim=-1)
        return C1, C2

    def forward(self, T, C1, C2, weight_1=None, weight_2=None):
        C1, C2 = self.transform(C1, C2)
        return super().forward(T, C1, C2, weight_1=weight_1, weight_2=weight_2)

    def forward_aligned(self, C1, C2, weight=None):
        C1, C2 = self.transform(C1, C2)
        return super().forward_aligned(C1, C2, weight = weight)

class MarginalLoss():
    '''
    Compute a loss on the marginals of T 
    Promotes T.sum(1) = 1, T.sum(2) = 1
    '''
    
    def marginal_loss(self, marginal):
        '''
        Compute the loss on a marginal
        '''
        raise NotImplementedError()
    
    def forward(self, T):
        marg1 = T.sum(1)
        marg2 = T.sum(2)
        return self.marginal_loss(marg1) + self.marginal_loss(marg2)
    
class MarginalKL(MarginalLoss):
    def marginal_loss(self, marginal):
        loss = (-torch.log(marginal) + marginal - 1).sum(-1)
        return loss
