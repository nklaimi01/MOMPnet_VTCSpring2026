import torch 
import time

import torch
def normalize(H):
    if H.dim() == 4 or H.dim()==5:
        dims = (-3,-2,-1)
    elif H.dim() == 2:
        dims = (1,)
    else:
        raise ValueError(f"channel must be 2-D, 4-D or 5-D, got {H.dim()}-D")
    return H / torch.sqrt(torch.sum(torch.abs(H)**2, dim=dims, keepdim=True))

def stack_with_padding(tensors, dim=0, length=None, zero_padding=False):
    """
    Pad a list of 1D tensors so they all have the same length, 
    then stack them along the given dimension.
    
    Args:
        tensors (list[torch.Tensor]): List of 1D tensors.
        dim (int): Dimension along which to stack.
        length (int, optional): Target length. Defaults to the max length.
        zero_padding (bool): If True, pad with zeros; else with the last value of each tensor.
    """
    if not tensors:
        raise ValueError("Input list 'tensors' cannot be empty.")

    # Check all tensors are 1D
    if not all(t.ndim == 1 for t in tensors):
        raise ValueError("All tensors must be 1D.")

    if length is None:
        length = max(t.size(0) for t in tensors)

    padded = []
    for t in tensors:
        pad_len = length - t.size(0)
        if pad_len > 0:
            # Ensure padding value has the same dtype and device
            if zero_padding:
                pad_value = torch.zeros(1, dtype=t.dtype, device=t.device)
            else:
                pad_value = t[-1:].clone()  # keep same dtype/device safely

            last_vals = pad_value.expand(pad_len)
            t = torch.cat([t, last_vals])

        padded.append(t)

    return torch.stack(padded, dim=dim)

def NMSE(channel,channel_estimation):
    if channel.dim() == 4 or channel.dim()==5:
        dims = (-3,-2, -1)
    elif channel.dim() == 2:
        dims = (1,)
    else:
        raise ValueError(f"channel must be 2-D , 4-D or 5-D, got {channel.dim()}-D")

    return torch.sum(torch.abs(channel - channel_estimation)**2, dim=dims) /torch.sum(torch.abs(channel)**2, dim=dims)

def model_estimation(Y, model, sigma2_est):
            H_est = torch.zeros_like(Y)
            for u in range(Y.shape[0]):
                for p in range(Y.shape[1]):
                    y = Y[u, p]
                    y = y.squeeze()

                    res, _, _ = model.forward(y, u, sigma2_est)
                    H_est[u, p] = y - res
            return H_est

def MOMP(Y,D1,D2,D3, sigma2_est, iter_max=30, refine_iter=2):
        """
        Performs MOMP to approximate
        the channel H using dictionaries D1, D2, and D3.

        Parameters
        ----------
        H : torch.Tensor
            Observed channel tensor to be approximated.
        sigma2_est : float
            Estimated noise variance for the stopping criterion.
        iter_max : int, optional
            Maximum number of iterations (default: 30).
        refine_iter : int, optional
            Number of refinement steps to improve atom selection (default: 2).

        Returns
        -------
        r : torch.Tensor
            Final residual tensor after MOMP iterations.
        I : torch.Tensor
            Indices of selected atoms [i1, i2, i3] at each iteration.
        x : torch.Tensor
            Coefficients corresponding to the selected atoms.

        """
        D1 = D1.to(dtype=torch.complex128)
        D2 = D2.to(dtype=torch.complex128)
        D3 = D3.to(dtype=torch.complex128)

        # --------------------------------------------------------------------------
        # Initialization
        # --------------------------------------------------------------------------
        N = Y.numel()              # Total number of elements in H
        stop = False
        iter = 0
        I_list = []                # List of selected index triplets per iteration
        h_reshaped = Y.reshape(-1) # Flattened channel tensor
        D_I_list = []              # List of selected atoms
        r = Y                      # Initialize residual with input tensor

        # --------------------------------------------------------------------------
        # Main MOMP iteration loop
        # --------------------------------------------------------------------------
        while not stop:
            # Step 1: Compute correlations with D1 along the first dimension
            corr1 = torch.einsum('ab,bms->ams', torch.conj(D1).T, r)
            i1 = torch.argmax((corr1.abs()**2).sum(dim=(1, 2)))

            # Step 2: Select best atom from D2 using the previous selection from D1
            corr2 = torch.conj(D2).T @ corr1[i1]
            i2 = torch.argmax((corr2.abs()**2).sum(dim=1))

            # Step 3: Select best atom from D3 using the previous selections
            corr3 = torch.conj(D3).T @ corr2[i2]
            i3 = torch.argmax(torch.abs(corr3)**2)

            # ----------------------------------------------------------------------
            # Optional refinement of atom indices via local coordinate updates
            # ----------------------------------------------------------------------
            if refine_iter is not None:
                atom = [i1, i2, i3]
                D = [D1, D2, D3]

                for _ in range(refine_iter):
                    for d in range(len(atom)):
                        # Identify the two remaining dimensions besides d
                        other_idx1, other_idx2 = (set(range(len(atom))) - {d})

                        # Extract the selected atoms along the other dimensions
                        vec_0 = D[other_idx1][:, atom[other_idx1]]
                        vec_1 = D[other_idx2][:, atom[other_idx2]]

                        # Permute residual to align dimensions [other1, other2, d]
                        r_permuted = r.permute(other_idx1, other_idx2, d)

                        # Compute correlation along dimension d
                        corr_d = torch.einsum('a,abc,b->c', torch.conj(vec_0), r_permuted, torch.conj(vec_1))
                        corr_d = torch.matmul(torch.conj(D[d]).T, corr_d)

                        # Update atom index along dimension d with the highest correlation
                        i_d = torch.argmax(torch.abs(corr_d)**2)
                        atom[d] = i_d

                # Update selected indices after refinement
                i1, i2, i3 = atom

            # Store current triplet of selected atom indices
            I_list.append(torch.tensor([i1, i2, i3]))

            # ----------------------------------------------------------------------
            # Construct current dictionary from selected atoms (Kronecker structure)
            # ----------------------------------------------------------------------
            vec1 = D1[:, i1]
            vec2 = D2[:, i2]
            vec3 = D3[:, i3]
            D_I_list.append(torch.kron(torch.kron(vec1, vec2), vec3))
            D_I = torch.stack(D_I_list, 1)

            # ----------------------------------------------------------------------
            # Solve least squares problem to estimate coefficients
            # ----------------------------------------------------------------------
            x = torch.linalg.lstsq(D_I, h_reshaped).solution
            proj_h = D_I @ x

            # ----------------------------------------------------------------------
            # Update residual and iteration counter
            # ----------------------------------------------------------------------
            r_reshaped = h_reshaped - proj_h
            r = r_reshaped.reshape(Y.shape)
            iter += 1

            # ----------------------------------------------------------------------
            # Check stopping criteria: residual energy or iteration limit
            # ----------------------------------------------------------------------
            if sigma2_est is None:
                SC=False
            else:
                SC=torch.sum(torch.abs(r)**2) <= N * sigma2_est

            if  SC or iter > iter_max - 1:
                stop = True

        # Stack selected atom indices across all iterations
        I = torch.stack(I_list, 0)

        return r, I, x

def MOMP_estimation(Y, D1, D2, D3, sigma2_est):
            H_est = torch.zeros_like(Y)
            for u in range(Y.shape[0]):
                if D2.dim()==2:
                    D2u=D2
                else:
                    D2u=D2[u]
                for p in range(Y.shape[1]):
                    y = Y[u, p]
                    y = y.squeeze()
                    res, _, _ = MOMP(y,D1,D2u,D3, sigma2_est)
                    H_est[u, p] = y - res
            return H_est

def OMP(Y, D,sigma2_est=None, iter_max=10):
    '''handles batched operations'''
    N=Y.shape[1:].numel()
    iter = 0
    I_list=[]
    D_I_list=[]
    y=Y.unsqueeze(-1)
    r = y  # ([204800, 8, 1]) => batch_size= 204800

    stop=False

    while not stop:
        corr=(torch.conj(D).T).unsqueeze(0)@r  #([*, 80,1])
        corr=corr.squeeze() #([*,80])
        i = torch.argmax(corr.abs()**2,dim=1) #([*])
        I_list.append(i)

        D_I_list.append(D[:,i].T)
        D_I=torch.stack(D_I_list,-1) #([*, 8, nb_active_atoms])

        # Step 4: projection (solve least-squares to update coefficients)
        gamma = torch.linalg.lstsq(D_I, y).solution
        proj_y = D_I @ gamma

        # Step 5: update residual
        r = y - proj_y

        iter += 1

        if sigma2_est is None:
            SC=False
        else:
            SC=torch.sum(torch.abs(r)**2) <= N * sigma2_est
            
        if  SC or iter > iter_max - 1:
            stop = True

    # Stack all estimations along first dimension
    I=torch.stack(I_list,-1) #([*, nb_active_atoms])
    gamma=gamma.squeeze(-1) #([*, nb_active_atoms])
    r=r.squeeze(-1)
    return r,I,gamma

def MOD(Y,D0,OMP_iter,epsilon,iter_max=1000,torchlstsq=False):
    '''implementation of the method of optimal directions (MOD), a Dictionary learning algorithm'''
    batch_size=Y.shape[0]
    stop=False
    iter=0
    start_MOD=time.time()
    while not stop: 
        # step 1: sparse recovery
        _,I,gamma=OMP(Y,D0,iter_max=OMP_iter)
        Gamma=torch.zeros((batch_size,D0.shape[1]),dtype=gamma.dtype)
        batch_idx = torch.arange(batch_size).unsqueeze(-1)
        Gamma[batch_idx, I] = gamma
        #step 2: update dictionary 
        if torchlstsq:
            #using torch.linalg.lstsq
            sol = torch.linalg.lstsq(Gamma, Y)
            D_MOD = sol.solution          
            D_MOD = D_MOD.T               
            D_MOD = D_MOD / (torch.norm(D_MOD, dim=0, keepdim=True) + 1e-8) # normalize atoms
        else:
            YmT=Y.T # shape ([N_M, N_obs])
            Gamma=Gamma.T # shape ([A_M, N_obs])
            Gamma_H = Gamma.conj().T  
            term = Gamma @ Gamma_H
            term_inv = torch.linalg.inv(term)
            D_MOD = YmT @ Gamma_H @ term_inv
            D_MOD = D_MOD / torch.norm(D_MOD, dim=0, keepdim=True)  # normalize atoms

        if torch.norm(D_MOD-D0)/torch.norm(D0)<epsilon or iter>iter_max:
            stop=True
        
        iter+=1
        # print(f'iteration: {iter}, SC={torch.norm(D_MOD-D0)/torch.norm(D0)}')
        D0=D_MOD
    end_MOD = time.time()
    print(f"MOD time: {end_MOD - start_MOD:.6f} seconds")
    
    return D_MOD

def mode_unfold(Y, m):
    '''reshape observation by m-mode unfolding'''
    # Y is an N-way tensor
    N = Y.ndim
    
    # Create permutation: bring dimension r to the front
    perm = [i for i in range(N) if i != m] + [m] 
    
    # Permute and reshape
    return Y.permute(*perm).reshape(-1, Y.shape[m])

def recover_unfold(Y_unf, r, shape):
    '''recover observation shape'''
    N = len(shape)

    # Build the same perm used in unfolding
    perm = [i for i in range(N) if i != r] + [r]

    # Compute the permuted shape (after unfolding)
    permuted_shape = [shape[i] for i in perm]

    # First reshape back to the permuted tensor
    Y_perm = Y_unf.reshape(*permuted_shape)

    # Now invert permutation
    # Create inverse permutation: inv_perm[perm[i]] = i
    inv_perm = [0] * N
    for i, p in enumerate(perm):
        inv_perm[p] = i

    # Return tensor in original order
    return Y_perm.permute(*inv_perm)