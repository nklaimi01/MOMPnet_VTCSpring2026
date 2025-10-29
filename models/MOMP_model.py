import torch
import torch.nn as nn
from utils.dictionary_gen_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MOMP_model(nn.Module):
    """
    PyTorch module implementing Multi-Dimensional Orthogonal Matching Pursuit (MOMP) 
    for sparse channel estimation with a Kronecker-structured dictionary.
    """

    def __init__(self, BS_ant_position, BS_ant_gains, BS_coupling_coeff,
                    MS_ant_position, subcarriers, BS_DoA, MS_DoA, delays, f0=28e9, device=device):
            super().__init__()

            # --- BS antenna positions ---
            BS_ant_position = BS_ant_position.to(device)
            self.BS_learnable_pos_y = nn.Parameter(BS_ant_position[:, 1].clone())   # learnable y-coordinates
            self.register_buffer('BS_fixed_pos_x', BS_ant_position[:, 0].detach())  # fixed x-coordinates
            self.register_buffer('BS_fixed_pos_z', BS_ant_position[:, 2].detach())  # fixed z-coordinates

            # --- BS antenna gains and coupling ---
            self.BS_ant_gains = nn.Parameter(BS_ant_gains.to(device).to(torch.complex128))
            self.BS_coupling_coeff = nn.Parameter(BS_coupling_coeff.to(device))     # complex coupling coefficitent

            # --- MS antenna positions ---
            MS_ant_position = MS_ant_position.to(device) #[u,8,3]
            self.MS_learnable_pos_list = nn.ParameterList([nn.Parameter(MS_ant_position[u, :, 1].clone()) for u in range(MS_ant_position.shape[0])])
            # self.MS_learnable_pos_y = nn.Parameter(MS_ant_position[:, 1].clone())   # learnable y-coordinates
            self.register_buffer('MS_fixed_pos_x', MS_ant_position[:,:, 0].detach())  # fixed x-coordinates
            self.register_buffer('MS_fixed_pos_z', MS_ant_position[:,:, 2].detach())  # fixed z-coordinates

            # --- MS antenna gains and coupling ---
            self.register_buffer('MS_ant_gains', torch.ones(MS_ant_position.shape[1], device=device))
            self.register_buffer('MS_coupling_coeff', torch.tensor(0, device=device, dtype=torch.complex128))

            # --- Other parameters ---
            self.register_buffer('subcarriers', subcarriers.to(device))
            self.register_buffer('BS_DoA', BS_DoA.to(device))
            self.register_buffer('MS_DoA', MS_DoA.to(device))
            self.register_buffer('delays', delays.to(device))
            self.lambda_ = 3e8 / f0                     # carrier wavelength
            self.nb_BS_antennas = len(BS_ant_position)  # number of BS antennas


    def forward(self, H, user_idx, sigma2_est, iter_max=30, refine_iter=2):
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


        # --------------------------------------------------------------------------
        # Construct steering vector dictionaries for BS, MS, and frequency responses
        # --------------------------------------------------------------------------
        BS_ant_position = torch.stack([self.BS_fixed_pos_x, self.BS_learnable_pos_y, self.BS_fixed_pos_z], dim=1)
        MS_ant_position = torch.stack([self.MS_fixed_pos_x[user_idx], self.MS_learnable_pos_list[user_idx], self.MS_fixed_pos_z[user_idx]], dim=1)
        
        D1 = steering_vect_dict(self.BS_DoA, BS_ant_position, self.BS_ant_gains, self.BS_coupling_coeff, self.lambda_)
        D2 = steering_vect_dict(self.MS_DoA, MS_ant_position, self.MS_ant_gains, self.MS_coupling_coeff, self.lambda_)
        D3 = frequency_response_vect_dict(self.delays, self.subcarriers, None)
        D3 = D3.to(dtype=torch.complex128)

        # --------------------------------------------------------------------------
        # Initialization
        # --------------------------------------------------------------------------
        N = H.numel()              # Total number of elements in H
        stop = False
        iter = 0
        I_list = []                # List of selected index triplets per iteration
        h_reshaped = H.reshape(-1) # Flattened channel tensor
        D_I_list = []              # List of selected atoms
        r = H                      # Initialize residual with input tensor

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
            I_list.append(torch.tensor([i1, i2, i3], device=device))

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
            r = r_reshaped.reshape(H.shape)
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

