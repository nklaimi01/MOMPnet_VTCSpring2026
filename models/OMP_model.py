import torch
import torch.nn as nn
from utils.dictionary_gen_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OMP_model(nn.Module):
    """
    PyTorch module implementing Orthogonal Matching Pursuit (OMP) for 
    sparse channel estimation with a Kronecker-structured dictionary.
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
            MS_ant_position = MS_ant_position.to(device)
            self.MS_learnable_pos_y = nn.Parameter(MS_ant_position[:, 1].clone())   # learnable y-coordinates
            self.register_buffer('MS_fixed_pos_x', MS_ant_position[:, 0].detach())  # fixed x-coordinates
            self.register_buffer('MS_fixed_pos_z', MS_ant_position[:, 2].detach())  # fixed z-coordinates

            # --- MS antenna gains and coupling ---
            self.register_buffer('MS_ant_gains', torch.ones(len(MS_ant_position), device=device))
            self.register_buffer('MS_coupling_coeff', torch.tensor(0, device=device, dtype=torch.complex128))

            # --- Other parameters ---
            self.register_buffer('subcarriers', subcarriers.to(device))
            self.register_buffer('BS_DoA', BS_DoA.to(device))
            self.register_buffer('MS_DoA', MS_DoA.to(device))
            self.register_buffer('delays', delays.to(device))
            self.lambda_ = 3e8 / f0                     # carrier wavelength
            self.nb_BS_antennas = len(BS_ant_position)  # number of BS antennas


    def forward(self, H, sigma2_est, iter_max=30):
        """
        Perform OMP to approximate the channel H using dictionaries D1, D2, D3.

        Parameters
        ----------
        H : torch.Tensor
            Observed channel tensor to be approximated.
        sigma2_est : float
            Estimated noise variance for the stopping criterion.
        iter_max : int, optional
            Maximum number of iterations (default: 30).

        Returns
        -------
        r : torch.Tensor
            Final residual tensor after MOMP iterations.
        I : torch.Tensor
            Indices of selected atoms [i1, i2, i3] at each iteration.
        x : torch.Tensor
            Coefficients corresponding to the selected atoms.

        """

        BS_ant_position = torch.stack([self.BS_fixed_pos_x,self.BS_learnable_pos_y,self.BS_fixed_pos_z], dim=1)
        MS_ant_position = torch.stack([self.MS_fixed_pos_x,self.MS_learnable_pos_y,self.MS_fixed_pos_z], dim=1)
        
        D1=steering_vect_dict(self.BS_DoA, BS_ant_position, self.BS_ant_gains, self.BS_coupling_coeff, self.lambda_)
        D2=steering_vect_dict(self.MS_DoA, MS_ant_position, self.MS_ant_gains, self.MS_coupling_coeff, self.lambda_)
        D3=frequency_response_vect_dict(self.delays, self.subcarriers, None)
        D3=D3.to(dtype=torch.complex128)

        N = H.numel()                # total number of elements in the channel
        iter = 0                        # iteration counter
        I_list = []                  # store indices of selected atoms
        h_reshaped = H.reshape(-1)   # flatten channel for linear algebra
        D_I_list = []                # store selected dictionary atoms
        r = H                        # initialize residual
        stop = False                 # stopping flag

        while not stop:
            # Step 1: compute correlations between residual and dictionary atoms
            AADM = torch.einsum('ab,bms->ams', torch.conj(D1).T, r) 
            AADM = torch.einsum('am,bms->bas', torch.conj(D2).T, AADM)
            AADM = torch.einsum('as,bms->bma', torch.conj(D3).T, AADM) 

            # Step 2: pick the atom with maximum correlation
            i1, i2, i3 = torch.unravel_index(torch.argmax(torch.abs(AADM)), AADM.shape)
            I_list.append(torch.tensor([i1, i2, i3], device=device))

            # Step 3: construct the current dictionary using selected atoms
            vec1 = D1[:, i1]
            vec2 = D2[:, i2]
            vec3 = D3[:, i3]
            D_I_list.append(torch.kron(torch.kron(vec1, vec2), vec3))
            D_I = torch.stack(D_I_list, 1)

            # Step 4: projection via least-squares to update coefficients
            x = torch.linalg.lstsq(D_I, h_reshaped).solution
            proj_h = D_I @ x

            # Step 5: update residual
            r_reshaped = h_reshaped - proj_h
            r = r_reshaped.reshape(H.shape)

            iter += 1

            # Step 6: stopping criterion based on residual energy or max iterations
            if sigma2_est is None:
                SC=False
            else:
                SC=torch.sum(torch.abs(r)**2) <= N * sigma2_est
                
            if  SC or iter > iter_max - 1:
                stop = True

        # Stack selected atom indices across iterations
        I = torch.stack(I_list, 0)

        return r, I, x
