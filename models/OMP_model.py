import torch
import torch.nn as nn
from utils.dictionary_gen_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OMP_3D_model(nn.Module):
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
            MS_ant_position = MS_ant_position.to(device) #![u,8,3]
            self.MS_learnable_pos_list = nn.ParameterList([nn.Parameter(MS_ant_position[u, :, 1].clone()) for u in range(MS_ant_position.shape[0])])
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


    def forward(self, Y, user_idx, sigma2_est, iter_max=30):
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
        MS_ant_position = torch.stack([self.MS_fixed_pos_x[user_idx], self.MS_learnable_pos_list[user_idx], self.MS_fixed_pos_z[user_idx]], dim=1)

        D1=steering_vect_dict(self.BS_DoA, BS_ant_position, self.BS_ant_gains, self.BS_coupling_coeff, self.lambda_)
        D2=steering_vect_dict(self.MS_DoA, MS_ant_position, self.MS_ant_gains, self.MS_coupling_coeff, self.lambda_)
        D3=frequency_response_vect_dict(self.delays, self.subcarriers, None)
        D3=D3.to(dtype=torch.complex128)

        N = Y.numel()                # total number of elements in the channel
        iter = 0                        # iteration counter
        I_list = []                  # store indices of selected atoms
        h_reshaped = Y.reshape(-1)   # flatten channel for linear algebra
        D_I_list = []                # store selected dictionary atoms
        r = Y                        # initialize residual
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
            r = r_reshaped.reshape(Y.shape)

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

class OMP_1D_model(nn.Module):
    def __init__(self, ant_position, ant_gains, coupling_coeff, DoA, f0=28e9, device=device):
        
        super().__init__()
        # --- BS antenna positions ---
        ant_position = ant_position.to(device)
        self.learnable_ant_pos_y = nn.Parameter(ant_position[:, 1].clone())   # learnable y-coordinates
        self.register_buffer('fixed_ant_pos_x', ant_position[:, 0].detach())  # fixed x-coordinates
        self.register_buffer('fixed_ant_pos_z', ant_position[:, 2].detach())  # fixed z-coordinates
        # --- BS antenna gains and coupling ---
        self.ant_gains = nn.Parameter(ant_gains.to(device).to(torch.complex128))
        self.coupling_coeff = nn.Parameter(coupling_coeff.to(device))     # complex coupling coefficitent

        # --- Other parameters ---

        self.register_buffer('DoA', DoA.to(device))

        self.lambda_ = 3e8 / f0                     # carrier wavelength
        self.nb_antennas = len(ant_position)  # number of antennas


    def forward(self,Y,sigma2_est,iter_max=10):
        '''handles batched operations'''

        ant_position = torch.stack([self.fixed_ant_pos_x,self.learnable_ant_pos_y,self.fixed_ant_pos_z], dim=1)

        D=steering_vect_dict(self.DoA, ant_position, self.ant_gains, self.coupling_coeff, self.lambda_)
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
    

class OMP_ML_model(nn.Module):
    def __init__(self, D, device=device):
        
        super().__init__()
        # --- BS antenna positions ---
        self.D = nn.Parameter(D.to(device))   # learnable Dictionary

    def forward(self,Y,sigma2_est,iter_max=10):
        '''handles batched operations'''


        N=Y.shape[1:].numel()
        iter = 0
        I_list=[]
        D_I_list=[]
        y=Y.unsqueeze(-1)
        r = y  # ([204800, 8, 1]) => batch_size= 204800

        stop=False

        while not stop:
            corr=(torch.conj(self.D).T).unsqueeze(0)@r  #([*, 80,1])
            corr=corr.squeeze() #([*,80])
            i = torch.argmax(corr.abs()**2,dim=1) #([*])
            I_list.append(i)

            D_I_list.append(self.D[:,i].T)
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