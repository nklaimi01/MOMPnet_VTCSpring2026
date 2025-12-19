import torch
import torch.nn as nn
from utils.dictionary_gen_utils import *

import torch.nn as nn
class MP_1D_model(nn.Module):
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


    def forward(self,y,sigma2_est=None,iter_max=10):
        '''handles batched operations'''

        ant_position = torch.stack([self.fixed_ant_pos_x,self.learnable_ant_pos_y,self.fixed_ant_pos_z], dim=1)
        D=steering_vect_dict(self.DoA, ant_position, self.ant_gains, self.coupling_coeff, self.lambda_)
        N=y.shape[1:].numel()
        iter = 0
        I_list=[]
        r = y  # ([*, 8, 1]) => batch_size= *
        stop=False
        gamma0 = None
        while not stop:
            corr=(torch.conj(D).T).unsqueeze(0)@(r.unsqueeze(-1))  #([*, 80,1])
            corr=corr.squeeze() #([*,80])
            i = torch.argmax(corr.abs()**2,dim=1) #([*])
            I_list.append(i)

            D_I=D[:,i].T #([*, 8]) 

            # Step 4: projection (solve least-squares to update coefficients)
            gamma = torch.linalg.lstsq(D_I, r).solution
            proj_r = D_I @ gamma

            # Step 5: update residual
            r = r - proj_r

            iter += 1

            if sigma2_est is None:
                SC=False
            else:
                SC=torch.sum(torch.abs(r)**2) <= N * sigma2_est
                
            if  SC or iter > iter_max - 1:
                stop = True
            
            if gamma0 is None: 
                gamma0=gamma
        # Stack all estimations along first dimension
        I=torch.stack(I_list,-1) #([*, nb_active_atoms])
        gamma0=gamma0.squeeze(-1) #([*])
        r=r.squeeze(-1)
        return r,I,gamma0