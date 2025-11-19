#%% importing libraries
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from models.MOMP_model import MOMP_model
from utils.dictionary_gen_utils import *
import matplotlib.pyplot as plt
from saved_data_loader import *
from utils.training_utils import *

#%% functions
def mode_unfold(Y, m):
    # Y is an N-way tensor
    N = Y.ndim
    
    # Create permutation: bring dimension r to the front
    perm = [i for i in range(N) if i != m] + [m] 
    
    # Permute and reshape
    return Y.permute(*perm).reshape(-1, Y.shape[m])

def recover_unfold(Ym, m, shape):
    # Y is an N-way tensor
    N = len(shape)
    
    # Create permutation: bring dimension r to the front
    perm = [i for i in range(N) if i != m] + [m] 
    permuted_shape=[shape[i] for i in perm]
    # Permute and reshape
    return Ym.reshape(*permuted_shape).permute(*perm)

# def recover_shape(Ym):
# # Ym = Y.permute(0, 1, 3, 2).reshape(-1, Y.shape[2]) #([204800, 8])
#     return Ym.reshape(Y.shape[0],Y.shape[1],Y.shape[3],Y.shape[2]).permute(0, 1, 3, 2) # (100, 16, 8, 128)

def OMP(Y, D,iter_max=10):
    '''handles batched perations'''
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

        # if sigma2_est is None:
        #    SC=False
        # else:
        #    SC= torch.sum(torch.abs(r)**2)<=N*sigma2_est # see mpnet paper   
        if iter>iter_max-1:
            stop=True

    # Stack all estimations along first dimension
    I=torch.stack(I_list,-1) #([*, nb_active_atoms])
    gamma=gamma.squeeze(-1) #([*, nb_active_atoms])
    r=r.squeeze(-1)
    return r,I,gamma


#%%--------------------------------------- preprocessing ------------------------------------------------------------
Umax,Pmax=5,100
H=channels[:Umax,:Pmax] #([Umax,Pmax, 16, 8, 128])
Y=observations[:Umax,:Pmax] #temporarily 
nb_users=H.shape[0]
#------------------------------------  normalize channels  ----------------------------------------------------------
H_normalized = H / torch.sqrt(torch.sum(torch.abs(H)**2, dim=(-3, -2, -1), keepdim=True))
Y_normalized = Y / torch.sqrt(torch.sum(torch.abs(Y)**2, dim=(-3, -2, -1), keepdim=True))
#-------------------------------Get train, validation and test data -------------------------------------------------
train_test_ratio=0.8
tt_split_index=int(H_normalized.shape[1] * train_test_ratio)

# test data 
H_test=H_normalized[:,tt_split_index:].to(device)
Y_test=Y_normalized[:,tt_split_index:].to(device)

#each User u has its own Dictionary D_M:
u=0 #u<Umax
H_test_u=H_test[u]
Y_test_u=Y_test[u]
m=2 #dimensions des antennes de la MS #([Pmax, 16, 8, 128])

Hm = mode_unfold(H_test_u,m)
Ym = mode_unfold(Y_test_u,m) #([204 800, 8])

#%% MOD 
batch_size=Ym.shape[0]


# D0_real = torch.rand(nominal_MS_Dictionary.shape,dtype=torch.complex128)
# D0_imag = torch.rand(nominal_MS_Dictionary.shape,dtype=torch.complex128)
# D0 = D0_real + 1j * D0_imag

D0=nominal_MS_Dictionary
epsilon=1e-2
stop=False
iter=0
while not stop: 
    # step 1: sparse recovery
    _,I,gamma=OMP(Ym,D0,iter_max=5)

    Gamma=torch.zeros((batch_size),D0.shape[1],dtype=gamma.dtype)
    batch_idx = torch.arange(batch_size).unsqueeze(-1)
    Gamma[batch_idx, I] = gamma
    #step 2: update dictionary 
    YmT=Ym.T # shape ([N_M, N_obs])
    Gamma=Gamma.T # shape ([A_M, N_obs])

    # Compute expression: D = Y * Gammaᴴ * (Gamma * Gammaᴴ)^(-1)
    # D shape ([N_M, A_M])

    Gamma_H = Gamma.conj().T  # Hermitian (conjugate transpose)
    term = Gamma @ Gamma_H
    term_inv = torch.linalg.inv(term)
    D_MOD = YmT @ Gamma_H @ term_inv
    D_MOD = D_MOD / torch.norm(D_MOD, dim=0, keepdim=True)  # normalize atoms

    if torch.norm(D_MOD-D0)/torch.norm(D0)<epsilon:
        stop=True
    
    iter+=1
    print(f'iteration: {iter}, SC={torch.norm(D_MOD-D0)/torch.norm(D0)}')
    D0=D_MOD


#%%
# LOAD TRAINED MODELS
############################ MOMP ############################
nb_users=1
nb_positions=100
nominal_MS_ant_position_stacked = torch.stack([nominal_MS_ant_position.clone() for _ in range(nb_users)], dim=0)
unfolded_MOMP_model = MOMP_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff, nominal_MS_ant_position_stacked,
                 subcarriers, BS_DoA, MS_DoA, delays)  # replace with your model class
# Load everything
# checkpoint = torch.load('.saved_data/.saved_models/MOMP_model_and_metrics.pth')
checkpoint = torch.load('MOMP_model_and_metrics.pth')
# Load model weights
unfolded_MOMP_model.load_state_dict(checkpoint['model_state_dict'])
learned_MS_pos_y=torch.stack([p.detach() for p in unfolded_MOMP_model.MS_learnable_pos_list], 0).cpu()  # 4th parameter tenso
learned_MS_pos_u=torch.stack([torch.tensor(nominal_MS_ant_position[:,0]), learned_MS_pos_y[u], torch.tensor(nominal_MS_ant_position[:,2])], dim=1)
D_unf=steering_vect_dict(MS_DoA,learned_MS_pos_u,MS_gains,MS_coupling_coeff,lambda_)

#%%
plt.imshow(torch.abs(torch.conj(nominal_MS_Dictionary).T@nominal_MS_Dictionary))
plt.show()
plt.imshow(torch.abs(torch.conj(D_MOD).T@D_MOD))
plt.show()
print(torch.norm(real_MS_Dictionaries[0]-nominal_MS_Dictionary))
print(torch.norm(real_MS_Dictionaries[0]-D_MOD))
print(torch.norm(real_MS_Dictionaries[0]-D_unf))
#%%
r,_,_=OMP(Ym,nominal_MS_Dictionary,iter_max=3)
r_MOD,_,_=OMP(Ym,D_MOD,iter_max=3)
r_real,_,_=OMP(Ym,real_MS_Dictionaries[0],iter_max=3)
r_unf,_,_=OMP(Ym,D_unf,iter_max=3)

# %%

nmse_0=NMSE(H_test_u,Y_test_u)
nmse_1=NMSE(H_test_u,recover_unfold(Ym-r,m,Y_test_u.shape))
nmse_2=NMSE(H_test_u,recover_unfold(Ym-r_MOD,m,Y_test_u.shape))
nmse_3=NMSE(H_test_u,recover_unfold(Ym-r_real,m,Y_test_u.shape))
nmse_02=NMSE(H_test_u,recover_unfold(Ym-r_unf,m,Y_test_u.shape))


# nmse_1=torch.sum(torch.abs(Hm-(Ym-r))**2,dim=(-1))/torch.sum(torch.abs(Hm)**2,dim=(-1))


#%%
# Filter and slice data
idx = torch.where(nmse_0 < 1)
nmse0 = nmse_0
nmse1 = nmse_1
nmse2 = nmse_2
nmse02 = nmse_02
nmse3 = nmse_3

# nmse0 = nmse_0[idx]
# nmse1 = nmse_1[idx]
# nmse2 = nmse_2[idx]
# nmse02 = nmse_02[idx]
# nmse3 = nmse_3[idx]

# Compute means
means = [
    nmse0.mean().item(),
    nmse1.mean().item(),
    nmse2.mean().item(),
    nmse02.mean().item(),
    nmse3.mean().item()
]

# Labels and colors
labels = [
    'Observation error',
    'OMP with nominal Dict',
    'OMP with MOD Dict',
    'OMP with DeepUnfolding Dict',
    'OMP with real Dict'
]
colors = [color_observation, color_nominal, color_MOD, color_MOMP, color_real]
width=0.5
# Plot
plt.figure(figsize=(8, 5))
x = np.arange(len(means))
plt.bar(x, means,width=width, color=colors, alpha=0.7)
plt.yscale('log')
plt.xticks(x, labels, rotation=20)
plt.ylabel('NMSE (mean)')
plt.title('Mean NMSE comparison')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#%% old school
nmse0=nmse_0[idx][:20]
nmse1=nmse_1[idx][:20]
nmse2=nmse_2[idx][:20]
nmse02=nmse_02[idx][:20]
nmse3=nmse_3[idx][:20]
channels_idx = np.arange(1, len(nmse1) + 1)
width = 0.2

plt.figure(figsize=(15, 5))
# Then detach and convert to numpy
bars1 = plt.bar(channels_idx - 3*width/2, nmse0.detach().cpu().numpy(), width,
                label='observation error', color='orange')
bars1 = plt.bar(channels_idx - width/2, nmse1.detach().cpu().numpy(), width,
                label='OMP with nominal Dict', color='red')
bars2 = plt.bar(channels_idx + width/2 , nmse2.detach().cpu().numpy(), width,
                label='OMP with MOD dict', color='skyblue',alpha=0.4)
bars02 = plt.bar(channels_idx + width/2 , nmse02.detach().cpu().numpy(), width,
                label='OMP with DeepUnfolding Dict', color='blue',alpha=0.4)
bars3 = plt.bar(channels_idx + 3*width/2, nmse3.detach().cpu().numpy(), width,
                label='OMP with real Dict', color='green')
plt.semilogy()
plt.xticks(channels_idx)
plt.legend()
plt.show()
# %%
