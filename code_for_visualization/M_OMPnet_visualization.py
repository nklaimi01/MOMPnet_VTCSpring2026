#%%
import torch
from pathlib import Path
import os,sys
import matplotlib.pyplot as plt
# Get the absolute path to this script
current_dir = Path(__file__).resolve()
# Traverse up until we reach the 'MOMP' folder
for parent in current_dir.parents:
    if parent.name == "MOMP":
        project_root = parent
        break
else:
    raise RuntimeError("Couldn't find 'MOMP' folder in path hierarchy.")
# Set working directory to MOMP
os.chdir(project_root)
# Add project root to sys.path so imports work everywhere
sys.path.append(str(project_root))
from models.MOMP_model import MOMP_model
from models.OMP_model import OMP_model
from saved_data_loader import *
from utils.training_utils import *

SNR_average=10*torch.log10(torch.mean(torch.sum(torch.abs(channels)**2, axis=(2, 3, 4))) / (16*8*128 * sigma2))
print(f'average SNR={SNR_average}')


# LOAD TRAINED MODELS
############################ MOMP ############################
nb_users=10
nb_positions=10
nominal_MS_ant_position_stacked = torch.stack([nominal_MS_ant_position.clone() for _ in range(nb_users)], dim=0)
unfolded_MOMP_model = MOMP_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff, nominal_MS_ant_position_stacked,
                 subcarriers, BS_DoA, MS_DoA, delays)  # replace with your model class
# Load everything
checkpoint = torch.load('.saved_data/.saved_models/MOMP_model_and_metrics.pth')
# Load model weights
unfolded_MOMP_model.load_state_dict(checkpoint['model_state_dict'])
# Load the lists
NMSE0 = checkpoint['NMSE0']
NMSEZ = checkpoint['NMSEZ']
train_losses_list = checkpoint['train_losses']
valid_losses_list = checkpoint['valid_losses']

# #%%
# learned_BS_pos_OMP=list(unfolded_OMP_model.parameters())[0].detach().numpy()  # first parameter tensor
# learned_BS_pos_MOMP=list(unfolded_MOMP_model.parameters())[0].detach().numpy()  # first parameter tensor

# # --- Calcul des erreurs quadratiques ---
# err_nominal = torch.sum(torch.abs(real_BS_ant_position[:, 1] - nominal_BS_ant_position[:, 1])**2)
# err_learned_OMP = torch.sum(torch.abs(real_BS_ant_position[:, 1] - torch.tensor(learned_BS_pos_OMP))**2)
# err_learned_MOMP = torch.sum(torch.abs(real_BS_ant_position[:, 1] - torch.tensor(learned_BS_pos_MOMP))**2)

# # --- Affichage des valeurs ---
# print(f"‖P_real - P_nominal‖²₂ = {err_nominal.item():.4e}")
# print(f"‖P_real - P_learnedOMP‖²₂ = {err_learned_OMP.item():.4e}")
# print(f"‖P_real - P_learnedMOMP‖²₂ = {err_learned_MOMP.item():.4e}")

#%% localization error
learned_BS_pos_y=list(unfolded_MOMP_model.parameters())[0].detach()  # first parameter tensor
learned_gains=list(unfolded_MOMP_model.parameters())[1].detach() # 2nd parameter tensor
learned_coupling=list(unfolded_MOMP_model.parameters())[2].detach()  # 3rd parameter tensor
learned_MS_pos_y=torch.stack([p.detach() for p in unfolded_MOMP_model.MS_learnable_pos_list], 0).cpu()  # 4th parameter tensor

learned_BS_pos=torch.stack([torch.tensor(nominal_BS_ant_position[:,0]), learned_BS_pos_y, torch.tensor(nominal_BS_ant_position[:,2])], dim=1)
learned_D_B=steering_vect_dict(BS_DoA,learned_BS_pos,learned_gains,learned_coupling,lambda_)
D_S=FRV_Dictionary


#%%
est_AoA_list,est_delay_list=[],[]
for u in range(nb_users):
    learned_MS_pos_u=torch.stack([torch.tensor(nominal_MS_ant_position[:,0]), learned_MS_pos_y[u], torch.tensor(nominal_MS_ant_position[:,2])], dim=1)
    D_M=steering_vect_dict(MS_DoA,learned_MS_pos_u,MS_gains,MS_coupling_coeff,lambda_)
    for upos in range(nb_positions):
        r,I,x=unfolded_MOMP_model.forward(observations[u,upos],u,sigma2)

        i_b,i_m,i_s=I[0]
        est_AoA_rd,a2,est_delay_us=[BS_angles[i_b],MS_angles[i_m],delays[i_s]*1e6]
        est_AoA_list.append(est_AoA_rd)
        est_delay_list.append(est_delay_us)
        dx, dy, dz = users_position[u,upos]-torch.tensor(BS_position)
        user_AoA_rd = np.pi - np.abs(np.arctan2(dx, dy))  
        user_delay_us = np.sqrt(dx**2 + dy**2 + dz**2) / 3e8 * 1e6
        user_AoAcos = np.cos(user_AoA_rd)
        print(f'user {u}, position {upos}')
        print( '-------------------------')
        print(f'estimated AoA: {est_AoA_rd:.2F} rd' )
        print(f'true user AoA: {user_AoA_rd:.2F} rd')
        print('\n')
        print(f'estimated delay: {est_delay_us:.2F} μs')
        print(f'true user delay: {user_delay_us:.2F} μs')
        print('==========================')

# displacement = users_position[:nb_users, :nb_positions] - BS_position  # shape [nb_users, nb_positions, 3]

# %%
