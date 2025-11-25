#%%
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import sys,os
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
print(f"Working directory set to: {Path.cwd()}")
# Add project root to sys.path so imports work everywhere
sys.path.append(str(project_root))
from models.MOMP_model import MOMP_model
from saved_data_loader import *
from utils.training_utils import *
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


SNR_average=10*torch.log10(torch.mean(torch.sum(torch.abs(channels)**2, axis=(2, 3, 4))) / (16*8*128 * sigma2))
print(f'average SNR={SNR_average}')

#DATA
Umax,Pmax=5,100
H=channels[:Umax,:Pmax] #([Umax,Pmax, 16, 8, 128])
Y=observations[:Umax,:Pmax] #temporarily 
#------------------------------------  normalize channels  ----------------------------------------------------------
H_normalized = H / torch.sqrt(torch.sum(torch.abs(H)**2, dim=(-3, -2, -1), keepdim=True))
Y_normalized = Y / torch.sqrt(torch.sum(torch.abs(Y)**2, dim=(-3, -2, -1), keepdim=True))
#-------------------------------Get train, validation and test data -------------------------------------------------
train_test_ratio=0.8
tt_split_index=int(H_normalized.shape[1] * train_test_ratio)

# test data 
H_test=H_normalized[:,tt_split_index:].to(device)
Y_test=Y_normalized[:,tt_split_index:].to(device)
users_position_test=users_position[:Umax,tt_split_index:].to(device)

if Umax>1:
    nb_test_positions=Y_test.shape[1]
else:
    nb_test_positions=Pmax

# LOAD TRAINED MODELS
############################ MOMP ############################
nominal_MS_ant_position_stacked = torch.stack([nominal_MS_ant_position.clone() for _ in range(Umax)], dim=0)
unfolded_MOMP_model = MOMP_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff, nominal_MS_ant_position_stacked,
                 subcarriers, BS_DoA, MS_DoA, delays)  # replace with your model class

checkpoint = torch.load('.saved_data/.saved_models/MOMP_5100.pth')
# checkpoint = torch.load('MOMP_model_and_metrics.pth')
# Load model weights
unfolded_MOMP_model.load_state_dict(checkpoint['model_state_dict'])

#%% localization error
learned_BS_pos_y=list(unfolded_MOMP_model.parameters())[0].detach()  # first parameter tensor
learned_gains=list(unfolded_MOMP_model.parameters())[1].detach() # 2nd parameter tensor
learned_coupling=list(unfolded_MOMP_model.parameters())[2].detach()  # 3rd parameter tensor
learned_MS_pos_y=torch.stack([p.detach() for p in unfolded_MOMP_model.MS_learnable_pos_list], 0).cpu()  # 4th parameter tensor

learned_BS_pos=torch.stack([torch.tensor(nominal_BS_ant_position[:,0]), learned_BS_pos_y, torch.tensor(nominal_BS_ant_position[:,2])], dim=1)
learned_D_B=steering_vect_dict(BS_DoA,learned_BS_pos,learned_gains,learned_coupling,lambda_)
D_S=FRV_Dictionary


#%%
true_AoA_list = []
true_delay_list = []
est_AoA_list = []
est_delay_list = []
for u in tqdm(range(Umax)):
    learned_MS_pos_u=torch.stack([torch.tensor(nominal_MS_ant_position[:,0]), learned_MS_pos_y[u], torch.tensor(nominal_MS_ant_position[:,2])], dim=1)
    D_M=steering_vect_dict(MS_DoA,learned_MS_pos_u,MS_gains,MS_coupling_coeff,lambda_)
    for upos in range(nb_test_positions):
        r,I,x=unfolded_MOMP_model.forward(Y_test[u,upos],u,sigma2)

        i_b,i_m,i_s=I[0]
        est_AoA_rd,a2,est_delay_us=[BS_angles[i_b],MS_angles[i_m],delays[i_s]*1e6]
        est_AoA_list.append(est_AoA_rd)
        est_delay_list.append(est_delay_us)
        dx, dy, dz = users_position_test[u,upos]-torch.tensor(BS_position)
        user_AoA_rd = np.pi - np.abs(np.arctan2(dx, dy))  
        user_delay_us = np.sqrt(dx**2 + dy**2 + dz**2) / 3e8 * 1e6
        user_AoAcos = np.cos(user_AoA_rd)
        true_AoA_list.append(user_AoA_rd)
        true_delay_list.append(user_delay_us)

        # print(f'user {u}, position {upos}')
        # print( '-------------------------')
        # print(f'estimated AoA: {est_AoA_rd:.2F} rd' )
        # print(f'true user AoA: {user_AoA_rd:.2F} rd')
        # print('\n')
        # print(f'estimated delay: {est_delay_us:.2F} μs')
        # print(f'true user delay: {user_delay_us:.2F} μs')
        # print('==========================')


# %% localization error heatmap
# ---------------------------------------------
# Convert lists to arrays
# ---------------------------------------------
true_AoA = np.array(true_AoA_list)
est_AoA = np.array(est_AoA_list)
true_delay = np.array(true_delay_list)
est_delay = np.array(est_delay_list)


true_d=true_delay*c*1e-6 #us -> m
est_d=est_delay*c*1e-6
relative_distance_error=np.sqrt(true_d**2+est_d**2-2*true_d*est_d*np.cos(est_AoA-true_AoA))

# User positions (flattened)
x_u = users_position_test[:, :, 0].reshape(-1)
y_u = users_position_test[:, :, 1].reshape(-1)

#%%
# ---------------------------------------------
# relative distance Error Heatmap
# ---------------------------------------------
fig, ax = plt.subplots(figsize=(6,5))

green_red_cmap = LinearSegmentedColormap.from_list(
    "green_red", 
    [
        (0.0, "#39FF14"),   
        (0.5, "orange"),    
        (1.0, "red")        
    ]
)
# Main scatter for AoA errors
sc = ax.scatter(x_u, y_u, c=relative_distance_error, cmap=green_red_cmap, vmin=0, vmax=100)
plt.colorbar(sc, label="relative distance Error (m)")

# BS marker
ax.scatter(BS_position[0], BS_position[1], color='black', s=100, label='BS')

# BS-centered 200 m circle
circle = patches.Circle((BS_position[0], BS_position[1]), 200,
                        edgecolor='black', facecolor='none', linestyle='--', linewidth=2)
ax.add_patch(circle)

ax.set_title("Localization error Heatmap over User Positions")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.legend()
ax.axis('equal')  # ensure circle is not distorted


#%%
# Compute errors
AoA_errors = np.abs(true_AoA - est_AoA)
delay_errors = np.abs(true_delay - est_delay)
AoA_lim = max(np.abs(AoA_errors.min()), AoA_errors.max())
delay_lim = max(np.abs(delay_errors.min()), delay_errors.max())
# ---------------------------------------------
# AoA Error Heatmap
# ---------------------------------------------
fig, ax = plt.subplots(figsize=(6,5))

green_red_cmap = LinearSegmentedColormap.from_list(
    "green_red", 
    [
        (0.0, "#39FF14"),   
        (0.5, "orange"),    
        (1.0, "red")        
    ]
)
# Main scatter for AoA errors
sc = ax.scatter(x_u, y_u, c=AoA_errors, cmap=green_red_cmap, vmin=0, vmax=AoA_errors.max())
plt.colorbar(sc, label="AoA Error (rad)")

# BS marker
ax.scatter(BS_position[0], BS_position[1], color='black', s=100, label='BS')

# BS-centered 200 m circle
circle = patches.Circle((BS_position[0], BS_position[1]), 200,
                        edgecolor='black', facecolor='none', linestyle='--', linewidth=2)
ax.add_patch(circle)

ax.set_title("AoA Error Heatmap over User Positions")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.legend()
ax.axis('equal')  # ensure circle is not distorted

# ---------------------------------------------
# Delay Error Heatmap
# ---------------------------------------------
fig, ax = plt.subplots(figsize=(6,5))

# Main scatter for delay errors
sc = ax.scatter(x_u, y_u, c=delay_errors, cmap=green_red_cmap, vmin=0, vmax=delay_errors.max())
plt.colorbar(sc, label="Delay Error (µs)")

# BS marker
ax.scatter(BS_position[0], BS_position[1], color='black', s=100, label='BS')

# BS-centered 200 m circle
circle = patches.Circle((BS_position[0], BS_position[1]), 200,
                        edgecolor='black', facecolor='none', linestyle='--', linewidth=2)
ax.add_patch(circle)

ax.set_title("Delay Error Heatmap over User Positions")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.legend()
ax.axis('equal')

plt.show()

# %%
