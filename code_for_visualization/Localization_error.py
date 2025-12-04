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

def loc(Y,D1,D2,D3=None):
    # Step 1: Compute correlations with D1 along the first dimension
    corr1 = torch.einsum('ab,bms->ams', torch.conj(D1).T, Y)
    i1 = torch.argmax((corr1.abs()**2).sum(dim=(1, 2)))
    
    # Step 2: Select best atom from D2 using the previous selection from D1
    corr2 = torch.conj(D2).T @ corr1[i1].T
    i2 = torch.argmax((corr2.abs()**2).sum(dim=1))
    # # Step 3: Select best atom from D3 using the previous selections
    # corr3 = torch.conj(D3).T @ corr2[i2]
    # i3 = torch.argmax(torch.abs(corr3)**2)
    return i1,i2

SNR_average=10*torch.log10(torch.mean(torch.sum(torch.abs(channels)**2, axis=(2, 3, 4))) / (16*8*128 * sigma2))
print(f'average SNR={SNR_average}')

#%%
#DATA #! dont put train data
Umax,Pmax=100,100
H=channels[5:Umax,:Pmax] #([Umax,Pmax, 16, 8, 128])
Y=observations[5:Umax,:Pmax]  
#------------------------------------  normalize channels  ----------------------------------------------------------
H = normalize(H).to(device)
Y = normalize(Y).to(device)
users_position_test=users_position[5:Umax,:].to(device)



#%% LOAD TRAINED MODELS
############################ MOMP ############################
nominal_MS_ant_position_stacked = torch.stack([nominal_MS_ant_position.clone() for _ in range(5)], dim=0)

MOMPnet_trained = MOMP_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff, nominal_MS_ant_position_stacked,
                 subcarriers, BS_DoA, MS_DoA, delays)  # replace with your model class
checkpoint = torch.load('.saved_data/.saved_models/MOMP_5100.pth')
# checkpoint = torch.load('MOMP_model_and_metrics.pth')
# Load model weights
MOMPnet_trained.load_state_dict(checkpoint['model_state_dict'])
learned_BS_pos_y=list(MOMPnet_trained.parameters())[0].detach()  # first parameter tensor
learned_gains=list(MOMPnet_trained.parameters())[1].detach() # 2nd parameter tensor
learned_coupling=list(MOMPnet_trained.parameters())[2].detach()  # 3rd parameter tensor
learned_MS_pos_y=torch.stack([p.detach() for p in MOMPnet_trained.MS_learnable_pos_list], 0).cpu()  # 4th parameter tensor
learned_BS_pos=torch.stack([torch.tensor(nominal_BS_ant_position[:,0]), learned_BS_pos_y, torch.tensor(nominal_BS_ant_position[:,2])], dim=1)
learned_D_B=steering_vect_dict(BS_DoA,learned_BS_pos,learned_gains,learned_coupling,lambda_)
D_S=torch.tensor(FRV_Dictionary,dtype=Y.dtype)

#%% localization error
# True users AoA and delay from geometric positions: 
delta=users_position_test-torch.tensor(BS_position)
dx, dy, dz = delta[..., 0], delta[..., 1], delta[..., 2]

user_AoA_rd = np.pi - torch.abs(torch.atan2(dx, dy))   # shape: (len(users), len(user_positions))
user_AoAcos = torch.cos(user_AoA_rd)

dist = torch.sqrt(dx**2 + dy**2 + dz**2)
user_delay_us = dist / 3e8 * 1e6

true_AoA = user_AoA_rd.flatten().numpy()
true_delay = user_delay_us.flatten().numpy()

# estimated AoA and delays before and after training:
est_AoA_list = []
est_delay_list = []
nom_AoA_list = []
nom_delay_list = []
for u in tqdm(range(Umax-5)):
    for upos in range(Pmax):
        
        # i_b,i_s=loc(Y[u,upos],learned_D_B,D_S)
        i_b,i_s=loc(Y[u,upos],real_BS_Dictionary,D_S)
        est_AoA_rd,est_delay_us=[BS_angles[i_b],delays[i_s]*1e6]
        est_AoA_list.append(est_AoA_rd)
        est_delay_list.append(est_delay_us)

        i_b_nom,i_s_nom=loc(Y[u,upos],nominal_BS_Dictionary,D_S)
        nom_AoA_rd,nom_delay_us=[BS_angles[i_b_nom],delays[i_s_nom]*1e6]
        nom_AoA_list.append(nom_AoA_rd)
        nom_delay_list.append(nom_delay_us)


# %% localization error heatmap
# ---------------------------------------------
# Convert lists to arrays
# ---------------------------------------------
est_AoA = np.array(est_AoA_list)
est_delay = np.array(est_delay_list)
nom_AoA = np.array(nom_AoA_list)
nom_delay = np.array(nom_delay_list)

true_d=true_delay*c*1e-6 #us -> m
est_d=est_delay*c*1e-6
nom_d=nom_delay*c*1e-6
rde=np.sqrt(true_d**2+est_d**2-2*true_d*est_d*np.cos(est_AoA-true_AoA))
rde_nominal=np.sqrt(true_d**2+nom_d**2-2*true_d*nom_d*np.cos(nom_AoA-true_AoA))

# User positions (flattened)
x_u = users_position_test[:, :, 0].reshape(-1)
y_u = users_position_test[:, :, 1].reshape(-1)

#%%
# ---------------------------------------------
# relative distance Error Heatmap
# ---------------------------------------------

z=(rde_nominal-rde).nonzero()[0]

green_red_cmap = LinearSegmentedColormap.from_list(
    "green_red", 
    [
        (0.0, "#39FF14"),   
        (0.5, "orange"),    
        (1.0, "red")        
    ]
)
fig, ax = plt.subplots(figsize=(6,5))
# Main scatter for AoA errors
sc = ax.hexbin(x_u, y_u, C=rde_nominal, cmap=green_red_cmap, vmin=0, vmax=20,alpha=0.8 )
plt.colorbar(sc, label="relative distance Error (m)")

# BS marker
ax.scatter(BS_position[0], BS_position[1], color='black', s=100, label='BS')

# BS-centered 200 m circle
circle = patches.Circle((BS_position[0], BS_position[1]), 200,
                        edgecolor='black', facecolor='none', linestyle='--', linewidth=2)
ax.add_patch(circle)

ax.set_title("Localization error Heatmap before training")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.legend()
ax.axis('equal')  # ensure circle is not distorted

fig, ax = plt.subplots(figsize=(6,5))
# Main scatter for AoA errors
sc = ax.hexbin(x_u, y_u, C=rde, cmap=green_red_cmap, vmin=0, vmax=20,alpha=0.8)
plt.colorbar(sc, label="relative distance Error (m)")

# BS marker
ax.scatter(BS_position[0], BS_position[1], color='black', s=100, label='BS')

# BS-centered 200 m circle
circle = patches.Circle((BS_position[0], BS_position[1]), 200,
                        edgecolor='black', facecolor='none', linestyle='--', linewidth=2)
ax.add_patch(circle)

ax.set_title("Localization error Heatmap after training")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.legend()
ax.axis('equal')  # ensure circle is not distorted

#%%
# # Compute errors
# AoA_errors = np.abs(true_AoA - est_AoA)
# delay_errors = np.abs(true_delay - est_delay)
# AoA_lim = max(np.abs(AoA_errors.min()), AoA_errors.max())
# delay_lim = max(np.abs(delay_errors.min()), delay_errors.max())
# # ---------------------------------------------
# # AoA Error Heatmap
# # ---------------------------------------------
# fig, ax = plt.subplots(figsize=(6,5))

# # Main scatter for AoA errors
# sc = ax.scatter(x_u, y_u, c=AoA_errors, cmap=green_red_cmap, vmin=0, vmax=AoA_errors.max())
# plt.colorbar(sc, label="AoA Error (rad)")

# # BS marker
# ax.scatter(BS_position[0], BS_position[1], color='black', s=100, label='BS')

# # BS-centered 200 m circle
# circle = patches.Circle((BS_position[0], BS_position[1]), 200,
#                         edgecolor='black', facecolor='none', linestyle='--', linewidth=2)
# ax.add_patch(circle)

# ax.set_title("AoA Error Heatmap over User Positions")
# ax.set_xlabel("x (m)")
# ax.set_ylabel("y (m)")
# ax.legend()
# ax.axis('equal')  # ensure circle is not distorted

# # ---------------------------------------------
# # Delay Error Heatmap
# # ---------------------------------------------
# fig, ax = plt.subplots(figsize=(6,5))

# # Main scatter for delay errors
# sc = ax.scatter(x_u, y_u, c=delay_errors, cmap=green_red_cmap, vmin=0, vmax=delay_errors.max())
# plt.colorbar(sc, label="Delay Error (µs)")

# # BS marker
# ax.scatter(BS_position[0], BS_position[1], color='black', s=100, label='BS')

# # BS-centered 200 m circle
# circle = patches.Circle((BS_position[0], BS_position[1]), 200,
#                         edgecolor='black', facecolor='none', linestyle='--', linewidth=2)
# ax.add_patch(circle)

# ax.set_title("Delay Error Heatmap over User Positions")
# ax.set_xlabel("x (m)")
# ax.set_ylabel("y (m)")
# ax.legend()
# ax.axis('equal')

# plt.show()

# %%
