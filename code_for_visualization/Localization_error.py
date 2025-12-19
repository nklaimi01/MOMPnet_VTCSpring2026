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

def rde(true_delay_us,true_AoA,est_delay_us,est_AoA):
    est_AoA = np.array(est_AoA_list)
    est_delay_us = np.array(est_delay_list)
    true_d=true_delay_us*c*1e-6 #us -> m
    est_d=est_delay_us*c*1e-6
    return np.sqrt(true_d**2+est_d**2-2*true_d*est_d*np.cos(est_AoA-true_AoA))

def ecdf(x):
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

SNR_av=15
print(f'average SNR={SNR_av}')

#%%
#DATA #! dont put train data
Umax=10
total_u=channels.shape[0]
total_upos=channels.shape[1]
H=channels[Umax:,:] #([50-Umax,150, 16, 8, 128])
Y=observations_dict[SNR_av][Umax:,:]  
#------------------------------------  normalize channels  ----------------------------------------------------------
H = normalize(H).to(device)
Y = normalize(Y).to(device)
users_position_test=users_position[Umax:,:].to(device)



#%% LOAD TRAINED MODELS
############################ MOMP ############################
nominal_MS_ant_position_stacked = torch.stack([nominal_MS_ant_position.clone() for _ in range(10)], dim=0)

MOMPnet_trained = MOMP_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff, nominal_MS_ant_position_stacked,
                 subcarriers, BS_DoA, MS_DoA, delays)  # replace with your model class
checkpoint = torch.load(f'MOMPnet_{SNR_av}_dB.pth')
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
est_AoA_list, est_delay_list = [],[]
nom_AoA_list, nom_delay_list = [],[]
real_AoA_list, real_delay_list = [],[]

for u in tqdm(range(total_u-Umax)):
    for upos in range(total_upos):
        
        # i_b,i_s=loc(Y[u,upos],learned_D_B,D_S)
        i_b,i_s=loc(Y[u,upos],real_BS_Dictionary,D_S)
        est_AoA_rd,est_delay_us=[BS_angles[i_b],delays[i_s]*1e6]
        est_AoA_list.append(est_AoA_rd)
        est_delay_list.append(est_delay_us)

        i_b_nom,i_s_nom=loc(Y[u,upos],nominal_BS_Dictionary,D_S)
        nom_AoA_rd,nom_delay_us=[BS_angles[i_b_nom],delays[i_s_nom]*1e6]
        nom_AoA_list.append(nom_AoA_rd)
        nom_delay_list.append(nom_delay_us)

        i_b_real,i_s_real=loc(Y[u,upos],real_BS_Dictionary,D_S)
        real_AoA_rd,real_delay_us=[BS_angles[i_b_real],delays[i_s_real]*1e6]
        real_AoA_list.append(real_AoA_rd)
        real_delay_list.append(real_delay_us)

# %%-------------------------------------------
#localization error heatmap
# ---------------------------------------------

rde_after=rde(true_delay,true_AoA,est_delay_list,est_AoA_list)
rde_before=rde(true_delay,true_AoA,nom_delay_list,nom_AoA_list)
rde_realdict=rde(true_delay,true_AoA,real_delay_list,real_AoA_list)


# User positions (flattened)
x_u = users_position_test[:, :, 0].reshape(-1)
y_u = users_position_test[:, :, 1].reshape(-1)

#%%
# ---------------------------------------------
# relative distance Error Heatmap
# ---------------------------------------------
x1, x2, y1, y2 = -80, 0, -40, 100  # subregion to plot

fig, axes = plt.subplots(1, 2, figsize=(12,5))  # 1 row, 2 columns
vmax=30
cmap='GnBu'
# Main scatter for AoA errors
sc = axes[0].hexbin(x_u, y_u, C=rde_before, cmap=cmap, vmin=0, vmax=vmax,alpha=0.8 )
# BS marker
# axes[0].scatter(BS_position[0], BS_position[1], color='black', s=100, label='BS')

# BS-centered 200 m circle
circle = patches.Circle((BS_position[0], BS_position[1]), 200,
                        edgecolor='black', facecolor='none', linestyle='--', linewidth=2)
axes[0].add_patch(circle)
axes[0].set_title("Localization error before training")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")
axes[0].legend()
axes[0].axis('equal')  # ensure circle is not distorted
axes[0].set_xlim(x1, x2)
axes[0].set_ylim(y1, y2)
#Zoom region inset Axes:


# Main scatter for AoA errors
sc = axes[1].hexbin(x_u, y_u, C=rde_after, cmap=cmap, vmin=0, vmax=vmax,alpha=0.8)


# BS marker
# axes[1].scatter(BS_position[0], BS_position[1], color='black', s=100, label='BS')

# BS-centered 200 m circle
circle = patches.Circle((BS_position[0], BS_position[1]), 200,
                        edgecolor='black', facecolor='none', linestyle='--', linewidth=2)
axes[1].add_patch(circle)

axes[1].set_title("Localization error after training")
axes[1].set_xlabel("x (m)")
axes[1].set_ylabel("y (m)")
axes[1].legend()
axes[1].axis('equal')  # ensure circle is not distorted
axes[1].set_xlim(x1, x2)
axes[1].set_ylim(y1, y2)
fig.colorbar(sc, ax=axes, location='right', fraction=0.046, pad=0.04, label='relative distance error (m)')

plt.show()
#%%
GnRd = LinearSegmentedColormap.from_list(
    "green_red", 
    [
        (0.0, "#2DF11B"),   
        (0.5, "#EFE816"),    
        (1.0, "#F80D0D")        
    ])
WtRd = LinearSegmentedColormap.from_list(
    "green_red", 
    [
        (0.0, "#89F97E"),   
        (0.5, "#F5F179"),    
        (1.0, "#F87A7A")        
    ])
cmap=WtRd
zoom_cmap=GnRd
vmax=60
fig, axes = plt.subplots(1, 2, figsize=(12,5))  # 1 row, 2 columns

# Main scatter for AoA errors
sc = axes[0].hexbin(x_u, y_u, C=rde_before, cmap=cmap, vmin=0, vmax=vmax,alpha=0.8 )
# BS marker
axes[0].scatter(BS_position[0], BS_position[1], color='black', s=100, label='BS')

# BS-centered 200 m circle
circle = patches.Circle((BS_position[0], BS_position[1]), 200,
                        edgecolor='black', facecolor='none', linestyle='--', linewidth=2)
axes[0].add_patch(circle)

axes[0].set_title("Localization error Heatmap before training")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")
axes[0].legend()
axes[0].axis('equal')  # ensure circle is not distorted
#Zoom region inset Axes:
x1, x2, y1, y2 = -100, 50, -50, 100  # subregion of the original image
axins0 = axes[0].inset_axes(
    [0.5, 0.5, 0.47, 0.47],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
axins0.hexbin(x_u, y_u, C=rde_before, cmap=zoom_cmap, vmin=0, vmax=vmax,alpha=0.8)
axes[0].indicate_inset_zoom(axins0, edgecolor="black")

# Main scatter for AoA errors
sc = axes[1].hexbin(x_u, y_u, C=rde_after, cmap=cmap, vmin=0, vmax=vmax,alpha=0.8)

# BS marker
axes[1].scatter(BS_position[0], BS_position[1], color='black', s=100, label='BS')

# BS-centered 200 m circle
circle = patches.Circle((BS_position[0], BS_position[1]), 200,
                        edgecolor='black', facecolor='none', linestyle='--', linewidth=2)
axes[1].add_patch(circle)

axes[1].set_title("Localization error Heatmap after training")
axes[1].set_xlabel("x (m)")
axes[1].set_ylabel("y (m)")
axes[1].legend()
axes[1].axis('equal')  # ensure circle is not distorted
#Zoom region inset Axes:
x1, x2, y1, y2 = -100, 50, -50, 100  # subregion of the original image
axins1 = axes[1].inset_axes(
    [0.5, 0.5, 0.47, 0.47],
    xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
axins1.hexbin(x_u, y_u, C=rde_after, cmap=zoom_cmap, vmin=0, vmax=vmax,alpha=0.8)
axes[1].indicate_inset_zoom(axins1, edgecolor="black")
fig.colorbar(sc, ax=axes, location='right', fraction=0.046, pad=0.04,label='relative distance error (m)')

plt.show()


#%%
x_b, y_b = ecdf(rde_before)
x_a, y_a = ecdf(rde_after)

plt.figure(figsize=(6,4))
plt.step(x_b, y_b, where='post', label='Before training')
plt.step(x_a, y_a, where='post', label='After training')

plt.xlabel('Localization error (m)')
plt.ylabel('CDF')
plt.title('CDF of Localization Error Before and After Training')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


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
