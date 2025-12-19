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
from matplotlib.ticker import MultipleLocator
#%%
def ticklabels_array(highest_int, spacing):
    result = []
    for i in range(highest_int + 1):
        result.append(str(i))
        if i < highest_int:
            result.extend([""] * (spacing-1))
    return result

#%%--------------------------------------- preprocessing ------------------------------------------------------------
Umax=10
Pmax=150
SNR_av=15
sigma2=sigma2_dict[SNR_av]
H=channels[:Umax,:Pmax] #dataset size
Y=observations_dict[SNR_av][:Umax,:Pmax] #dataset size
nb_users=H.shape[0]
#------------------------------------  normalize channels  ----------------------------------------------------------
H_normalized = normalize(H)
Y_normalized = normalize(Y)
#-------------------------------Get train and validation data -------------------------------------------------
train_val_ratio=2/3
tt_split_index=int(H_normalized.shape[1] * train_val_ratio)
H_train=H_normalized[:,:tt_split_index].to(device)
Y_train=Y_normalized[:,:tt_split_index].to(device)

# Validation data 
H_val=H_normalized[:,tt_split_index:].to(device)
Y_val=Y_normalized[:,tt_split_index:].to(device)

# #%%############################### MOMP with real dictionary ################################################################
# H_val_realdict=MOMP_estimation(Y_val,real_BS_Dictionary,real_MS_Dictionaries,FRV_Dictionary,sigma2)
# NMSE_real=NMSE(H_val,H_val_realdict)

# H_val_nominaldict=MOMP_estimation(Y_val,nominal_BS_Dictionary,nominal_MS_Dictionary,FRV_Dictionary,sigma2)
# NMSE_nominal=NMSE(H_val,H_val_nominaldict)

#%%
# LOAD TRAINED MODELS
############################ MOMP ############################
nominal_MS_ant_position_stacked = torch.stack([nominal_MS_ant_position.clone() for _ in range(Umax)], dim=0)
MOMPnet = MOMP_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff, nominal_MS_ant_position_stacked,
                 subcarriers, BS_DoA, MS_DoA, delays)  # replace with your model class
# Load everything
# checkpoint = torch.load(f'MOMPnet_{Umax}_{Pmax}.pth')
checkpoint = torch.load(f'MOMPnet_{SNR_av}_dB.pth')
# Load model weights
MOMPnet.load_state_dict(checkpoint['model_state_dict'])
# Load the lists
train_NMSE_list = checkpoint['train_NMSE']
MOMPnet_NMSE_list = checkpoint['MOMPnet_NMSE']
NMSE_nominal = checkpoint['nominal_NMSE']
NMSE_real = checkpoint['real_NMSE']



#%%#############################################################################################################################################################################
##################################################################  plot evaluation ############################################################################################
################################################################################################################################################################################
#--------------------------------------------------Plotting learning curve------------------------------------------------------------------
# Convert list of tensors -> average NMSE per epoch
train_NMSE_avg = [t.mean().item() for t in train_NMSE_list]
MOMPnet_NMSE_avg = [v.mean().item() for v in MOMPnet_NMSE_list]
nb_epochs=int((len(train_NMSE_avg)-1)/nb_users)
epochs = range(0, len(train_NMSE_avg))

plt.figure(figsize=(8, 5))
P=5
plt.plot(epochs, train_NMSE_avg, label='Train NMSE', marker='o', color='blue',markevery=P)
plt.plot(epochs, MOMPnet_NMSE_avg, label='Validation NMSE', marker='s', color='orange',markevery=P)

plt.gca().spines['left'].set_position('zero')
plt.xlabel('Epoch')
plt.xticks(ticks=epochs,labels=ticklabels_array(nb_epochs,nb_users))
plt.ylabel('NMSE')
plt.title('Learning Curve')
plt.gca().xaxis.set_major_locator(MultipleLocator(5))
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
#%%--------------------------------------------------Plotting validation NMSE------------------------------------------------------------------
# Labels and colors
labels = [
    'MOMP with nominal Dicts',
    'MOMPnet',
    'MOMP with real Dicts'
]
colors = [color_nominal, color_MOMP, color_real]
markers = ['o','^', 'x']
linestyles=['--','-','--']
nmse_obs=NMSE(H_val,Y_val).mean().item()
nmse_nominal = NMSE_nominal.mean().item()
nmse_MOMPnet = MOMPnet_NMSE_avg[-1]
nmse_real = NMSE_real.mean().item()



#plot throughout training 
MOMPnet_NMSE_arr=np.array(MOMPnet_NMSE_avg)
means_arr=np.vstack([np.full_like(MOMPnet_NMSE_arr, nmse_nominal), MOMPnet_NMSE_arr,np.full_like(MOMPnet_NMSE_arr, nmse_real)])



P=10
fig=plt.figure(figsize=(8, 5))

# means_arr must have shape (5, len(nb_obs_list))
# one row per method

for i in range(len(labels)):
    plt.plot(epochs,
             means_arr[i],
             color=colors[i],
             label=labels[i],
             marker=markers[i], linestyle=linestyles[i],markevery=P)

# plt.yscale('log')
ticks = epochs
plt.xticks(ticks=ticks, labels=ticklabels_array(nb_epochs,nb_users))
plt.xlabel('epochs')
plt.xlim(left=0)
plt.ylabel('NMSE (mean)')
# plt.title('Mean NMSE vs number of seen channels')
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.ylim([0.14,0.5])
fig.savefig(f"MOMPnet_{SNR_av}.pdf")
plt.show()
#%%
# bar plot for last value
labels = [
    'Observation',
    'MOMP with nominal Dicts',
    'MOMPnet',
    'MOMP with real Dicts'
]
colors = [color_observation, color_nominal, color_MOMP, color_real]
# Compute means
means = [
    nmse_obs,
    nmse_nominal,
    nmse_MOMPnet,
    nmse_real
]
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
#%%#############################################################################################################################################################################
##################################################################  learned parameters #########################################################################################
################################################################################################################################################################################
learned_BS_pos=list(MOMPnet.parameters())[0].detach().numpy()  # first parameter tensor
learned_gains=list(MOMPnet.parameters())[1].detach().numpy()  # 2nd parameter tensor
learned_coupling=list(MOMPnet.parameters())[2].detach().numpy()  # 3rd parameter tensor
learned_MS_pos=torch.stack([p.detach() for p in MOMPnet.MS_learnable_pos_list], 0).cpu().numpy()  # 4th parameter tensor
nominal_BS_gains = np.asarray(BS_gains['nominal_BS_gains'])
nominal_BS_coupling_coeff = np.asarray(BS_coupling['nominal_BS_coupling_coeff'],dtype=np.complex128)
real_BS_ant_position = np.asarray(real_BS_ant_position)
nominal_BS_ant_position = np.asarray(nominal_BS_ant_position)
real_MS_ant_position = np.asarray(real_MS_ant_position)
nominal_MS_ant_position = np.asarray(nominal_MS_ant_position)
real_BS_gains = np.asarray(real_BS_gains)

real_BS_gains_normalized = real_BS_gains / np.sqrt(np.sum((np.abs(real_BS_gains)**2)))
nominal_BS_gains_normalized = nominal_BS_gains / np.sqrt(np.sum((np.abs(nominal_BS_gains)**2)))
learned_gains_normalized= learned_gains / np.sqrt(np.sum((np.abs(learned_gains)**2)))
#%%
####################################### plot learned BS postions #############################################

# # --- X and Y coordinates ---
# x = real_BS_ant_position[:, 0]
# y_nominal = nominal_BS_ant_position[:, 1]
# y_real = real_BS_ant_position[:, 1]
# y_MOMP = learned_BS_pos

# # --- Apply small horizontal offsets for visibility ---
# offset = 0.03  # adjust if antennas are close
# x_nominal = x - offset *1.5
# x_real    = x - offset * 0.5
# x_MOMP     = x + offset *0.5

# # --- Plot ---
# plt.figure(figsize=(6,5))

# plt.scatter(x_nominal, y_nominal, label='Nominal BS', marker='x', color=color_nominal, s=50, linewidths=1)
# plt.scatter(x_real, y_real, label='Real BS', color=color_real_BS, s=70, edgecolors='k', alpha=0.8)
# plt.scatter(x_MOMP, y_MOMP, label='Learned BS (MOMP)', marker='d',color=color_MOMP, s=70, edgecolors='k', alpha=0.8)

# # --- Optional: connect each antenna index with dotted lines ---
# for i in range(len(x)):
#     plt.plot([x_nominal[i], x_real[i], x_MOMP[i]],
#              [y_nominal[i], y_real[i], y_MOMP[i]],
#              color='gray', linestyle='--', alpha=0.4, linewidth=1)

# # --- Labels and style ---
# plt.title('Mobile Station Antenna Positions with real gains and mutual coupling', fontsize=14)
# plt.xlabel('X-axis [m]')
# plt.ylabel('Y-axis [m]')
# plt.legend(loc='best')
# plt.grid(True, linestyle='--', alpha=0.4)
# plt.tight_layout()
# plt.show()
#%%
############################################ plot learned BS antenna Gains ####################################
# --- Prepare data for plotting ---

# idx = np.arange(len(real_BS_gains_normalized))
# mag_real = np.abs(real_BS_gains_normalized)
# mag_nominal = np.abs(nominal_BS_gains_normalized)
# mag_MOMP = np.abs(learned_gains_normalized)

# phase_real = np.angle(real_BS_gains_normalized)
# phase_nominal = np.angle(nominal_BS_gains_normalized)
# phase_MOMP = np.angle(learned_gains_normalized)

# # --- Plot magnitude comparison ---
# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1)
# plt.plot(idx, mag_real, 'o-', label='Real', color=color_real_BS)
# plt.plot(idx, mag_nominal, 'x--', label='Nominal', color=color_nominal)
# plt.plot(idx, mag_MOMP, 'd-', label='MOMP Learned', color=color_MOMP)
# plt.title('Antenna Gain Magnitudes')
# plt.xlabel('Antenna Index')
# plt.ylabel('|Gain|')
# plt.legend()
# plt.grid(True)

# # --- Plot phase comparison ---
# plt.subplot(1,2,2)
# plt.plot(idx, phase_real, 'o-', label='Real', color=color_real_BS)
# plt.plot(idx, phase_nominal, 'x--', label='Nominal', color=color_nominal)
# plt.plot(idx, phase_MOMP, 'd-', label='MOMP Learned', color=color_MOMP)
# plt.title('Antenna Gain Phases')
# plt.xlabel('Antenna Index')
# plt.ylabel('Phase [rad]')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()
#%%
####################################### plot learned MS postions #############################################
# --- X and Y coordinates ---
for u in range(Umax):
    x = real_MS_ant_position[0, :, 0]
    y_nominal = nominal_MS_ant_position[:, 1]
    y_real = real_MS_ant_position[u, :, 1]
    y_MOMP = learned_MS_pos[u]

    # --- Apply small horizontal offsets for visibility ---
    offset = 0.02  # adjust if antennas are close
    x_nominal = x - offset * 1.5
    x_real    = x - offset * 0.5
    x_MOMP    = x + offset * 0.5

    # --- Plot ---
    plt.figure(figsize=(8,5))

    plt.scatter(x_nominal, y_nominal, label='Nominal MS ', marker='x', color=color_nominal, s=50, linewidths=1)
    plt.scatter(x_real, y_real, label='Real MS', color=color_real_MS, s=70, edgecolors='k', alpha=0.8)
    plt.scatter(x_MOMP, y_MOMP, label='Learned MS (MOMP)', marker='d',color=color_MOMP, s=70, edgecolors='k', alpha=0.8)

    # --- Optional: connect each antenna index with dotted lines ---
    for i in range(len(x)):
        plt.plot([x_nominal[i], x_real[i], x_MOMP[i]],
                [y_nominal[i], y_real[i], y_MOMP[i]],
                color='gray', linestyle='--', alpha=0.4, linewidth=1)

    # --- Labels and style ---
    plt.title('Mobile Station Antenna Positions', fontsize=14)
    plt.xlabel('X-axis [m]')
    plt.ylabel('Y-axis [m]')
    plt.legend(loc='best')
    plt.grid(True, linestyle='-', alpha=0.4)
    plt.tight_layout()
    plt.show()

# %% ALL BS parameters in one fig
   
l=2/lambda_
nominal_min=nominal_BS_ant_position[:,1].min()
scaled_positions = [(pos-nominal_min) * l for pos in [real_BS_ant_position[:,1], learned_BS_pos, nominal_BS_ant_position[:,1]]]
fig, ax = plot_multiple_parameter_sets(
    scaled_positions,
    [real_BS_gains_normalized, learned_gains_normalized, nominal_BS_gains_normalized],
    [real_BS_coupling_coeff, learned_coupling, nominal_BS_coupling_coeff],
    colors=[color_real,color_MOMP,color_nominal],labels=["Real ", "Learned", "Nominal"],
    y_spacing=2.0,positions_scale=0.8,mag_scale=1.2,
    figsize=(12,8)
)
plt.show()

#%% -------------------------------------- Quantitatif evaluation: quadratic error --------------------------------
# BS antenna positions
print('---------- BS Antenna parameters ---------')
print(f"‖g_real - g_nominal‖²₂ = {np.linalg.norm(real_BS_gains - nominal_BS_gains)**2:.4e}")
print(f"‖g_real - g_learned‖²₂ = {np.linalg.norm(real_BS_gains - learned_gains)**2:.4e}")

print(f"‖P_real - P_nominal‖²₂ = {np.linalg.norm(real_BS_ant_position[:,1] - nominal_BS_ant_position[:,1])**2:.4e}")
print(f"‖P_real - P_learned‖²₂ = {np.linalg.norm(real_BS_ant_position[:,1] - learned_BS_pos)**2:.4e}")

print(f'real coupling coeff: {real_BS_coupling_coeff:.2e} \nlearned coupling coeff: {learned_coupling:.2e}')
#MS antenna positions
print('\n---------- MS Antenna parameters ---------')
rows = []
for u in range(Umax):
    err_nominal = np.linalg.norm(real_MS_ant_position[u, :, 1] - nominal_MS_ant_position[:, 1])
    err_learned = np.linalg.norm(real_MS_ant_position[u, :, 1] - learned_MS_pos[u])

    rows.append([u, err_nominal, err_learned])

# ---- print as a formatted table ----
header = f"{'user':>4} | {'‖P_real - P_nominal‖²₂':>25} | {'‖P_real - P_learned‖²₂':>25}"
print(header)
print("-"*len(header))

for u, e_nom, e_learn in rows:
    print(f"{u:4d} | {e_nom:25.4e} | {e_learn:25.4e}")

