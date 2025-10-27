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

SNR_average=10*torch.log10(torch.mean(torch.sum(torch.abs(channels)**2, axis=(2, 3, 4))) / (16*8*128 * sigma2))
print(f'average SNR={SNR_average}')
# LOAD TRAINED MODELS
############################ MOMP ############################

unfolded_MOMP_model = MOMP_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff, nominal_MS_ant_position,
                 subcarriers, BS_DoA, MS_DoA, delays)  # replace with your model class
# Load everything
checkpoint = torch.load('.saved_data/.saved_models/MOMP_model_and_metrics.pth')
# Load model weights
unfolded_MOMP_model.load_state_dict(checkpoint['model_state_dict'])
# Load the lists
NMSE0_MOMP = checkpoint['NMSE0']
NMSEZ_MOMP = checkpoint['NMSEZ']
train_losses_MOMP = checkpoint['train_losses']
valid_losses_MOMP = checkpoint['valid_losses']

############################ OMP ############################
unfolded_OMP_model = OMP_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff, nominal_MS_ant_position,
                 subcarriers, BS_DoA, MS_DoA, delays)  # replace with your model class
# Load everything
checkpoint = torch.load('.saved_data/.saved_models/OMP_model_and_metrics.pth')
# Load model weights
unfolded_OMP_model.load_state_dict(checkpoint['model_state_dict'])
# Load the lists
NMSE0_OMP = checkpoint['NMSE0']
NMSEZ_OMP = checkpoint['NMSEZ']
train_losses_OMP = checkpoint['train_losses']
valid_losses_OMP = checkpoint['valid_losses']

print("Model and lists loaded successfully!")

#----------------------------------- plots -----------------------------------
#%% Test nmse
channels_idx = torch.arange(1, len(NMSE0_OMP) + 1)
width = 0.3
NMSE0_OMP_tensor = torch.stack(NMSE0_OMP).reshape(-1)  # shape: [num_channels]
NMSE0_MOMP_tensor = torch.stack(NMSE0_MOMP).reshape(-1)  # shape: [num_channels]
NMSEZ_OMP_tensor = torch.stack(NMSEZ_OMP).reshape(-1)
NMSEZ_MOMP_tensor = torch.stack(NMSEZ_MOMP).reshape(-1)
plt.figure(figsize=(8, 5))
# Then detach and convert to numpy
bars1 = plt.bar(channels_idx - width/2, NMSE0_OMP_tensor.detach().cpu().numpy(), width,
                label='OMP before training', color='red',alpha=0.5)
bars2 = plt.bar(channels_idx - width/2, NMSEZ_OMP_tensor.detach().cpu().numpy(), width,
                label='after training with OMPnet', color='red')
bars4 = plt.bar(channels_idx + width/2, NMSE0_MOMP_tensor.detach().cpu().numpy(), width,
                label='MOMP before training', color='green',alpha=0.5)
bars3 = plt.bar(channels_idx + width/2, NMSEZ_MOMP_tensor.detach().cpu().numpy(), width,
                label='after training with MOMPnet', color='green')
plt.xticks(channels_idx)
plt.semilogy()
plt.legend()
plt.show()

#%% Plotting learning curve
# Convert list of tensors -> average NMSE per epoch
train_losses_avg_OMP = [t.mean().item() for t in train_losses_OMP]
valid_losses_avg_OMP = [v.mean().item() for v in valid_losses_OMP]
train_losses_avg_MOMP = [t.mean().item() for t in train_losses_MOMP]
valid_losses_avg_MOMP = [v.mean().item() for v in valid_losses_MOMP]
epochs = range(0, len(train_losses_avg_MOMP))

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses_avg_OMP, label='OMPnet Training loss', marker='x',linestyle='--', color='blue')
plt.plot(epochs, train_losses_avg_MOMP, label='MOMPnet Training loss', marker='o', color='blue')

plt.plot(epochs, valid_losses_avg_OMP, label='OMPnet Validation loss', marker='^',linestyle='--', color='orange')
plt.plot(epochs, valid_losses_avg_MOMP, label='MOMPnet Validation loss', marker='s', color='orange')

plt.gca().spines['left'].set_position('zero')
plt.xlabel('Epoch')
plt.xticks(epochs)
plt.ylabel('NMSE')
plt.title('Learning Curve')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

#%%
learned_BS_pos_OMP=list(unfolded_OMP_model.parameters())[0].detach().numpy()  # first parameter tensor
learned_gains_OMP=list(unfolded_OMP_model.parameters())[1].detach().numpy()  # first parameter tensor
learned_coupling_OMP=list(unfolded_OMP_model.parameters())[2].detach().numpy()  # first parameter tensor
learned_MS_pos_OMP=list(unfolded_OMP_model.parameters())[3].detach().numpy()  # first parameter tensor

learned_BS_pos_MOMP=list(unfolded_MOMP_model.parameters())[0].detach().numpy()  # first parameter tensor
learned_gains_MOMP=list(unfolded_MOMP_model.parameters())[1].detach().numpy()  # first parameter tensor
learned_coupling_MOMP=list(unfolded_MOMP_model.parameters())[2].detach().numpy()  # first parameter tensor
learned_MS_pos_MOMP=list(unfolded_MOMP_model.parameters())[3].detach().numpy()  # first parameter tensor


####################################### plot learned MS postions #############################################
# --- Colors ---
color_real_MS =(1.0, 0.6, 0.6) #pastel_red
color_real_BS = 'orange'
color_nominal = 'black'
color_OMP = 'red'
color_MOMP = 'green'

# --- X and Y coordinates ---
x = real_MS_ant_position[0, :, 0].cpu().numpy()
y_nominal = nominal_MS_ant_position[:, 1].cpu().numpy()
y_real = real_MS_ant_position[0, :, 1].cpu().numpy()
y_OMP = torch.tensor(learned_MS_pos_OMP).cpu().numpy()
y_MOMP = torch.tensor(learned_MS_pos_MOMP).cpu().numpy()

# --- Apply small horizontal offsets for visibility ---
offset = 0.02  # adjust if antennas are close
x_nominal = x - offset * 1.5
x_real    = x - offset * 0.5
x_OMP     = x + offset * 0.5
x_MOMP    = x + offset * 1.5

# --- Plot ---
plt.figure(figsize=(8,5))

plt.scatter(x_nominal, y_nominal, label='Nominal MS', marker='x', color=color_nominal, s=50, linewidths=1)
plt.scatter(x_real, y_real, label='Real MS', color=color_real_MS, s=70, edgecolors='k', alpha=0.8)
plt.scatter(x_OMP, y_OMP, label='Learned MS (OMP)', color=color_OMP, s=70, edgecolors='k', alpha=0.8)
plt.scatter(x_MOMP, y_MOMP, label='Learned MS (MOMP)', color=color_MOMP, s=70, edgecolors='k', alpha=0.8)

# --- Optional: connect each antenna index with dotted lines ---
for i in range(len(x)):
    plt.plot([x_nominal[i], x_real[i], x_OMP[i], x_MOMP[i]],
             [y_nominal[i], y_real[i], y_OMP[i], y_MOMP[i]],
             color='gray', linestyle='--', alpha=0.4, linewidth=1)

# --- Labels and style ---
plt.title('Mobile Station Antenna Positions', fontsize=14)
plt.xlabel('X-axis [m]')
plt.ylabel('Y-axis [m]')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

############################################ plot learned BS antenna Gains ####################################
# --- Prepare data for plotting ---
nominal_BS_gains = torch.from_numpy(BS_gains['nominal_BS_gains']).to(device)
real_BS_gains_normalized = real_BS_gains / torch.sqrt(torch.sum((torch.abs(real_BS_gains)**2)))
nominal_BS_gains_normalized = nominal_BS_gains / torch.sqrt(torch.sum((torch.abs(nominal_BS_gains)**2)))
learned_gains_OMP_normalized= learned_gains_OMP / np.sqrt(np.sum((np.abs(learned_gains_OMP)**2)))
learned_gains_MOMP_normalized= learned_gains_MOMP / np.sqrt(np.sum((np.abs(learned_gains_MOMP)**2)))

idx = np.arange(len(real_BS_gains_normalized))
mag_real = torch.abs(real_BS_gains_normalized).cpu()
mag_nominal = torch.abs(nominal_BS_gains_normalized).cpu()
mag_OMP = torch.abs(torch.tensor(learned_gains_OMP_normalized)).cpu()
mag_MOMP = torch.abs(torch.tensor(learned_gains_MOMP_normalized)).cpu()

phase_real = torch.angle(real_BS_gains_normalized).cpu()
phase_nominal = torch.angle(nominal_BS_gains_normalized).cpu()
phase_OMP = torch.angle(torch.tensor(learned_gains_OMP_normalized)).cpu()
phase_MOMP = torch.angle(torch.tensor(learned_gains_MOMP_normalized)).cpu()

# --- Plot magnitude comparison ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(idx, mag_real, 'o-', label='Real', color=color_real_BS)
plt.plot(idx, mag_nominal, 'x--', label='Nominal', color=color_nominal)
plt.plot(idx, mag_OMP, 's-', label='OMP Learned', color=color_OMP)
plt.plot(idx, mag_MOMP, 'd-', label='MOMP Learned', color=color_MOMP)
plt.title('Antenna Gain Magnitudes')
plt.xlabel('Antenna Index')
plt.ylabel('|Gain|')
plt.legend()
plt.grid(True)

# --- Plot phase comparison ---
plt.subplot(1,2,2)
plt.plot(idx, phase_real, 'o-', label='Real', color=color_real_BS)
plt.plot(idx, phase_nominal, 'x--', label='Nominal', color=color_nominal)
plt.plot(idx, phase_OMP, 's-', label='OMP Learned', color=color_OMP)
plt.plot(idx, phase_MOMP, 'd-', label='MOMP Learned', color=color_MOMP)
plt.title('Antenna Gain Phases')
plt.xlabel('Antenna Index')
plt.ylabel('Phase [rad]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#%%
# --- Calcul des erreurs quadratiques ---
err_nominal = torch.sum(torch.abs(real_BS_ant_position[:, 1] - nominal_BS_ant_position[:, 1])**2)
err_learned_OMP = torch.sum(torch.abs(real_BS_ant_position[:, 1] - torch.tensor(learned_BS_pos_OMP))**2)
err_learned_MOMP = torch.sum(torch.abs(real_BS_ant_position[:, 1] - torch.tensor(learned_BS_pos_MOMP))**2)

# --- Affichage des valeurs ---
print(f"‖P_real - P_nominal‖²₂ = {err_nominal.item():.4e}")
print(f"‖P_real - P_learnedOMP‖²₂ = {err_learned_OMP.item():.4e}")
print(f"‖P_real - P_learnedMOMP‖²₂ = {err_learned_MOMP.item():.4e}")
# %%
