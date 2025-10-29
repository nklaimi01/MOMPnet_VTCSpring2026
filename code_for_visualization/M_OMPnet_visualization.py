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

# --- Colors ---
color_real_MS ='green'#(1.0, 0.6, 0.6) #pastel_red
color_real_BS = 'green'
color_real='green'
color_nominal = 'purple'
color_OMP = 'blue'
color_MOMP = 'orange'
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
                label='OMP before training', color=color_OMP,alpha=0.5)
bars2 = plt.bar(channels_idx - width/2, NMSEZ_OMP_tensor.detach().cpu().numpy(), width,
                label='after training with OMPnet', color=color_OMP)
bars4 = plt.bar(channels_idx + width/2, NMSE0_MOMP_tensor.detach().cpu().numpy(), width,
                label='MOMP before training', color=color_MOMP,alpha=0.5)
bars3 = plt.bar(channels_idx + width/2, NMSEZ_MOMP_tensor.detach().cpu().numpy(), width,
                label='after training with MOMPnet', color=color_MOMP)
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
# --- Calcul des erreurs quadratiques ---
err_nominal = torch.sum(torch.abs(real_BS_ant_position[:, 1] - nominal_BS_ant_position[:, 1])**2)
err_learned_OMP = torch.sum(torch.abs(real_BS_ant_position[:, 1] - torch.tensor(learned_BS_pos_OMP))**2)
err_learned_MOMP = torch.sum(torch.abs(real_BS_ant_position[:, 1] - torch.tensor(learned_BS_pos_MOMP))**2)

# --- Affichage des valeurs ---
print(f"‖P_real - P_nominal‖²₂ = {err_nominal.item():.4e}")
print(f"‖P_real - P_learnedOMP‖²₂ = {err_learned_OMP.item():.4e}")
print(f"‖P_real - P_learnedMOMP‖²₂ = {err_learned_MOMP.item():.4e}")
# %%
