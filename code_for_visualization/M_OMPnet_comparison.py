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
from saved_data_loader import *
from models.MOMP_model import MOMP_model
from models.OMP_model import OMP_model
from utils.training_utils import *

SNR_average=10*torch.log10(torch.mean(torch.sum(torch.abs(channels)**2, axis=(2, 3, 4))) / (16*8*128 * sigma2))
print(f'average SNR={SNR_average}')

#DATA
Umax,Pmax=5,10
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

checkpoint = torch.load('MOMP_model_and_metrics.pth')
# Load model weights
unfolded_MOMP_model.load_state_dict(checkpoint['model_state_dict'])
# Load the lists
NMSE_nominal_MOMP = checkpoint['NMSE0']
NMSE_MOMP = checkpoint['NMSEZ']
NMSE_real_MOMP = checkpoint['NMSE_real']
train_losses_MOMP = checkpoint['train_losses']
valid_losses_MOMP = checkpoint['valid_losses']

unfolded_OMP_model = OMP_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff, nominal_MS_ant_position_stacked,
                 subcarriers, BS_DoA, MS_DoA, delays)  # replace with your model class

checkpoint = torch.load('OMP_model_and_metrics.pth')
# Load model weights
unfolded_OMP_model.load_state_dict(checkpoint['model_state_dict'])
# Load the lists
NMSE_nominal_OMP = checkpoint['NMSE0']
NMSE_OMP = checkpoint['NMSEZ']
NMSE_real_OMP = checkpoint['NMSE_real']
train_losses_OMP= checkpoint['train_losses']
valid_losses_OMP = checkpoint['valid_losses']

#%%#############################################################################################################################################################################
##################################################################  compare models ############################################################################################
################################################################################################################################################################################
 
#--------------------------------------------------Plotting learning curve------------------------------------------------------------------
# Convert list of tensors -> average NMSE per epoch
train_losses_avg_MOMP = [t.mean().item() for t in train_losses_MOMP]
valid_losses_avg_MOMP = [v.mean().item() for v in valid_losses_MOMP]
train_losses_avg_OMP = [t.mean().item() for t in train_losses_OMP]
valid_losses_avg_OMP = [v.mean().item() for v in valid_losses_OMP]

epochs = range(0, len(train_losses_avg_MOMP))

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses_avg_MOMP, label='MOMPnet Train NMSE', marker='o', color='blue')
plt.plot(epochs, valid_losses_avg_MOMP, label='MOMPnet Validation NMSE', marker='v', color='orange')
plt.plot(epochs, train_losses_avg_OMP, label='OMPnet Train NMSE', linestyle='--', marker='o', color='blue')
plt.plot(epochs, valid_losses_avg_OMP, label='OMPnet Validation NMSE', linestyle='--', marker='^', color='orange')

plt.gca().spines['left'].set_position('zero')
plt.xlabel('Epoch')
# plt.xticks(epochs)
plt.ylabel('NMSE')
plt.title('Learning Curve')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
#--------------------------------------------------Plotting testing NMSE------------------------------------------------------------------

# Filter and slice data

nmse0=NMSE(H_test,Y_test)
nmse1 = NMSE_nominal_MOMP
nmse2 = NMSE_OMP
nmse3 = NMSE_MOMP
nmse4 = NMSE_real_MOMP
# idx = torch.where(nmse0 < 1)
# nmse0 = nmse_0[idx]
# nmse1 = nmse_1[idx]
# nmse2 = nmse_2[idx]
# nmse3 = nmse_3[idx]

# Compute means
means = [
    nmse0.mean().item(),
    nmse1.mean().item(),
    nmse2.mean().item(),
    nmse3.mean().item(),
    nmse4.mean().item()
]

# Labels and colors
labels = [
    'Observation error',
    'MOMP with nominal Dicts',
    'OMPnet',
    'MOMPnet',
    'MOMP with real Dicts'
]
colors = [color_observation, color_nominal,color_OMP, color_MOMP, color_real]
width=0.5
# Plot
# Plot
plt.figure(figsize=(8, 5))
x = np.arange(len(means))
bars = plt.bar(x, means, width=width, color=colors, alpha=0.7)

# Annotate only OMPnet (index 2) and MOMPnet (index 3)
for idx in [2, 3]:
    plt.text(
        x[idx],                         # x-position at the bar
        means[idx],                     # y-position at the top of bar
        f"{means[idx]:.2e}",            # scientific notation
        ha='center', va='bottom',       # centered horizontally, above bar
        fontsize=10
    )

plt.yscale('log')
plt.xticks(x, labels, rotation=20)
plt.ylabel('NMSE (mean)')
plt.title('Mean NMSE comparison')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
