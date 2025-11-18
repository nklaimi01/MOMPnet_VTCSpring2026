import torch
from pathlib import Path
import numpy as np
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
from utils.dictionary_gen_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
f0 = 28e9  # Hz
c = 3e8    # m/s
lambda_ = c / f0
delta_f = 120e3 * 12  # subcarrier distance
BS_position=[60, -90, 30]
# --- Colors ---
color_observation='#7f7f7f'
color_real_MS ='green'#(1.0, 0.6, 0.6) #pastel_red
color_real_BS = 'green'
color_real='green'
color_nominal = 'red'
color_OMP = 'blue'
color_MOD='purple'
color_MOMP = 'orange'
###########################

save_dir = Path.cwd()/'.saved_data/Data'
# load Channels
channels_dict = np.load(save_dir/'Channels.npz')
channels = torch.from_numpy(channels_dict['channels']).to(device)
users, users_positions, nb_BS_antennas, nb_MS_antennas, nb_subcarriers = channels.shape
subcarriers = f0 + torch.arange(nb_subcarriers, device=device) * delta_f

# load UEs positions
users_positions_dict = np.load(save_dir/'users_position.npz')
users_position = torch.from_numpy(users_positions_dict['users_position']).to(device)

# load MS antennas positions per user
MS_ant_position_dict = np.load(save_dir/'MS_ant_position.npz')
nominal_MS_ant_position = torch.from_numpy(MS_ant_position_dict['nominal_MS_ant_position']).double().to(device)
real_MS_ant_position = torch.from_numpy(MS_ant_position_dict['real_MS_ant_position']).double().to(device)

# load antenna gains at the BS
BS_gains = np.load(save_dir/'BS_gains.npz')
nominal_BS_gains = torch.from_numpy(BS_gains['nominal_BS_gains']).to(device)
real_BS_gains = torch.from_numpy(BS_gains['real_BS_gains']).to(device)

# load antenna positions at the BS
BS_ant_position = np.load(save_dir/'BS_ant_position.npz')
nominal_BS_ant_position = torch.from_numpy(BS_ant_position['nominal_BS_ant_position']).double().to(device)
real_BS_ant_position = torch.from_numpy(BS_ant_position['real_BS_ant_position']).double().to(device)

# load mutual coupling matrix at the BS
BS_coupling = np.load(save_dir/'BS_coupling.npz')
nominal_BS_coupling_coeff = torch.tensor(BS_coupling['nominal_BS_coupling_coeff'], device=device, dtype=torch.complex128)
real_BS_coupling_coeff = torch.tensor(BS_coupling['real_BS_coupling_coeff'], device=device, dtype=torch.complex128)
#load observations
observations_dict = np.load(save_dir/'Observations.npz')
observations = torch.from_numpy(observations_dict['observations']).to(device)
sigma2 = torch.from_numpy(observations_dict['sigma2']).to(device)

########################## Dictionaries #################################
# SV dictionary For BS antennas
nb_BS_atoms = nb_BS_antennas * 10
BS_DoA, BS_angles = generate_DoA(nb_BS_atoms)

real_BS_Dictionary = steering_vect_dict(BS_DoA, real_BS_ant_position, real_BS_gains, real_BS_coupling_coeff, lambda_)
nominal_BS_Dictionary = steering_vect_dict(BS_DoA, nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff, lambda_)

# SV dictionary For MS antennas
nb_MS_atoms = nb_MS_antennas * 10
MS_DoA, MS_angles = generate_DoA(nb_MS_atoms)
MS_gains = torch.ones(nb_MS_antennas, device=device)
MS_coupling_coeff = torch.tensor(0, device=device, dtype=torch.complex128)

nominal_MS_Dictionary = steering_vect_dict(MS_DoA, nominal_MS_ant_position, MS_gains, MS_coupling_coeff, lambda_)
real_MS_dicts_list = []
for user in range(users):
    real_MS_dicts_list.append(steering_vect_dict(MS_DoA, real_MS_ant_position[user], MS_gains, MS_coupling_coeff, lambda_))
real_MS_Dictionaries = torch.stack(real_MS_dicts_list, dim=0)

# FRV dictionary For subcarriers
nb_Subc_atoms = nb_subcarriers * 10
delays = generate_delays(nb_Subc_atoms,delta_f)
FRV_Dictionary = frequency_response_vect_dict(delays, subcarriers, None)