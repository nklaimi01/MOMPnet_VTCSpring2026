#%%
from pathlib import Path
import matplotlib.pyplot as plt
import os
from utils.data_gen_utils import *

nb_users=50
np_positions_per_user=150
nb_BS_antennas=16
nb_MS_antennas=8
nb_subcarriers=128
f0=28e9 #HZ 
c = 3e8  # m/s
lambda_ =  c/f0
BS_position=[60, -90, 30]
delta_f = 120e3 * 12 #according to 5G norms (subcarrier distance) #for every 12 subcarriers can appear a pilot subcarrier
save_dir=Path.cwd()/'.saved_data/Data_new'
os.makedirs(save_dir,exist_ok=True)

#%%########################################################################################################################
################################## realistic synthetic channels generation: ###############################################
###########################################################################################################################
# initialize scene
scene=init_scene_ULA(save_dir,BS_position,f0,nb_BS_antennas,nb_MS_antennas,delta_p_BS=0.24,delta_g_BS=[0.4,np.pi/2],coupling_coeff_BS=0.3*np.exp(1j*(-np.pi/6)))
# Generate Dataset
subcarriers = f0 + np.arange(nb_subcarriers) * delta_f 
#%% channels
channels=generate_channels(save_dir,scene,nb_users,np_positions_per_user,subcarriers,delta_p_MS=0.24)

#%% observations
SNR_av_dB_list = [0, 5, 15] #dB
for snr in SNR_av_dB_list:
    generate_observations(save_dir,channels,snr)

#%%###################################################################
####################### coverage map #################################
######################################################################
def flatten(array):
    return array.reshape(-1, *array.shape[2:])
#load Channels:
channels_dict=np.load(save_dir/'Channels.npz')
channels=channels_dict['channels']
channels_flattened=flatten(channels)
channels_norms=(np.abs(channels_flattened)**2).sum(axis=(1,2,3))
#load UEs positions
users_positions_dict=np.load(save_dir/'users_position.npz')
users_position=users_positions_dict['users_position']
positions_flattened=flatten(users_position)
add_UEs_to_scene(scene,positions_flattened)
scene.preview()

# Extract
x = positions_flattened[:, 0]
y = positions_flattened[:, 1]
# Convert norm → dB (power scale)
norm_db = 10 * np.log10(channels_norms)

# Scatter plot with colormap in dB
plt.figure(figsize=(6, 5))
# sc = plt.scatter(x, y, c=norm_db, cmap="viridis", s=80)
sc = plt.hexbin(
    x, 
    y, 
    C=norm_db,       # values to color by
    cmap="viridis", 
    vmin=-100,
    gridsize=50,     # number of hexagons in x-direction
    alpha=0.8        # transparency
)
plt.scatter(BS_position[0],BS_position[1],color='red',label='BS',s=100,edgecolors='k')
# Add color scale (in dB)
cbar = plt.colorbar(sc)
cbar.set_label("Norm (dB)")
plt.legend()
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Coverage map")
plt.show()

