#%%
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
from sionna.rt import Transmitter
import sionna
import tensorflow as tf
from matplotlib.gridspec import GridSpec
import sys
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
from utils.data_gen_utils import init_scene_ULA

nb_BS_antennas=16
nb_MS_antennas=8
nb_subcarriers=128
f0=28e9 #HZ 
c = 3e8  # m/s
lambda_ =  c/f0
BS_position=[60, -90, 30]
delta_f = 120e3 * 12 #according to 5G norms (subcarrier distance) #for every 12 subcarriers can appear a pilot subcarrier
subcarriers = f0 + np.arange(nb_subcarriers) * delta_f 
sigma_p=0.2
sigma_g=0.4
coupling_strength=0.4
path_init = Path.cwd()/'.saved_data'
save_dir=path_init/'visual'
os.makedirs(save_dir,exist_ok=True)
#%% functions

def generate_DoA(nb_DoA: int):
    """
    Generate directions of arrival (DoA) vectors and corresponding angles (NumPy version).
    
    Returns:
        DoA : [nb_DoA, 3] array
        angles : [nb_DoA] array
    """
    DoA = np.zeros((nb_DoA, 3), dtype=np.float64)

    cos_vals = np.linspace(-1, 1, nb_DoA, dtype=np.float64)
    angles = np.flip(np.arccos(cos_vals))

    DoA[:, 0] = np.sin(angles)
    DoA[:, 1] = np.cos(angles)
    # DoA[:, 2] remains 0

    return DoA, angles

def steering_vect_dict(DoA: np.ndarray,
                       antenna_pos: np.ndarray,
                       antenna_gains: np.ndarray,
                       antenna_coupling: np.ndarray,
                       lambda_: float) -> np.ndarray:
    """
    Compute normalized steering vector dictionary (NumPy version).

    DoA: [A, 3] array of directions of arrival
    antenna_pos: [N, 3] array of antenna positions
    antenna_gains: [N] array of complex gains
    antenna_coupling: [N, N] array (mutual coupling matrix)
    lambda_: wavelength (float)
    """

    # Exponential term: [N, A]
    expo = np.exp(-1j * 2 * np.pi * (1.0 / lambda_) * (antenna_pos @ DoA.T))

    # Apply gains
    dict_ = antenna_gains[:, None] * expo

    # Apply antenna coupling
    dict_ = antenna_coupling @ dict_

    # Normalize each column
    norm_factor = np.sqrt(np.sum(np.abs(dict_) ** 2, axis=0))
    dict_ = dict_ / norm_factor

    return dict_

def frequency_response_vect_dict(Delays: np.ndarray,
                                 subcarrier_freq: np.ndarray,
                                 antenna_gains: np.ndarray = None) -> np.ndarray:
    """
    Compute frequency response vector dictionary (NumPy version).
    
    Delays : [A] array
        Array of delays.
    subcarrier_freq : [subc] array
        Array of subcarrier frequencies.
    antenna_gains : optional [A] array
        Complex gains (if needed, else ignored).
    """

    # Outer product: [subc, A]
    exponent = -1j * 2 * np.pi * np.outer(subcarrier_freq, Delays)
    dict_ = np.exp(exponent)

    # Optionally apply gains
    if antenna_gains is not None:
        dict_ = antenna_gains[None, :] * dict_

    # Normalize each column
    norm_factor = np.sqrt(np.sum(np.abs(dict_) ** 2, axis=0))
    dict_ = dict_ / norm_factor

    return dict_

def angle_delay_map(channel,angles_dict,delays_dict):
    '''
    computes angle delay matrix 
    channel [nb users,nb BS antennas, nb MS antennas, nb subcarriers]
    '''
    angle_delay_map=np.einsum('ab,ubmk->uamk',np.conj(angles_dict).T,channel)
    angle_delay_map=np.einsum('ak,ubmk->ubma',np.conj(delays_dict).T,angle_delay_map)
    angle_delay_map=((np.abs(angle_delay_map)**2).sum(axis=2))
    
    return angle_delay_map

def subplot_ADM(position, real_ADM, nominal_ADM, angles, delays, user=None, zoom=None,show_range=False,dB=True,show_max=False):
    '''
    user=[dx,dy,dz] user's relative position w/ respect to the BS
    zoom=[tau_min, tau_max, phi_min, phi_max] / delays (tau) in [us] and angles (phi) in [deg]
    '''
    real_ADM = real_ADM[position]
    nominal_ADM = nominal_ADM[position]
    if dB:
        real_ADM=10*np.log10(real_ADM)
        nominal_ADM=10*np.log10(nominal_ADM)

    fig, axes = plt.subplots(1, 2, figsize=(12,5))  # 1 row, 2 columns

    # Convert to usable scales
    angles_deg = np.rad2deg(angles)         # [deg]
    cos_angles = np.cos(angles)             # [-1,1]
    delays_us = delays * 1e6                # [µs]

    if zoom is None:
        extent = [delays_us.min(), delays_us.max(), cos_angles.max(), cos_angles.min()]  
        # y-axis = cos(angle) (from 1 to -1)
        len_y=angles_deg.shape[0]
        yticks_deg = angles_deg[np.arange(0,len_y,int(len_y/7))]  # choose tick positions in deg
        xticks_m=delays*3e8
        len_x=len(xticks_m)
        xticks_m=xticks_m[np.arange(0,len_x,int(len_x/7))]
    else:
        tau_min_us, tau_max_us, phi_min_deg, phi_max_deg = zoom
        # Masks on delay and angle
        delay_mask = (delays_us >= tau_min_us) & (delays_us <= tau_max_us)
        angle_mask = (angles_deg >= phi_min_deg) & (angles_deg <= phi_max_deg)

        # Crop ADMs
        real_ADM = real_ADM[np.ix_(angle_mask, delay_mask)]
        nominal_ADM  = nominal_ADM[np.ix_(angle_mask, delay_mask)]

        extent = [tau_min_us, tau_max_us,
                  np.cos(np.deg2rad(phi_min_deg)), np.cos(np.deg2rad(phi_max_deg))]
        
        amin = np.argmin(np.abs(angles_deg - phi_min_deg))
        amax=np.argmin(np.abs(angles_deg - phi_max_deg))
        len_y=angles_deg[amin:amax].shape[0]
        yticks_deg = angles_deg[np.arange(amin,amax,int(len_y/7))]  # choose tick positions in deg
        xticks_m=delays[delay_mask]*3e8
        len_x=len(xticks_m)
        xticks_m=xticks_m[np.arange(0,len_x,int(len_x/7))]

        

    # Color limits shared
    vmin = min(real_ADM.min(), nominal_ADM.min())
    vmax = max(real_ADM.max(), nominal_ADM.max())

    # First subplot
    im1 = axes[0].imshow(real_ADM,
                         aspect='auto',
                         extent=extent,
                         origin='lower',
                         vmin=vmin, vmax=vmax)
    axes[0].set_title("Full BS Imperfection Knowledge")
    axes[0].set_xlabel("Delay [µs]")
    axes[0].set_ylabel("Angle [deg]")

    # Second subplot
    im2 = axes[1].imshow(nominal_ADM,
                         aspect='auto',
                         extent=extent,
                         origin='lower',
                         vmin=vmin, vmax=vmax)
    axes[1].set_title("Assuming No Imperfections")
    axes[1].set_xlabel("Delay [µs]")
    axes[1].set_ylabel("Angle [deg]")

    # Shared colorbar
    cbar = fig.colorbar(im1, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("[dB]" if dB else "")
    fig.suptitle("Angle-Delay Maps", fontsize=14)

    # Fix y-ticks: show cos(angle) but label in degrees
    yticks_cos = np.cos(np.deg2rad(yticks_deg))
    for ax in axes:
        ax.set_yticks(yticks_cos)
        ax.set_yticklabels([f"{d:.0f}°" for d in yticks_deg])
        if show_range:
            ax.set_xticklabels([f"{d:.1f}" for d in xticks_m])
            ax.set_xlabel("Range [m]")
            fig.suptitle("Angle-Range Maps", fontsize=14)

    if show_max:  
        # find maximum of the correlation matrix
        r_angle_idx,r_delay_idx = np.unravel_index(np.argmax(real_ADM) , real_ADM.shape)  # (BS_angle_idx, delay_idx)
        r_max_cos_angle=cos_angles[r_angle_idx]
        r_max_delay_us=delays_us[r_delay_idx]
        axes[0].scatter(r_max_delay_us, r_max_cos_angle, color='blue', marker='s', s=100, label='Maximum')

        n_angle_idx,n_delay_idx = np.unravel_index(np.argmax(nominal_ADM), nominal_ADM.shape)  # (BS_angle_idx, delay_idx)
        n_max_cos_angle=cos_angles[n_angle_idx]
        n_max_delay_us=delays_us[n_delay_idx]
        axes[1].scatter(n_max_delay_us, n_max_cos_angle, color='blue', marker='s', s=100, label='Maximum')

    # Optional: mark user
    if user is not None:
        dx, dy, dz = user
        user_angle_rd = np.pi - np.abs(np.arctan2(dx, dy))  
        user_delay_us = np.sqrt(dx**2 + dy**2 + dz**2) / 3e8 * 1e6
        user_cos = np.cos(user_angle_rd)

        axes[0].scatter(user_delay_us, user_cos, color='red', marker='x', s=100, label='User')
        axes[1].scatter(user_delay_us, user_cos, color='red', marker='x', s=100, label='User')

    plt.show()

def plot_w_marginals(ADM,angle_map,delay_map, angles, delays, dB=True, show_max=True, user=None):
    '''
    user=[dx,dy,dz] user's relative position w/ respect to the BS
    '''
    if dB:
        ADM=10*np.log10(ADM)
        # angle_map=10*np.log10(angle_map)
        # delay_map=10*np.log10(delay_map)

    # Convert to usable scales
    angles_deg = np.rad2deg(angles)         # [deg]
    cos_angles = np.cos(angles)             # [-1,1]
    delays_us = delays * 1e6                # [µs]

    right_marginal = angle_map                 
    top_marginal = delay_map     
    fig = plt.figure(figsize=(10,9))
    gs = GridSpec(2, 4, width_ratios=[0.5,1, 8, 2], height_ratios=[2, 8],
                wspace=0, hspace=0)
    
    # main Heatmap
    ax_main = fig.add_subplot(gs[1, 2])

    # Top marginal (column) as line
    ax_top = fig.add_subplot(gs[0, 2],sharex=ax_main)
    ax_top.semilogy(delays_us,top_marginal)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.tick_params(axis='x',bottom=False,labelbottom=False)
    ax_top.set_title("Delay map")
    ax_top.axvline(delays_us[top_marginal.argmax()],color='orange')

    # Right marginal (row) as line
    ax_right = fig.add_subplot(gs[1, 3],sharey=ax_main)
    ax_right.semilogx(right_marginal, cos_angles)
    ax_right.invert_yaxis()
    ax_right.set_yticks([])
    ax_right.set_yticklabels([])
    ax_right.tick_params(axis='y',left=False,labelleft=False)
    ax_right.tick_params(axis='x',bottom=True,labelbottom=True)
    ax_right.set_title("Angle map")
    ax_right.axhline(cos_angles[right_marginal.argmax()],color='orange')

    # # Remove spines from top marginal
    # for spine in ["top", "right", "left"]:
    #     ax_top.spines[spine].set_visible(False)

    # # Remove spines from right marginal
    # for spine in ["top", "right"]:
    #     ax_right.spines[spine].set_visible(False)

    # Main heatmap
    extent = [delays_us.min(), delays_us.max(), cos_angles.max(), cos_angles.min()]  
    # y-axis = cos(angle) (from 1 to -1)
    len_y=angles_deg.shape[0]
    yticks_deg = angles_deg[np.arange(0,len_y,int(len_y/7))]  # choose tick positions in deg

    # Color limits shared
    vmin = ADM.min()
    vmax = ADM.max()

    # First subplot
    im = ax_main.imshow(ADM,
                         aspect='auto',
                         extent=extent,
                         origin='lower',
                         vmin=vmin, vmax=vmax)
    # ax_main.set_title("Full BS Imperfection Knowledge")
    ax_main.set_xlabel("Delay [µs]")
    ax_main.set_ylabel("Angle [deg]")


    # Shared colorbar
    cax = fig.add_subplot(gs[1, 0])
    cbar = fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04)
    cbar.set_label("[dB]" if dB else "")
    cbar.ax.yaxis.set_ticks_position('left')   # 'left', 'right', or 'both'
    cbar.ax.yaxis.set_label_position('left')   # move the label too if needed
    fig.suptitle("Angle-Delay Maps", fontsize=14)

    # Fix y-ticks: show cos(angle) but label in degrees
    yticks_cos = np.cos(np.deg2rad(yticks_deg))
    ax_main.set_yticks(yticks_cos)
    ax_main.set_yticklabels([f"{d:.0f}°" for d in yticks_deg])

    xticks_us = delays_us[np.arange(0, len(delays_us), max(1, len(delays_us)//7))]
    ax_main.set_xticks(xticks_us)
    ax_main.set_xticklabels([f"{x:.1f}" for x in xticks_us])

    if show_max:  
        # find maximum of the correlation matrix
        ax_main.scatter(delays_us[top_marginal.argmax()], cos_angles[right_marginal.argmax()], color='orange', marker='s', s=80, label='MOMP Maximum correlation')

        r_angle_idx,r_delay_idx = np.unravel_index(np.argmax(ADM) , ADM.shape)  # (BS_angle_idx, delay_idx)
        ax_main.scatter(delays_us[r_delay_idx], cos_angles[r_angle_idx], color='red', marker='s', s=80, label='OMP Maximum correlation')


    # Optional: mark user
    if user is not None:
        dx, dy, dz = user
        user_angle_rd = np.pi - np.abs(np.arctan2(dx, dy))  
        user_delay_us = np.sqrt(dx**2 + dy**2 + dz**2) / 3e8 * 1e6
        user_cos = np.cos(user_angle_rd)

        ax_main.scatter(user_delay_us, user_cos, color='black', marker='x', s=100, label='User position (geometry)')
    fig.legend()
    plt.show()
#%% initialize scene
scene=init_scene_ULA(save_dir,BS_position,f0,nb_BS_antennas,nb_MS_antennas,delta_p_BS=0.2,delta_g_BS=[0.2,0.2],coupling_coeff_BS=0.15*np.exp(1j*(-np.pi/6)))

#load antenna gains at the BS:
BS_gains=np.load(save_dir/'BS_gains.npz')
nominal_BS_gains=BS_gains['nominal_BS_gains']
real_BS_gains=BS_gains['real_BS_gains']
#load antenna positions at the BS:
BS_ant_position=np.load(save_dir/'BS_ant_position.npz')
nominal_BS_ant_position=BS_ant_position['nominal_BS_ant_position']
real_BS_ant_position=BS_ant_position['real_BS_ant_position']
#load mutual coupling matrix at the BS
# load mutual coupling matrix at the BS
BS_coupling = np.load(save_dir/'BS_coupling.npz')
nominal_BS_coupling_coeff = BS_coupling['nominal_BS_coupling_coeff']
real_BS_coupling_coeff = BS_coupling['real_BS_coupling_coeff']


#%% Generate Channels
rng = np.random.default_rng(seed=None)
# position_array=np.array(([160,-90,1.5],[-75,55,1.5],[0,-190,1.5],[225,-150,1.5]))#,[210,-90,1.5])) #positions of presentation 25
position_array=np.array(([ -34.7816, -108.0387,    1.5000],[-75,55,1.5]))
nb_positions=position_array.shape[0] # number of random positions in the grid

#Generate random UE antenna array orientation : A 3D rotation with yaw, pitch, and roll angles
yaw = 2*np.pi*rng.random(nb_positions)
pitch= 2*np.pi*rng.random(nb_positions)
roll= 2*np.pi*rng.random(nb_positions)
orientation_array=np.column_stack([yaw, pitch, roll])  # combine into table (nb_positions, 3)
#generate real and nominal positions at the BS
scene.tx_array.positions=tf.Variable(scene.tx_array.positions)
scene.tx_array.positions[:,1].assign(scene.tx_array.positions[:,1]+ 0.1*lambda_*np.random.randn(scene.tx_array.num_ant))
#generate realistic synthetic channels using sionna RT
#Add UE 
for tx in scene.transmitters.copy():   # copy to avoid modifying while iterating
        scene.remove(tx)
for idx in range(len(position_array)):
    tx = Transmitter(name='tx'+f'_{idx}',
            position=position_array[idx],
                orientation=orientation_array[idx])
    scene.add(tx)
paths=scene.compute_paths()
paths.normalize_delays=False
scene.preview(paths=paths)
types=paths.types.numpy() #0=LOS #TODO problem is it gives all paths without knowing the sources
#%%
a,tau=paths.cir()
# Make sure a and tau are float64
a = tf.cast(a, tf.complex128)      # amplitudes can be complex
tau = tf.cast(tau, tf.float64)     # delays
# Also make sure subcarriers are float64
subcarriers = tf.cast(subcarriers, tf.float64)
#built in function method : cir
H_sionna = sionna.channel.cir_to_ofdm_channel(subcarriers,a,tau,normalize=False)
H_sionna = H_sionna.numpy().squeeze() #[nb_BS_antenna , nb_chan_train]
channel=H_sionna.transpose(1,0,2,3)
# remove 0 norm paths and save the rest
norms=(np.abs(channel).sum(axis=(1,2,3)))
idx_to_delete=np.where(norms==0)
channel_filtered=np.delete(channel,idx_to_delete,axis=0)
channel_with_gains=np.einsum('a,uamk->uamk',real_BS_gains,channel_filtered) #consider antennas gains at BS

off_diag = np.full(nb_BS_antennas - 1, real_BS_coupling_coeff, dtype=np.complex128)
real_BS_coupling = np.eye(nb_BS_antennas, dtype=np.complex128) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
channel_coupled=np.einsum('ab,ubmk->uamk',real_BS_coupling,channel_with_gains) #consider antennas coupling at BS

H=channel_coupled

#%% dictionary generation:
# For BS antennas:
nb_BS_atoms=nb_BS_antennas*10 #DoAs
#generate different DoAs at BS
BS_DoA,BS_angles=generate_DoA(nb_BS_atoms)
real_BS_Dictionary=steering_vect_dict(BS_DoA,real_BS_ant_position,real_BS_gains,real_BS_coupling_coeff,lambda_)
nominal_BS_Dictionary=steering_vect_dict(BS_DoA,nominal_BS_ant_position,nominal_BS_gains,nominal_BS_coupling_coeff,lambda_)
# delay dictionary at subcarriers level
nb_Subc_atoms=nb_subcarriers*10 #delays
max_distance=(c/delta_f)
delays=np.linspace(0,max_distance,nb_Subc_atoms)/c
FRV_Dictionary=frequency_response_vect_dict(delays,subcarriers,None)

#%%##################################################################################################################
################################################ Angle delay map #####################################################
######################################################################################################################
real_angle_delay_map=angle_delay_map(H,real_BS_Dictionary,FRV_Dictionary)
nominal_angle_delay_map=angle_delay_map(H,nominal_BS_Dictionary,FRV_Dictionary)

#%%

# for p in range(real_angle_delay_map.shape[0]):
#     subplot_ADM(p,real_angle_delay_map,nominal_angle_delay_map,BS_angles,delays,user=position_array[p]-BS_position,show_max=True)

plt.figure()
corr=np.abs(np.conj(FRV_Dictionary).T@FRV_Dictionary)
plt.imshow(corr)

#%% angle map / delay map
#angle map
real_angle_map=np.einsum('ab,ubmk->uamk',np.conj(real_BS_Dictionary).T,H)
real_angle_map=((np.abs(real_angle_map)**2).sum(axis=(2,3)))

nominal_angle_map=np.einsum('ab,ubmk->uamk',np.conj(nominal_BS_Dictionary).T,H)
nominal_angle_map=((np.abs(nominal_angle_map)**2).sum(axis=(2,3)))

# delay map
delay_map=np.einsum('ak,ubmk->ubma',np.conj(FRV_Dictionary).T,H)
delay_map=((np.abs(delay_map)**2).sum(axis=(1,2)))

# position=1
# #placer position utilisateur (géométrie)
# #Position utilisateur

# dU=position_array[position]-BS_position
# dx,dy,dz=dU
# user_angle_rd = np.pi-np.abs(np.arctan2(dx,dy))
# user_delay = np.sqrt(dx**2 + dy**2 + dz**2)/c

# plt.figure()
# plt.plot(BS_angles,real_angle_map[position],label='real dicitonary')
# plt.plot(BS_angles,nominal_angle_map[position],label='nominal dicitonary')
# # plt.scatter(user_tau,0)
# plt.axvline(user_angle_rd,color='r',label='user angle (geometry)')
# idxmax=real_angle_map[position].argmax()
# # plt.axvline(BS_angles[idxmax], color='grey', linestyle='--', label='max')
# plt.legend()


# #placer position utilisateur (géométrie)
# plt.figure()
# plt.plot(delays,delay_map[position],label='delay map')
# # plt.scatter(user_tau,0)
# plt.axvline(user_delay,color='r',label='user delay (geometry)')
# idxmax=delay_map[position].argmax()
# # plt.axvline(delays[idxmax]*c, color='grey', linestyle='--', label='max')
# plt.legend()


# %%

# plot_w_marginals(real_angle_delay_map[position],real_angle_map[position],delay_map[position],BS_angles,delays)
#%%
# for p in range(len(position_array)):
for p in [0]:
    plot_w_marginals(real_angle_delay_map[p],real_angle_map[p],delay_map[p],BS_angles,delays,user=position_array[p]-BS_position)
    plot_w_marginals(nominal_angle_delay_map[p],nominal_angle_map[p],delay_map[p],BS_angles,delays,user=position_array[p]-BS_position)





# %%
