#%% importing libraries
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from models.MOMP_model import MOMP_model
from utils.dictionary_gen_utils import *
import matplotlib.pyplot as plt
from saved_data_loader import *

# --- Colors ---
color_real_MS ='green'#(1.0, 0.6, 0.6) #pastel_red
color_real_BS = 'green'
color_real='green'
color_nominal = 'purple'
color_OMP = 'blue'
color_MOMP = 'orange'
#%% functions: 
def NMSE(channel,channel_estimation):
    if channel.dim() == 3:
        channel = channel.unsqueeze(0)  # [1, Nbs, Nms, Nsub]
    if channel_estimation.dim() == 3:  
        channel_estimation = channel_estimation.unsqueeze(0)  # add batch dimension
    return torch.sum(torch.abs(channel-channel_estimation)**2,dim=(-3,-2,-1))/torch.sum(torch.abs(channel)**2,dim=(-3,-2,-1))

def model_estimation(Y, model, sigma2):
            H_est = torch.zeros_like(Y)
            for u in range(Y.shape[0]):
                for p in range(Y.shape[1]):
                    y = Y[u, p]
                    y = y.squeeze()

                    res, _, _ = model.forward(y, u, sigma2)
                    H_est[u, p] = y - res
            return H_est
#%%--------------------------------------- preprocessing ------------------------------------------------------------
Umax,Pmax=5,10
H=channels[:Umax,:Pmax] #emporarly 
Y=observations[:Umax,:Pmax] #emporarly 
nb_users=H.shape[0]
#------------------------------------  normalize channels  ----------------------------------------------------------
H_normalized = H / torch.sqrt(torch.sum(torch.abs(H)**2, dim=(-3, -2, -1), keepdim=True))
Y_normalized = Y / torch.sqrt(torch.sum(torch.abs(Y)**2, dim=(-3, -2, -1), keepdim=True))
#-------------------------------Get train, validation and test data -------------------------------------------------
train_test_ratio=0.8
tt_split_index=int(H_normalized.shape[1] * train_test_ratio)
H_aux=H_normalized[:,:tt_split_index].to(device)
Y_aux=Y_normalized[:,:tt_split_index].to(device)

# test data 
H_test=H_normalized[:,tt_split_index:].to(device)
Y_test=Y_normalized[:,tt_split_index:].to(device)

#train data
train_valid_ratio=0.8
tv_split_index = int(H_aux.shape[1] * train_valid_ratio)
H_train    = H_aux [:,:tv_split_index].to(device)
Y_train   = Y_aux [:,:tv_split_index].to(device)
# validation data 
H_val      = H_aux [:,tv_split_index:] # int(valid_size/U)].to(device)
Y_val     = Y_aux[:,tv_split_index:] # int(valid_size/U)].to(device)

#%% ----------------------------------- Deep unfolding ------------------------------------------
# parameters defining
# model defining
nominal_MS_ant_position_stacked = torch.stack([nominal_MS_ant_position.clone() for _ in range(nb_users)], dim=0)
unfolded_MOMP_model = MOMP_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff,nominal_MS_ant_position_stacked,
                 subcarriers, BS_DoA, MS_DoA, delays)
#optimizer
# optimizer = torch.optim.Adam(unfolded_MOMP_model.parameters(), lr=1e-4)
optimizer = torch.optim.Adam([
    {'params': unfolded_MOMP_model.BS_learnable_pos_y, 'lr':1e-4},
    {'params': unfolded_MOMP_model.BS_ant_gains, 'lr':1e-2},
    {'params': unfolded_MOMP_model.BS_coupling_coeff, 'lr':1e-2},
    {'params': unfolded_MOMP_model.MS_learnable_pos_list, 'lr':1e-4},
])
# scheduler= torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.9)

#%%--------------------------- evaluate model BEFORE training and model with real dictionary----------------------------------

real_dictionary_MOMP_model = MOMP_model(real_BS_ant_position, real_BS_gains, real_BS_coupling_coeff, real_MS_ant_position,
                 subcarriers, BS_DoA, MS_DoA, delays)

unfolded_MOMP_model.eval()
real_dictionary_MOMP_model.eval()
# --- Evaluate both models ---
with torch.no_grad():
    H_test_nominaldict = model_estimation(Y_test, unfolded_MOMP_model, sigma2)
    H_test_realdict  = model_estimation(Y_test, real_dictionary_MOMP_model, sigma2)

    # Compute NMSEs
    NMSE0=NMSE(H_test.reshape(-1, *H_test.shape[2:]), H_test_nominaldict.reshape(-1, *H_test_nominaldict.shape[2:]))
    NMSE_opt=NMSE(H_test.reshape(-1, *H_test.shape[2:]),H_test_realdict.reshape(-1, *H_test_realdict.shape[2:]))
#%%---------------------------------------training-----------------------------------------------
unfolded_MOMP_model.train()
nb_epochs = 10
# batch_size = 1 # batch size
train_losses_list, valid_losses_list = [], []
train_losses_list.append(NMSE(H_train,Y_train))
valid_losses_list.append(NMSE(H_val,Y_val))

best_loss=torch.inf
for i in tqdm(range(nb_epochs)):
    
    for user_idx in range(nb_users):
        Y_batched =   Y_train[user_idx].to(device)
        H_batched  =   H_train[user_idx].to(device)
        Y_batched=Y_batched.squeeze()
        H_batched=H_batched.squeeze()
        # if i==0 and b==0: TODO see if forward handles more than 1 channel
        #     with torch.no_grad():
        #         # train loss
        #         res,_,_ = unfolded_OMP_model.forward(Y_train, sigma2)
        #         train_losses.append(NMSE(H_train,Y_train-res))
        #         # validation loss
        #         res,_,_ = unfolded_OMP_model.forward(Y_val, sigma2)
        #         valid_losses.append(NMSE(H_val,Y_val-res))
        for i, p in enumerate(unfolded_MOMP_model.MS_learnable_pos_list):
            p.requires_grad_(i == user_idx)
    ################################## channel estimation #####################################################
        # res_batched, _,_ = unfolded_MOMP_model.forward(Y_batched,sigma2)
        res_batched=torch.stack([unfolded_MOMP_model.forward(Y_batched[p],user_idx,sigma2)[0] for p in range(len(Y_batched))], dim=0)
        H_est_batched=Y_batched-res_batched
        loss = torch.mean(NMSE(Y_batched,H_est_batched))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step() # Update the learning rate using the scheduler
    with torch.no_grad():
        # --- TRAIN ---
        H_est_train = model_estimation(Y_train, unfolded_MOMP_model, sigma2)
        train_loss = NMSE(
            H_train.reshape(-1, *H_train.shape[2:]),
            H_est_train.reshape(-1, *H_est_train.shape[2:])
        )
        train_losses_list.append(train_loss)

        # --- VALIDATION ---
        H_est_val = model_estimation(Y_val, unfolded_MOMP_model, sigma2)
        valid_loss = NMSE(
            H_val.reshape(-1, *H_val.shape[2:]),
            H_est_val.reshape(-1, *H_est_val.shape[2:])
        )
        valid_losses_list.append(valid_loss)

        # --- SAVE BEST ---
        if torch.mean(valid_loss) < best_loss:
            torch.save(unfolded_MOMP_model.state_dict(),'best_momp_model.pth')
            best_loss = torch.mean(valid_loss)
            best_epoch = i


    
# %%--------------- evaluate model after training ----------------------------
unfolded_MOMP_model.eval()
with torch.no_grad():
    H_test_MOMPnet = model_estimation(Y_test, unfolded_MOMP_model, sigma2)

    # Compute NMSEs
    NMSEZ=NMSE(H_test.reshape(-1, *H_test.shape[2:]), H_test_MOMPnet.reshape(-1, *H_test_MOMPnet.shape[2:]))


#%% Save data

# Save everything together in a dictionary
save_dict = {
    'model_state_dict': unfolded_MOMP_model.state_dict(),
    'NMSE0': NMSE0,
    'NMSEZ': NMSEZ,
    'train_losses': train_losses_list,
    'valid_losses': valid_losses_list
}

# Save to a file
torch.save(save_dict, 'MOMP_model_and_metrics.pth')

print("Model and lists saved successfully!")

################################################################################################################################################################################
##################################################################  plot evaluation #########################################################################################
################################################################################################################################################################################
#%% bar plot 

channels_idx = np.arange(1, len(NMSE0) + 1)
width = 0.2
NMSE0 = NMSE0.reshape(-1)  # shape: [num_channels]
NMSEZ = NMSEZ.reshape(-1)
NMSE_opt = NMSE_opt.reshape(-1)
plt.figure(figsize=(8, 5))
# Then detach and convert to numpy

bars1 = plt.bar(channels_idx - width, NMSE0.detach().cpu().numpy(), width,
                label='MOMP with nominal Dict', color=color_nominal)
bars2 = plt.bar(channels_idx , NMSEZ.detach().cpu().numpy(), width,
                label='unfolded MOMP', color=color_MOMP)
bars0 = plt.bar(channels_idx + width, NMSE_opt.detach().cpu().numpy(), width,
                label='MOMP with real Dict', color=color_real)
plt.semilogy()
plt.xticks(channels_idx)
plt.legend()
plt.show()

# %% 
# Plotting learning curve
# Convert list of tensors -> average NMSE per epoch
train_losses_avg = [t.mean().item() for t in train_losses_list]
valid_losses_avg = [v.mean().item() for v in valid_losses_list]

epochs = range(0, len(train_losses_avg))

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses_avg, label='Train NMSE', marker='o', color='blue')
plt.plot(epochs, valid_losses_avg, label='Validation NMSE', marker='s', color='orange')

plt.gca().spines['left'].set_position('zero')
plt.xlabel('Epoch')
# plt.xticks(epochs)
plt.ylabel('NMSE')
plt.title('Learning Curve')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

################################################################################################################################################################################
##################################################################  learned parameters #########################################################################################
################################################################################################################################################################################
#%%
learned_BS_pos=list(unfolded_MOMP_model.parameters())[0].detach().numpy()  # first parameter tensor
learned_gains=list(unfolded_MOMP_model.parameters())[1].detach().numpy()  # 2nd parameter tensor
learned_coupling=list(unfolded_MOMP_model.parameters())[2].detach().numpy()  # 3rd parameter tensor
learned_MS_pos=torch.stack([p.detach() for p in unfolded_MOMP_model.MS_learnable_pos_list], 0).cpu().numpy()  # 4th parameter tensor
nominal_BS_gains = np.asarray(BS_gains['nominal_BS_gains'])
nominal_BS_coupling_coeff = np.asarray(BS_coupling['nominal_BS_coupling_coeff'],dtype=np.complex128)
real_BS_ant_position = np.asarray(real_BS_ant_position)
nominal_BS_ant_position = np.asarray(nominal_BS_ant_position)
real_MS_ant_position = np.asarray(real_MS_ant_position)
nominal_MS_ant_position = np.asarray(nominal_MS_ant_position)
real_BS_gains = np.asarray(real_BS_gains)
#%%
####################################### plot learned BS postions #############################################

# --- X and Y coordinates ---
x = real_BS_ant_position[:, 0]
y_nominal = nominal_BS_ant_position[:, 1]
y_real = real_BS_ant_position[:, 1]
y_MOMP = learned_BS_pos

# --- Apply small horizontal offsets for visibility ---
offset = 0.03  # adjust if antennas are close
x_nominal = x - offset *1.5
x_real    = x - offset * 0.5
x_MOMP     = x + offset *0.5

# --- Plot ---
plt.figure(figsize=(6,5))

plt.scatter(x_nominal, y_nominal, label='Nominal BS', marker='x', color=color_nominal, s=50, linewidths=1)
plt.scatter(x_real, y_real, label='Real BS', color=color_real_BS, s=70, edgecolors='k', alpha=0.8)
plt.scatter(x_MOMP, y_MOMP, label='Learned BS (MOMP)', marker='d',color=color_MOMP, s=70, edgecolors='k', alpha=0.8)

# --- Optional: connect each antenna index with dotted lines ---
for i in range(len(x)):
    plt.plot([x_nominal[i], x_real[i], x_MOMP[i]],
             [y_nominal[i], y_real[i], y_MOMP[i]],
             color='gray', linestyle='--', alpha=0.4, linewidth=1)

# --- Labels and style ---
plt.title('Mobile Station Antenna Positions with real gains and mutual coupling', fontsize=14)
plt.xlabel('X-axis [m]')
plt.ylabel('Y-axis [m]')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
#%%
############################################ plot learned BS antenna Gains ####################################
# --- Prepare data for plotting ---

real_BS_gains_normalized = real_BS_gains / np.sqrt(np.sum((np.abs(real_BS_gains)**2)))
nominal_BS_gains_normalized = nominal_BS_gains / np.sqrt(np.sum((np.abs(nominal_BS_gains)**2)))
learned_gains_normalized= learned_gains / np.sqrt(np.sum((np.abs(learned_gains)**2)))

idx = np.arange(len(real_BS_gains_normalized))
mag_real = np.abs(real_BS_gains_normalized)
mag_nominal = np.abs(nominal_BS_gains_normalized)
mag_MOMP = np.abs(learned_gains_normalized)

phase_real = np.angle(real_BS_gains_normalized)
phase_nominal = np.angle(nominal_BS_gains_normalized)
phase_MOMP = np.angle(learned_gains_normalized)

# --- Plot magnitude comparison ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(idx, mag_real, 'o-', label='Real', color=color_real_BS)
plt.plot(idx, mag_nominal, 'x--', label='Nominal', color=color_nominal)
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
plt.plot(idx, phase_MOMP, 'd-', label='MOMP Learned', color=color_MOMP)
plt.title('Antenna Gain Phases')
plt.xlabel('Antenna Index')
plt.ylabel('Phase [rad]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
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
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()



# %% ALL BS parameters in one fig

from matplotlib.patches import Circle

def plot_antennas_with_parameters(
    ax, positions_y, gains, coupling_c1, color="C0", label=None,
    positions_scale=1.0,mag_scale=0.45, coupling_line_scale=1.0,

    circle_alpha=0.25, zorder_base=10, y_offset=0.0,
    center_marker=True, center_marker_size=20
):
    median_spacing = np.median(np.diff(positions_y))  # in λ
    mag_scale = mag_scale * median_spacing                 # circles ≈ 30% of spacing

    """Plot one set of antenna parameters (offset by y_offset)."""
    gains = np.asarray(gains).astype(np.complex128)
    positions_y=np.asarray(positions_y)
    coupling_c1=np.asarray(coupling_c1)
    N = positions_y.shape[0]
    positions = np.zeros((N,2))
    positions[:, 0]=positions_y*positions_scale
    assert gains.shape[0] == N, "gains must have same length as positions"

    # Apply vertical offset
    positions[:, 1] -= y_offset

    mags = np.abs(gains)
    phases = np.angle(gains)
    radii = (mags + 1e-8) * mag_scale

    # Plot antennas (circles + phase tick + optional center point)
    for pos, r, phi in zip(positions, radii, phases):
        circ = Circle(pos, radius=r, facecolor=color, alpha=circle_alpha,
                      edgecolor=color, linewidth=1.0, zorder=zorder_base + 2)
        ax.add_patch(circ)

        # phase tick
        vec = np.array([np.cos(phi), np.sin(phi)])
        end = pos + vec * r
        ax.plot([pos[0], end[0]], [pos[1], end[1]],
                color=color, linewidth=1.5, zorder=zorder_base + 3)

        # center marker
        if center_marker:
            ax.scatter(pos[0], pos[1], s=center_marker_size,
                       color=color, zorder=zorder_base + 4)

    # Coupling between adjacent antennas
    if N >= 2:
        c_abs, c_ang = np.abs(coupling_c1), np.angle(coupling_c1)
        for i in range(N - 1):
            # fixed tick length (absolute, not dependent on seg_vec)
            tick_len = coupling_line_scale * (c_abs + 1e-8)

            p0, p1 = positions[i], positions[i + 1]
            mid = 0.5 * (p0 + p1)
            seg_vec = p1 - p0
            seg_norm = np.linalg.norm(seg_vec)
            if seg_norm == 0:
                continue

            # unit vector along the segment
            seg_unit = seg_vec / seg_norm

            # rotate the unit vector by coupling phase to get tick direction
            rot = np.array([[np.cos(c_ang), -np.sin(c_ang)],
                            [np.sin(c_ang),  np.cos(c_ang)]])
            tick_dir = rot @ seg_unit  # now magnitude = 1

            # draw tick of fixed length
            start = mid - 0.5 * tick_len * tick_dir
            end   = mid + 0.5 * tick_len * tick_dir
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color='k', linewidth=1.8, zorder=zorder_base+4)


    if label is not None:
        ax.scatter([], [], color=color, alpha=0.9, label=label)
        


def plot_multiple_parameter_sets(
    list_of_positions, list_of_gains, list_of_c1,colors=None, labels=None, y_spacing=2.0, figsize=(10,6)
):
    """Plot multiple parameter sets with vertical offset."""
    assert len(list_of_positions) == len(list_of_gains) == len(list_of_c1) 

    fig, ax = plt.subplots(figsize=figsize)
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for k, (p, g, c1) in enumerate(zip(list_of_positions, list_of_gains, list_of_c1)):
        color = colors[k % len(colors)]
        label = None if labels is None else labels[k]
        y_offset = k * y_spacing
        plot_antennas_with_parameters(
            ax, p, g, c1, color=color, label=label,
            coupling_line_scale=1.0, circle_alpha=0.25,
            y_offset=y_offset, zorder_base=10+k,
            center_marker=True, center_marker_size=25
        )

    ax.set_aspect("equal", "box")
    # ax.set_xlabel("x (position)")
    ax.set_xticklabels(range(16))
    ax.set_yticklabels([])
    ax.set_title("BS Antenna parameters")
    ax.plot([],[],color='k',label='mutual coupling')


    if labels is not None:
        # ax.legend(loc="upper right")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    return fig, ax



l=2/lambda_
nominal_min=nominal_BS_ant_position[:,1].min()
scaled_positions = [(pos-nominal_min) * l for pos in [real_BS_ant_position[:,1], learned_BS_pos, nominal_BS_ant_position[:,1]]]
fig, ax = plot_multiple_parameter_sets(
    scaled_positions,
    [real_BS_gains, learned_gains, nominal_BS_gains],
    [real_BS_coupling_coeff, learned_coupling, nominal_BS_coupling_coeff],
    colors=[color_real,color_MOMP,color_nominal],labels=["Real ", "Learned", "Nominal"],
    y_spacing=2.0,
    figsize=(15,8)
)
plt.show()


# %%
