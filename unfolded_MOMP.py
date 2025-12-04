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
#%%
def MOMP(Y,D1,D2,D3, sigma2_est, iter_max=30, refine_iter=2):
        """
        Performs MOMP to approximate
        the channel H using dictionaries D1, D2, and D3.

        Parameters
        ----------
        H : torch.Tensor
            Observed channel tensor to be approximated.
        sigma2_est : float
            Estimated noise variance for the stopping criterion.
        iter_max : int, optional
            Maximum number of iterations (default: 30).
        refine_iter : int, optional
            Number of refinement steps to improve atom selection (default: 2).

        Returns
        -------
        r : torch.Tensor
            Final residual tensor after MOMP iterations.
        I : torch.Tensor
            Indices of selected atoms [i1, i2, i3] at each iteration.
        x : torch.Tensor
            Coefficients corresponding to the selected atoms.

        """
        D1 = D1.to(dtype=torch.complex128)
        D2 = D2.to(dtype=torch.complex128)
        D3 = D3.to(dtype=torch.complex128)

        # --------------------------------------------------------------------------
        # Initialization
        # --------------------------------------------------------------------------
        N = Y.numel()              # Total number of elements in H
        stop = False
        iter = 0
        I_list = []                # List of selected index triplets per iteration
        h_reshaped = Y.reshape(-1) # Flattened channel tensor
        D_I_list = []              # List of selected atoms
        r = Y                      # Initialize residual with input tensor

        # --------------------------------------------------------------------------
        # Main MOMP iteration loop
        # --------------------------------------------------------------------------
        while not stop:
            # Step 1: Compute correlations with D1 along the first dimension
            corr1 = torch.einsum('ab,bms->ams', torch.conj(D1).T, r)
            i1 = torch.argmax((corr1.abs()**2).sum(dim=(1, 2)))

            # Step 2: Select best atom from D2 using the previous selection from D1
            corr2 = torch.conj(D2).T @ corr1[i1]
            i2 = torch.argmax((corr2.abs()**2).sum(dim=1))

            # Step 3: Select best atom from D3 using the previous selections
            corr3 = torch.conj(D3).T @ corr2[i2]
            i3 = torch.argmax(torch.abs(corr3)**2)

            # ----------------------------------------------------------------------
            # Optional refinement of atom indices via local coordinate updates
            # ----------------------------------------------------------------------
            if refine_iter is not None:
                atom = [i1, i2, i3]
                D = [D1, D2, D3]

                for _ in range(refine_iter):
                    for d in range(len(atom)):
                        # Identify the two remaining dimensions besides d
                        other_idx1, other_idx2 = (set(range(len(atom))) - {d})

                        # Extract the selected atoms along the other dimensions
                        vec_0 = D[other_idx1][:, atom[other_idx1]]
                        vec_1 = D[other_idx2][:, atom[other_idx2]]

                        # Permute residual to align dimensions [other1, other2, d]
                        r_permuted = r.permute(other_idx1, other_idx2, d)

                        # Compute correlation along dimension d
                        corr_d = torch.einsum('a,abc,b->c', torch.conj(vec_0), r_permuted, torch.conj(vec_1))
                        corr_d = torch.matmul(torch.conj(D[d]).T, corr_d)

                        # Update atom index along dimension d with the highest correlation
                        i_d = torch.argmax(torch.abs(corr_d)**2)
                        atom[d] = i_d

                # Update selected indices after refinement
                i1, i2, i3 = atom

            # Store current triplet of selected atom indices
            I_list.append(torch.tensor([i1, i2, i3], device=device))

            # ----------------------------------------------------------------------
            # Construct current dictionary from selected atoms (Kronecker structure)
            # ----------------------------------------------------------------------
            vec1 = D1[:, i1]
            vec2 = D2[:, i2]
            vec3 = D3[:, i3]
            D_I_list.append(torch.kron(torch.kron(vec1, vec2), vec3))
            D_I = torch.stack(D_I_list, 1)

            # ----------------------------------------------------------------------
            # Solve least squares problem to estimate coefficients
            # ----------------------------------------------------------------------
            x = torch.linalg.lstsq(D_I, h_reshaped).solution
            proj_h = D_I @ x

            # ----------------------------------------------------------------------
            # Update residual and iteration counter
            # ----------------------------------------------------------------------
            r_reshaped = h_reshaped - proj_h
            r = r_reshaped.reshape(Y.shape)
            iter += 1

            # ----------------------------------------------------------------------
            # Check stopping criteria: residual energy or iteration limit
            # ----------------------------------------------------------------------
            if sigma2_est is None:
                SC=False
            else:
                SC=torch.sum(torch.abs(r)**2) <= N * sigma2_est

            if  SC or iter > iter_max - 1:
                stop = True

        # Stack selected atom indices across all iterations
        I = torch.stack(I_list, 0)

        return r, I, x

def MOMP_estimation(Y, D1, D2, D3, sigma2_est):
            H_est = torch.zeros_like(Y)
            for u in range(Y.shape[0]):
                if D2.dim()==2:
                    D2u=D2
                else:
                    D2u=D2[u]
                for p in range(Y.shape[1]):
                    y = Y[u, p]
                    y = y.squeeze()
                    res, _, _ = MOMP(y,D1,D2u,D3, sigma2_est)
                    H_est[u, p] = y - res
            return H_est

#%%--------------------------------------- preprocessing ------------------------------------------------------------
Umax,Pmax=5,10
# Umax,Pmax=10,10
H=channels[:Umax,:Pmax] #temporarily 
Y=observations[:Umax,:Pmax] #temporarily 
nb_users=H.shape[0]
#------------------------------------  normalize channels  ----------------------------------------------------------
H_normalized = normalize(H)
Y_normalized = normalize(Y)
#-------------------------------Get train and validation data -------------------------------------------------
train_val_ratio=0.8
tt_split_index=int(H_normalized.shape[1] * train_val_ratio)
H_train=H_normalized[:,:tt_split_index].to(device)
Y_train=Y_normalized[:,:tt_split_index].to(device)

# Validation data 
H_val=H_normalized[:,tt_split_index:].to(device)
Y_val=Y_normalized[:,tt_split_index:].to(device)


#%%############################### MOMP with real dictionary ################################################################
H_val_realdict=MOMP_estimation(Y_val,real_BS_Dictionary,real_MS_Dictionaries,FRV_Dictionary,sigma2)
NMSE_real=NMSE(H_val,H_val_realdict)

H_val_nominaldict=MOMP_estimation(Y_val,nominal_BS_Dictionary,nominal_MS_Dictionary,FRV_Dictionary,sigma2)
NMSE_nominal=NMSE(H_val,H_val_nominaldict)
#%% ----------------------------------- Deep unfolding ------------------------------------------
# parameters defining
# model defining
nominal_MS_ant_position_stacked = torch.stack([nominal_MS_ant_position.clone() for _ in range(nb_users)], dim=0) #!!!
MOMPnet = MOMP_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff,nominal_MS_ant_position_stacked,
                 subcarriers, BS_DoA, MS_DoA, delays)
#optimizer
# optimizer = torch.optim.Adam(unfolded_MOMP_model.parameters(), lr=1e-4)
optimizer = torch.optim.Adam([
    {'params': MOMPnet.BS_learnable_pos_y, 'lr':1e-3},
    {'params': MOMPnet.BS_ant_gains, 'lr':1e-2},
    {'params': MOMPnet.BS_coupling_coeff, 'lr':1e-2},
    {'params': MOMPnet.MS_learnable_pos_list, 'lr':1e-3},
])
# scheduler= torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.9)

#%%--------------------------- evaluate model BEFORE training and model with real dictionary----------------------------------
MOMPnet.eval()
# --- Evaluate both models ---
with torch.no_grad():
    H_val_nominaldict = model_estimation(Y_val, MOMPnet, sigma2)
    # Compute NMSEs
    NMSE_nominal=NMSE(H_val, H_val_nominaldict)
#%%---------------------------------------training-----------------------------------------------
MOMPnet.train()
nb_epochs = 3
# batch_size = 1 # batch size
train_NMSE_list, MOMPnet_NMSE_list = [], []
with torch.no_grad():
    # --- TRAIN ---
    H_est_train = model_estimation(Y_train, MOMPnet, sigma2)
    train_NMSE_list.append(NMSE(H_train,H_est_train))
    # --- VALIDATION ---
    H_est_val = model_estimation(Y_val, MOMPnet, sigma2)
    MOMPnet_NMSE_list.append(NMSE(H_val,H_est_val))

best_loss=torch.inf
for i in tqdm(range(nb_epochs)):
    
    for user_idx in range(nb_users):
        Y_batched =   Y_train[user_idx].to(device)
        H_batched  =   H_train[user_idx].to(device)
        Y_batched=Y_batched.squeeze()
        H_batched=H_batched.squeeze()

        for i, p in enumerate(MOMPnet.MS_learnable_pos_list):
            p.requires_grad_(i == user_idx)
    ################################## channel estimation #####################################################

        res_batched=torch.stack([MOMPnet.forward(Y_batched[p],user_idx,sigma2)[0] for p in range(len(Y_batched))], dim=0)
        H_est_batched=Y_batched-res_batched
        loss = torch.mean(NMSE(Y_batched,H_est_batched))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step() # Update the learning rate using the scheduler
    with torch.no_grad():
        # --- TRAIN ---
        H_est_train = model_estimation(Y_train, MOMPnet, sigma2)
        train_NMSE_list.append(NMSE(H_train,H_est_train))

        # --- VALIDATION ---
        H_est_val = model_estimation(Y_val, MOMPnet, sigma2)
        MOMPnet_NMSE_list.append(NMSE(H_val,H_est_val))

        # # --- SAVE BEST --- #TODO




#%% Save data

## Save everything together in a dictionary
# save_dict = {
#     'model_state_dict': MOMPnet.state_dict(),
#     'NMSE_nominal': NMSE_nominal,
#     'NMSE_real': NMSE_real,
#     'train_losses': train_NMSE_list,
#     'valid_losses': MOMPnet_NMSE_list
# }

# # Save to a file
# torch.save(save_dict, 'MOMP_model_and_metrics.pth')

# print("Model and lists saved successfully!")

#%%#############################################################################################################################################################################
##################################################################  plot evaluation ############################################################################################
################################################################################################################################################################################
 
#--------------------------------------------------Plotting learning curve------------------------------------------------------------------
# Convert list of tensors -> average NMSE per epoch
train_NMSE_avg = [t.mean().item() for t in train_NMSE_list]
MOMPnet_NMSE_avg = [v.mean().item() for v in MOMPnet_NMSE_list]

epochs = range(0, len(train_NMSE_avg))

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_NMSE_avg, label='Train NMSE', marker='o', color='blue')
plt.plot(epochs, MOMPnet_NMSE_avg, label='Validation NMSE', marker='s', color='orange')

plt.gca().spines['left'].set_position('zero')
plt.xlabel('Epoch')
# plt.xticks(epochs)
plt.ylabel('NMSE')
plt.title('Learning Curve')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
#%%--------------------------------------------------Plotting validation NMSE------------------------------------------------------------------
# Labels and colors
labels = [
    'Observation error',
    'MOMP with nominal Dicts',
    'MOMPnet',
    'MOMP with real Dicts'
]
colors = [color_observation, color_nominal, color_MOMP, color_real]

# bar plot for last value
nmse_obs=NMSE(H_val,Y_val).mean().item()
nmse_nominal = NMSE_nominal.mean().item()
nmse_MOMPnet = MOMPnet_NMSE_avg[-1]
nmse_real = NMSE_real.mean().item()

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

#plot throughout training 

MOMPnet_NMSE_arr=np.array(MOMPnet_NMSE_avg)
means_arr=np.vstack([np.full_like(MOMPnet_NMSE_arr, nmse_obs),np.full_like(MOMPnet_NMSE_arr, nmse_nominal), MOMPnet_NMSE_arr,np.full_like(MOMPnet_NMSE_arr, nmse_real)])
# NMSE means vs SNR OR vs Dataset size

colors = [ color_observation,color_nominal, color_MOMP, color_real]
markers = ['o','s','^', 'x']
linestyles=['--','-','-','--']
P=50
plt.figure(figsize=(8, 5))

# means_arr must have shape (5, len(nb_obs_list))
# one row per method

for i in range(len(labels)):
    plt.plot(epochs,
             means_arr[i],
             color=colors[i],
             label=labels[i],
             marker=markers[i], linestyle=linestyles[i])

# plt.yscale('log')
ticks = epochs
plt.xticks(ticks=ticks, labels=ticks)
plt.xlabel(r'Number of seen channels ($10^3$)')
plt.xlim(left=0)
plt.ylabel('NMSE (mean)')
plt.title('Mean NMSE vs number of seen channels')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("LUC.pdf", bbox_inches="tight")
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

real_BS_gains_normalized = real_BS_gains / np.sqrt(np.sum((np.abs(real_BS_gains)**2)))
nominal_BS_gains_normalized = nominal_BS_gains / np.sqrt(np.sum((np.abs(nominal_BS_gains)**2)))
learned_gains_normalized= learned_gains / np.sqrt(np.sum((np.abs(learned_gains)**2)))

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
    plt.grid(True, linestyle='--', alpha=0.4)
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

#%%
# from saved_data_loader import *
# # LOAD TRAINED MODELS
# ############################ MOMP ############################
# Umax=5
# Pmax=100
# nominal_MS_ant_position_stacked = torch.stack([nominal_MS_ant_position.clone() for _ in range(Umax)], dim=0)
# unfolded_MOMP_model = MOMP_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff, nominal_MS_ant_position_stacked,
#                  subcarriers, BS_DoA, MS_DoA, delays)  # replace with your model class
# # Load everything
# checkpoint = torch.load('.saved_data/.saved_models/MOMP_5100.pth')
# # checkpoint = torch.load('MOMP_model_and_metrics.pth')
# # Load model weights
# unfolded_MOMP_model.load_state_dict(checkpoint['model_state_dict'])
# # Load the lists
# NMSE_nominal = checkpoint['NMSE0']
# NMSE_MOMP = checkpoint['NMSEZ']
# # NMSE_real = checkpoint['NMSE_real']
# train_losses_list = checkpoint['train_losses']
# valid_losses_list = checkpoint['valid_losses']

# learned_BS_pos=list(unfolded_MOMP_model.parameters())[0].detach().numpy()  # first parameter tensor
# learned_gains=list(unfolded_MOMP_model.parameters())[1].detach().numpy()  # 2nd parameter tensor
# learned_coupling=list(unfolded_MOMP_model.parameters())[2].detach().numpy()  # 3rd parameter tensor
# learned_MS_pos=torch.stack([p.detach() for p in unfolded_MOMP_model.MS_learnable_pos_list], 0).cpu().numpy()  # 4th parameter tensor
# nominal_BS_gains = np.asarray(BS_gains['nominal_BS_gains'])
# nominal_BS_coupling_coeff = np.asarray(BS_coupling['nominal_BS_coupling_coeff'],dtype=np.complex128)
# real_BS_ant_position = np.asarray(real_BS_ant_position)
# nominal_BS_ant_position = np.asarray(nominal_BS_ant_position)
# real_MS_ant_position = np.asarray(real_MS_ant_position)
# nominal_MS_ant_position = np.asarray(nominal_MS_ant_position)
# real_BS_gains = np.asarray(real_BS_gains)
