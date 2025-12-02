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



#%%--------------------------------------- preprocessing ------------------------------------------------------------
Umax,Pmax=5,10
# Umax,Pmax=10,10
H=channels[:Umax,:Pmax] #temporarily 
Y=observations[:Umax,:Pmax] #temporarily 
nb_users=H.shape[0]
#------------------------------------  normalize channels  ----------------------------------------------------------
H_normalized = normalize(H)
Y_normalized = normalize(Y)
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
nominal_MS_ant_position_stacked = torch.stack([nominal_MS_ant_position.clone() for _ in range(nb_users)], dim=0) #!!!
unfolded_MOMP_model = MOMP_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff,nominal_MS_ant_position_stacked,
                 subcarriers, BS_DoA, MS_DoA, delays)
#optimizer
# optimizer = torch.optim.Adam(unfolded_MOMP_model.parameters(), lr=1e-4)
optimizer = torch.optim.Adam([
    {'params': unfolded_MOMP_model.BS_learnable_pos_y, 'lr':1e-3},
    {'params': unfolded_MOMP_model.BS_ant_gains, 'lr':1e-2},
    {'params': unfolded_MOMP_model.BS_coupling_coeff, 'lr':1e-2},
    {'params': unfolded_MOMP_model.MS_learnable_pos_list, 'lr':1e-3},
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
    NMSE_nominal=NMSE(H_test.reshape(-1, *H_test.shape[2:]), H_test_nominaldict.reshape(-1, *H_test_nominaldict.shape[2:]))
    NMSE_real=NMSE(H_test.reshape(-1, *H_test.shape[2:]),H_test_realdict.reshape(-1, *H_test_realdict.shape[2:]))
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

        # # --- SAVE BEST --- #TODO
        # if torch.mean(valid_loss) < best_loss:
        #     torch.save(unfolded_MOMP_model.state_dict(),'best_momp_model.pth')
        #     best_loss = torch.mean(valid_loss)
        #     best_epoch = i


    
# %%--------------- evaluate model after training ----------------------------
unfolded_MOMP_model.eval()
with torch.no_grad():
    H_test_MOMPnet = model_estimation(Y_test, unfolded_MOMP_model, sigma2)
    # Compute NMSEs
    NMSE_MOMP=NMSE(H_test.reshape(-1, *H_test.shape[2:]), H_test_MOMPnet.reshape(-1, *H_test_MOMPnet.shape[2:]))


#%% Save data

# Save everything together in a dictionary
save_dict = {
    'model_state_dict': unfolded_MOMP_model.state_dict(),
    'NMSE0': NMSE_nominal,
    'NMSEZ': NMSE_MOMP,
    'NMSE_real': NMSE_real,
    'train_losses': train_losses_list,
    'valid_losses': valid_losses_list
}

# Save to a file
torch.save(save_dict, 'MOMP_model_and_metrics.pth')

print("Model and lists saved successfully!")

#%%#############################################################################################################################################################################
##################################################################  plot evaluation ############################################################################################
################################################################################################################################################################################
 
#--------------------------------------------------Plotting learning curve------------------------------------------------------------------
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
#%%--------------------------------------------------Plotting testing NMSE------------------------------------------------------------------

# Filter and slice data

nmse0=NMSE(H_test,Y_test)
nmse1 = NMSE_nominal
nmse2 = NMSE_MOMP
nmse3 = NMSE_real

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
    nmse3.mean().item()
]

# Labels and colors
labels = [
    'Observation error',
    'MOMP with nominal Dicts',
    'MOMPnet',
    'MOMP with real Dicts'
]
colors = [color_observation, color_nominal, color_MOMP, color_real]
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
