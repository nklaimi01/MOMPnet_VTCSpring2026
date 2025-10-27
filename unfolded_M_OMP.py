#%% importing libraries
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from models.MOMP_model import MOMP_model
from utils.dictionary_gen_utils import *
import matplotlib.pyplot as plt
from saved_data_loader import *

#%% functions: 
def NMSE(channel,channel_estimation):
    if channel.dim() == 3:
        channel = channel.unsqueeze(0)  # [1, Nbs, Nms, Nsub]
    if channel_estimation.dim() == 3:  
        channel_estimation = channel_estimation.unsqueeze(0)  # add batch dimension
    return torch.sum(torch.abs(channel-channel_estimation)**2,dim=(-3,-2,-1))/torch.sum(torch.abs(channel)**2,dim=(-3,-2,-1))

#%%--------------------------------------- preprocessing ------------------------------------------------------------
H=channels[0]
Y=observations[0]
#------------------------------------  normalize channels  ----------------------------------------------------------
H_normalized = H / torch.sqrt(torch.sum(torch.abs(H)**2, dim=(-3, -2, -1), keepdim=True))
Y_normalized = Y / torch.sqrt(torch.sum(torch.abs(Y)**2, dim=(-3, -2, -1), keepdim=True))
#-------------------------------Get train, validation and test data -------------------------------------------------
train_test_ratio=0.8
tt_split_index=int(len(H_normalized) * train_test_ratio)
H_aux=H_normalized[:tt_split_index].to(device)
Y_aux=Y_normalized[:tt_split_index].to(device)

# test data 
H_test=H_normalized[tt_split_index:].to(device)
Y_test=Y_normalized[tt_split_index:].to(device)

#train data
train_valid_ratio=0.8
tv_split_index = int(len(H_aux) * train_valid_ratio)
H_train    = H_aux [:tv_split_index].to(device)
Y_train   = Y_aux [:tv_split_index].to(device)
# validation data 
H_val      = H_aux [tv_split_index:] # int(valid_size/U)].to(device)
Y_val     = Y_aux[tv_split_index:] # int(valid_size/U)].to(device)

#%% ----------------------------------- Deep unfolding ------------------------------------------
# parameters defining
# model defining
unfolded_MOMP_model = MOMP_model(real_BS_ant_position, real_BS_gains, nominal_BS_coupling_coeff,real_MS_ant_position[0],
                 subcarriers, BS_DoA, MS_DoA, delays)
#optimizer
# optimizer = torch.optim.Adam(unfolded_MOMP_model.parameters(), lr=1e-4)
optimizer = torch.optim.Adam([
    {'params': unfolded_MOMP_model.BS_learnable_pos_y, 'lr':0},#1e-4},
    {'params': unfolded_MOMP_model.BS_ant_gains, 'lr':0},#1e-2},
    {'params': unfolded_MOMP_model.BS_coupling_coeff, 'lr':1e-2},
    {'params': unfolded_MOMP_model.MS_learnable_pos_y, 'lr':0},# 1e-4},
])
scheduler= torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.9)

#%%
NMSE0,NMSEZ=[],[]
##%%-----------------------evaluating model BEFORE training--------------------------------------
for i in range(len(H_test[:10])):
    res_0,I,x = unfolded_MOMP_model.forward(Y_test[i],sigma2)
    NMSE0.append(NMSE(H_test[i],Y_test[i]-res_0))

#%%---------------------------------------training-----------------------------------------------
H_train=H_train[:30] #TEMPORARLY
Y_train=Y_train[:30] #TEMPORARLY
nb_epochs = 10
batch_size = 1 # batch size
train_losses_list, valid_losses_list = [], []
train_losses_list.append(NMSE(H_train,Y_train))
valid_losses_list.append(NMSE(H_val,Y_val))

best_loss=torch.inf
for i in tqdm(range(nb_epochs)):
    
    for b in range(0, len(H_train), batch_size):
        Y_batched =   Y_train[b:b+batch_size].to(device)
        H_batched  =   H_train[b:b+batch_size].to(device)
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
    ################################## channel estimation #####################################################
        res_batched, _,_ = unfolded_MOMP_model.forward(Y_batched,sigma2)
        H_est_b=Y_batched-res_batched
        loss = NMSE(Y_batched,H_est_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() # Update the learning rate using the scheduler
    with torch.no_grad():        
        est_train_list, est_val_list = [], []
        # -------- TRAIN LOSS --------
        for r in range(len(H_train)):
            h_train = H_train[r]
            y_train = Y_train[r]
            # Ensure proper shape (remove singleton dims if needed)
            if y_train.ndim > 3:
                y_train = y_train.squeeze()
            if h_train.ndim > 3:
                h_train = h_train.squeeze()
            # Forward pass
            res_train, _, _ = unfolded_MOMP_model.forward(y_train, sigma2)
            # Store residual
            est_train_list.append(h_train - res_train)
        # -------- VALIDATION LOSS --------
        for r in range(len(H_val)):
            h_val = H_val[r]
            y_val = Y_val[r]
            if y_val.ndim > 3:
                y_val = y_val.squeeze()
            if h_val.ndim > 3:
                h_val = h_val.squeeze()
            res_val, _, _ = unfolded_MOMP_model.forward(y_val, sigma2)
            est_val_list.append(h_val - res_val)  


        # Stack results into tensors
        H_est_train = torch.stack(est_train_list, 0)
        H_est_val = torch.stack(est_val_list, 0)

        # Compute NMSE and store
        train_losses_list.append(NMSE(H_train, H_est_train))
        valid_losses_list.append(NMSE(H_val, H_est_val))

        # # train loss
        # res,_,_ = unfolded_OMP_model.forward(Y_train, sigma2)
        # train_losses.append(NMSE(H_train,Y_train-res))
        # # validation loss
        # res,_,_ = unfolded_OMP_model.forward(Y_val, sigma2)
        # valid_loss=NMSE(H_val,Y_val-res)
        # valid_losses.append(valid_loss)
        # if valid_loss<best_loss:
        #     torch.save(unfolded_OMP_model,save_dir/f'best_omp_model.pth')
        #     best_loss=valid_loss
        #     best_epoch=i+1



    #--------------- evaluate best model ----------------------------


    
# %%--------------- evaluate model after training ----------------------------
for i in range(len(H_test[:10])):
    res_0,I,x = unfolded_MOMP_model.forward(Y_test[i],sigma2)
    NMSEZ.append(NMSE(H_test[i],Y_test[i]-res_0))
# %%--------------- evaluate perfect model (no impairements) ----------------------------
NMSE_opt=[]
optimal_unf_MOMP_model = MOMP_model(real_BS_ant_position, real_BS_gains, real_BS_coupling_coeff, real_MS_ant_position[0],
                 subcarriers, BS_DoA, MS_DoA, delays)
for i in range(len(H_test[:10])):
    res_0,I,x = optimal_unf_MOMP_model.forward(Y_test[i],sigma2)
    NMSE_opt.append(NMSE(H_test[i],Y_test[i]-res_0))
#%% bar plot 

channels_idx = np.arange(1, len(NMSE0) + 1)
width = 0.2
NMSE0_tensor = torch.stack(NMSE0).reshape(-1)  # shape: [num_channels]
NMSEZ_tensor = torch.stack(NMSEZ).reshape(-1)
NMSE_opt_tensor = torch.stack(NMSE_opt).reshape(-1)
plt.figure(figsize=(8, 5))
# Then detach and convert to numpy

bars1 = plt.bar(channels_idx - width, NMSE0_tensor.detach().cpu().numpy(), width,
                label='MOMP with nominal Dict', color='red')
bars2 = plt.bar(channels_idx , NMSEZ_tensor.detach().cpu().numpy(), width,
                label='unfolded MOMP', color='green')
bars0 = plt.bar(channels_idx + width, NMSE_opt_tensor.detach().cpu().numpy(), width,
                label='MOMP with real Dict', color='skyblue')
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
#%%
learned_BS_pos=list(unfolded_MOMP_model.parameters())[0].detach().numpy()  # first parameter tensor
learned_gains=list(unfolded_MOMP_model.parameters())[1].detach().numpy()  # first parameter tensor
learned_coupling=list(unfolded_MOMP_model.parameters())[2].detach().numpy()  # first parameter tensor
learned_MS_pos=list(unfolded_MOMP_model.parameters())[3].detach().numpy()  # first parameter tensor
#%%
####################################### plot learned BS postions #############################################
# --- Colors ---
color_real_MS =(1.0, 0.6, 0.6) #pastel_red
color_real_BS = 'orange'
color_nominal = 'black'
color_OMP = 'red'
color_MOMP = 'green'

# --- X and Y coordinates ---
x = real_BS_ant_position[:, 0].cpu().numpy()
y_nominal = nominal_BS_ant_position[:, 1].cpu().numpy()
y_real = real_BS_ant_position[:, 1].cpu().numpy()
y_MOMP = torch.tensor(learned_BS_pos).cpu().numpy()

# --- Apply small horizontal offsets for visibility ---
offset = 0.03  # adjust if antennas are close
x_nominal = x - offset *1.5
x_real    = x - offset * 0.5
x_MOMP     = x + offset *0.5

# --- Plot ---
plt.figure(figsize=(6,5))

plt.scatter(x_nominal, y_nominal, label='Nominal BS', marker='x', color=color_nominal, s=50, linewidths=1)
plt.scatter(x_real, y_real, label='Real BS', color=color_real_BS, s=70, edgecolors='k', alpha=0.8)
plt.scatter(x_MOMP, y_MOMP, label='Learned BS (MOMP)', color=color_MOMP, s=70, edgecolors='k', alpha=0.8)

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
BS_gains = np.load(save_dir/'BS_gains.npz')
nominal_BS_gains = torch.from_numpy(BS_gains['nominal_BS_gains']).to(device)
real_BS_gains_normalized = real_BS_gains / torch.sqrt(torch.sum((torch.abs(real_BS_gains)**2)))
nominal_BS_gains_normalized = nominal_BS_gains / torch.sqrt(torch.sum((torch.abs(nominal_BS_gains)**2)))
learned_gains_normalized= learned_gains / np.sqrt(np.sum((np.abs(learned_gains)**2)))

idx = np.arange(len(real_BS_gains_normalized))
mag_real = torch.abs(real_BS_gains_normalized).cpu()
mag_nominal = torch.abs(nominal_BS_gains_normalized).cpu()
mag_MOMP = torch.abs(torch.tensor(learned_gains_normalized)).cpu()

phase_real = torch.angle(real_BS_gains_normalized).cpu()
phase_nominal = torch.angle(nominal_BS_gains_normalized).cpu()
phase_MOMP = torch.angle(torch.tensor(learned_gains_normalized)).cpu()

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
y_MOMP = torch.tensor(learned_MS_pos).cpu().numpy()

# --- Apply small horizontal offsets for visibility ---
offset = 0.02  # adjust if antennas are close
x_nominal = x - offset * 1.5
x_real    = x - offset * 0.5
x_MOMP    = x + offset * 0.5

# --- Plot ---
plt.figure(figsize=(8,5))

plt.scatter(x_nominal, y_nominal, label='Nominal MS', marker='x', color=color_nominal, s=50, linewidths=1)
plt.scatter(x_real, y_real, label='Real MS', color=color_real_MS, s=70, edgecolors='k', alpha=0.8)
plt.scatter(x_MOMP, y_MOMP, label='Learned MS (MOMP)', color=color_MOMP, s=70, edgecolors='k', alpha=0.8)

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



# %%
