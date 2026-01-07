# importing libraries
import torch
from tqdm import tqdm
from pathlib import Path
from models.MOMP_model import MOMP_model
from utils.dictionary_gen_utils import *
from saved_data_loader import *
from utils.training_utils import *
#
def ticklabels_array(highest_int, spacing):
    result = []
    for i in range(highest_int + 1):
        result.append(str(i))
        if i < highest_int:
            result.extend([""] * (spacing-1))
    return result

#--------------------------------------- preprocessing ------------------------------------------------------------
Umax=10
Pmax=150
SNR_av=5
print(f"average SNR={SNR_av}")
sigma2=sigma2_dict[SNR_av]
H=channels[:Umax,:Pmax] #dataset size
Y=observations_dict[SNR_av][:Umax,:Pmax] #dataset size
nb_users=H.shape[0]
#------------------------------------  normalize channels  ----------------------------------------------------------
H_normalized = normalize(H)
Y_normalized = normalize(Y)
#-------------------------------Get train and validation data -------------------------------------------------
train_val_ratio=2/3
tt_split_index=int(H_normalized.shape[1] * train_val_ratio)
H_train=H_normalized[:,:tt_split_index].to(device)
Y_train=Y_normalized[:,:tt_split_index].to(device)

# Validation data 
H_val=H_normalized[:,tt_split_index:].to(device)
Y_val=Y_normalized[:,tt_split_index:].to(device)

################################ MOMP with real dictionary ################################################################
H_val_realdict=MOMP_estimation(Y_val,real_BS_Dictionary,real_MS_Dictionaries,FRV_Dictionary,sigma2)
NMSE_real=NMSE(H_val,H_val_realdict)

H_val_nominaldict=MOMP_estimation(Y_val,nominal_BS_Dictionary,nominal_MS_Dictionary,FRV_Dictionary,sigma2)
NMSE_nominal=NMSE(H_val,H_val_nominaldict)

# ----------------------------------- Deep unfolding ------------------------------------------
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

#--------------------------- evaluate model BEFORE training and model with real dictionary----------------------------------
MOMPnet.eval()
# --- Evaluate both models ---
with torch.no_grad():
    H_val_nominaldict = model_estimation(Y_val, MOMPnet, sigma2)
    # Compute NMSEs
    NMSE_nominal=NMSE(H_val, H_val_nominaldict)
#---------------------------------------training-----------------------------------------------
MOMPnet.train()
nb_epochs = 8
# batch_size = 1 # batch size
train_NMSE_list, MOMPnet_NMSE_list = [], []
with torch.no_grad():
    # --- TRAIN ---
    H_est_train = model_estimation(Y_train, MOMPnet, sigma2)
    train_NMSE_list.append(NMSE(H_train,H_est_train))
    # --- VALIDATION ---
    H_est_val = model_estimation(Y_val, MOMPnet, sigma2)
    MOMPnet_NMSE_list.append(NMSE(H_val,H_est_val))


epoch_bar = tqdm(range(nb_epochs), desc="Epochs", position=0, smoothing=0.1)

for epoch in epoch_bar:
    for user_idx in range(nb_users):
        epoch_bar.set_postfix(user=user_idx)
        optimizer.zero_grad()
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

    # Save so far trained model
    save_dict = {
        'model_state_dict': MOMPnet.state_dict(),
        'train_NMSE': train_NMSE_list,
        'MOMPnet_NMSE': MOMPnet_NMSE_list
    }
    # Save to a file
    torch.save(save_dict, f'MOMPnet_{SNR_av}_dB_new.pth')

#Save data
save_dict = {
    'model_state_dict': MOMPnet.state_dict(),
    'train_NMSE': train_NMSE_list,
    'MOMPnet_NMSE': MOMPnet_NMSE_list,
    'nominal_NMSE': NMSE_nominal,
    'real_NMSE': NMSE_real
}
# Save to a file
torch.save(save_dict, f'MOMPnet_{SNR_av}_dB_new.pth')
print("Model saved!")