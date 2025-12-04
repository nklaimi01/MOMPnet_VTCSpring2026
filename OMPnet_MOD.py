#%% importing libraries
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from models.OMP_model import OMP_1D_model
from utils.dictionary_gen_utils import *
import matplotlib.pyplot as plt
from saved_data_loader import *
from utils.training_utils import *

#%% ---------------------------------------------DATA PREPROCESSING-------------------------------------------------
# Umax,Pmax=5,100
# H=channels[:Umax,:Pmax] #([Umax,Pmax, 16, 8, 128])
# Y=observations[:Umax,:Pmax] #temporarily 
# #------------------------------------  normalize channels  ----------------------------------------------------------
# H_normalized = normalize(H)
# Y_normalized = normalize(Y)
# #------------------------------------ reshape observations ----------------------------------------------------------
# m=2 #dimensions des antennes de la BS #([Umax,Pmax, 16, 8, 128])
# Hm_normalized = mode_unfold(H_normalized,m) #([Umax*Pmax*1024, 16])
# Ym_normalized = mode_unfold(Y_normalized,m) #([Umax*Pmax*1024, 16])
# perm=torch.randperm(Hm_normalized.shape[0])
# Hm=Hm_normalized[perm]
# Ym=Ym_normalized[perm]

torch.manual_seed(42)   # any integer you like
nb_atoms=len(BS_DoA)
total_nb_obs=21_000
a_idx=torch.randint(nb_atoms,(total_nb_obs,4))
a_val=torch.randn(a_idx.shape,dtype=torch.complex128)+1j*torch.randn(a_idx.shape,dtype=torch.complex128)
alpha=torch.zeros((total_nb_obs,nb_atoms),dtype=a_val.dtype)
alpha.scatter_(1,a_idx,a_val)
real_D=real_BS_Dictionary
H=alpha@real_D.T
# #data visualization:
# for o in range(5):
#     plt.plot(torch.abs(real_D.H@H[o]))
#     plt.vlines(torch.nonzero(alpha[o]),0,0.6,colors='red')
#     plt.show()

#%% ######################################### for different nb of observations ##############################################################
#generate observation:
SNR_avg_dB=10 #dB
snr_avg_lin = 10.0 ** (SNR_avg_dB / 10.0)
# Compute noise variance
nb_elements = H.shape[1:].numel()
sigma2 = (H.abs().square().sum(dim=1).mean()) / (nb_elements * snr_avg_lin)
# Generate complex Gaussian noise
noise = torch.sqrt(sigma2 / 2) * (torch.randn(*H.shape) + 1j * torch.randn(*H.shape))
# Add noise to the channel
Y = H + noise
H = normalize(H)
Y = normalize(Y)

nb_obs_list=np.linspace(200,20_000,8,dtype=int)
means_list=[]
for nb_obs in nb_obs_list:
    Hm=H[:nb_obs]
    Ym=Y[:nb_obs]
    Hm_test=H[-1000:]
    Ym_test=Y[-1000:]

    #train data
    train_valid_ratio = 0.8
    tv_split_index = int(Hm.shape[0] * train_valid_ratio)
    Hm_train   = Hm [:tv_split_index].to(device)
    Ym_train   = Ym [:tv_split_index].to(device)
    # validation data 
    Hm_val     = Hm [tv_split_index:].to(device)
    Ym_val     = Ym[tv_split_index:].to(device)

    #--------------------------- evaluate model BEFORE training and model with real dictionary----------------------------------
    iter_max=4

    H_test_nominaldict = Ym_test - OMP(Ym_test,nominal_BS_Dictionary, sigma2, iter_max=iter_max)[0] #Y-r
    H_test_realdict = Ym_test - OMP(Ym_test,real_BS_Dictionary, sigma2, iter_max=iter_max)[0]

    # Compute NMSEs
    NMSE_nominal=NMSE(Hm_test, H_test_nominaldict)
    NMSE_real=NMSE(Hm_test,H_test_realdict)

    # ----------------------------------- Deep unfolding ------------------------------------------
    # parameters defining
    # model defining
    OMPnet = OMP_1D_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff, BS_DoA)
    # OMPnet = OMP_1D_model(nominal_BS_ant_position, real_BS_gains, real_BS_coupling_coeff, BS_DoA)
    #optimizer
    # optimizer = torch.optim.Adam(unfolded_MOMP_model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam([
        {'params': OMPnet.learnable_ant_pos_y, 'lr':1e-5},
        {'params': OMPnet.ant_gains, 'lr':1e-2},
        {'params': OMPnet.coupling_coeff, 'lr':1e-2},
    ])
    # scheduler= torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.7)    
    #---------------------------------------eval before training-----------------------------------------------
    OMPnet.eval()
    train_losses_list, valid_losses_list = [], []
    with torch.no_grad():
        train_losses_list.append(NMSE(Hm_train,Ym_train - OMPnet(Ym_train,iter_max=iter_max)[0]))
        valid_losses_list.append(NMSE(Hm_val,Ym_val - OMPnet(Ym_val,iter_max=iter_max)[0]))
    #---------------------------------------training-----------------------------------------------
    OMPnet.train()
    if nb_obs<1_000:
        nb_epochs = 100
        lstsq=True
    else:
        nb_epochs=20
        lstsq=False
    # batch_size = 1 # batch size
    batch_size = 300
    train_size = Ym_train.shape[0]

    best_loss=torch.inf
    for i in tqdm(range(nb_epochs)):
        
        for i in range(0, train_size, batch_size):
            Y_batched =   Ym_train[i:i + batch_size].to(device)
            H_batched  =   Hm_train[i:i + batch_size].to(device)
            Y_batched=Y_batched.squeeze()
            H_batched=H_batched.squeeze()

        ################################## channel estimation #####################################################
            res_batched=OMPnet(Y_batched,sigma2,iter_max=iter_max)[0]
            H_est_batched=Y_batched-res_batched
            loss = torch.mean(NMSE(Y_batched,H_est_batched))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step() # Update the learning rate using the scheduler
        with torch.no_grad():
            # --- TRAIN ---
            H_est_train = Ym_train - OMPnet(Ym_train, sigma2,iter_max=iter_max)[0] # Y - r
            train_loss = NMSE(Hm_train,H_est_train)
            train_losses_list.append(train_loss)

            # --- VALIDATION ---
            H_est_val = Ym_val - OMPnet(Ym_val, sigma2,iter_max=iter_max)[0]
            valid_loss = NMSE(Hm_val, H_est_val)
            valid_losses_list.append(valid_loss)

    #--------------- evaluate model after training ----------------------------
    OMPnet.eval()
    with torch.no_grad():
        H_test_MOMPnet = Ym_test - OMPnet(Ym_test, sigma2,iter_max=iter_max)[0]
        # Compute NMSEs
        NMSE_OMP=NMSE(Hm_test, H_test_MOMPnet)

    # --------------------------------------- MOD -----------------------------------------------------
    D0=nominal_BS_Dictionary
    # D0 = torch.randn(nominal_BS_Dictionary.shape, dtype=nominal_BS_Dictionary.dtype)
    # D0=D0/torch.norm(D0, dim=0, keepdim=True)
    D_MOD=MOD(Ym_train,D0,OMP_iter=iter_max,epsilon=2e-2,iter_max=2000,torchlstsq=lstsq)
    nmse_MOD=NMSE(Hm_test,Ym_test-OMP(Ym_test,D_MOD,iter_max=iter_max)[0])

    nmse0=NMSE(Hm_test,Ym_test)
    nmse1 = NMSE_nominal
    nmse2 = NMSE_OMP
    nmse3 = NMSE_real

    means_list.append([
        nmse0.mean().item(),
        nmse1.mean().item(),
        nmse_MOD.mean().item(),
        nmse2.mean().item(),
        nmse3.mean().item()
    ])
means_arr=np.stack(means_list,axis=1)
# NMSE means vs SNR OR vs Dataset size
labels = [
    'Observation error',
    'OMP with nominal Dict',
    'OMP with MOD',
    'OMPnet',
    'OMP with real Dict'
]
colors = [color_observation, color_nominal, color_MOD, color_MOMP, color_real]
markers = ['o', 's', 'D', '^', 'x']
plt.figure(figsize=(8, 5))

# means_arr must have shape (5, len(nb_obs_list))
# one row per method

for i in range(len(labels)):
    plt.plot(train_valid_ratio*nb_obs_list,
             means_arr[i],
             color=colors[i],
             label=labels[i],
             marker=markers[i])

plt.yscale('log')
plt.xticks(ticks=train_valid_ratio*nb_obs_list, labels=(train_valid_ratio*nb_obs_list).astype(int))
plt.xlabel('Number of observations in training')
plt.xlim(left=0)
plt.ylabel('NMSE (mean)')
plt.title('Mean NMSE vs number of observations')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()




#%% ######################################### for different SNR ##############################################################
means=[]
nb_obs=10_000
snr_list_dB=np.linspace(0,20,10, dtype=int)
snr_list = 10.0 ** (snr_list_dB / 10.0)
for snr_avg_lin in snr_list:
    nb_elements = H.shape[1:].numel()
    sigma2 = (H.abs().square().sum(dim=1).mean()) / (nb_elements * snr_avg_lin)
    # Generate complex Gaussian noise
    noise = torch.sqrt(sigma2 / 2) * (torch.randn(*H.shape) + 1j * torch.randn(*H.shape))
    # Add noise to the channel
    Y = H + noise
    H = normalize(H)
    Y = normalize(Y)
    Hm=H[:nb_obs]
    Ym=Y[:nb_obs]
    Hm_test=H[-1000:]
    Ym_test=Y[-1000:]

    #train data
    train_valid_ratio = 0.8
    tv_split_index = int(Hm.shape[0] * train_valid_ratio)
    Hm_train   = Hm [:tv_split_index].to(device)
    Ym_train   = Ym [:tv_split_index].to(device)
    # validation data 
    Hm_val     = Hm [tv_split_index:].to(device)
    Ym_val     = Ym[tv_split_index:].to(device)

    #--------------------------- evaluate model BEFORE training and model with real dictionary----------------------------------
    iter_max=4
    
    H_test_nominaldict = Ym_test - OMP(Ym_test,nominal_BS_Dictionary, sigma2, iter_max=iter_max)[0] #Y-r
    H_test_realdict = Ym_test - OMP(Ym_test,real_BS_Dictionary, sigma2, iter_max=iter_max)[0]
    # Compute NMSEs
    NMSE_nominal=NMSE(Hm_test, H_test_nominaldict)
    NMSE_real=NMSE(Hm_test,H_test_realdict)

    # ----------------------------------- Deep unfolding ------------------------------------------
    # parameters defining
    # model defining
    OMPnet = OMP_1D_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff, BS_DoA)
    # OMPnet = OMP_1D_model(nominal_BS_ant_position, real_BS_gains, real_BS_coupling_coeff, BS_DoA)
    #optimizer
    # optimizer = torch.optim.Adam(unfolded_MOMP_model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam([
        {'params': OMPnet.learnable_ant_pos_y, 'lr':1e-5},
        {'params': OMPnet.ant_gains, 'lr':1e-2},
        {'params': OMPnet.coupling_coeff, 'lr':1e-2},
    ])
    # scheduler= torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.7)
    #---------------------------------------eval before training-----------------------------------------------
    OMPnet.eval()
    train_losses_list, valid_losses_list = [], []
    with torch.no_grad():
        train_losses_list.append(NMSE(Hm_train,Ym_train - OMPnet(Ym_train,iter_max=iter_max)[0]))
        valid_losses_list.append(NMSE(Hm_val,Ym_val - OMPnet(Ym_val,iter_max=iter_max)[0]))
    #---------------------------------------training-----------------------------------------------
    OMPnet.train()
    if nb_obs<1_000:
        nb_epochs = 100
        lstsq=True
    else:
        nb_epochs=20
        lstsq=False

    batch_size = 300
    train_size = Ym_train.shape[0]

    best_loss=torch.inf
    for i in tqdm(range(nb_epochs)):
        
        for i in range(0, train_size, batch_size):
            Y_batched =   Ym_train[i:i + batch_size].to(device)
            H_batched  =   Hm_train[i:i + batch_size].to(device)
            Y_batched=Y_batched.squeeze()
            H_batched=H_batched.squeeze()

        ################################## channel estimation #####################################################
            res_batched=OMPnet(Y_batched,sigma2,iter_max=iter_max)[0]
            H_est_batched=Y_batched-res_batched
            loss = torch.mean(NMSE(Y_batched,H_est_batched))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step() # Update the learning rate using the scheduler
        with torch.no_grad():
            # --- TRAIN ---
            H_est_train = Ym_train - OMPnet(Ym_train, sigma2,iter_max=iter_max)[0] # Y - r
            train_loss = NMSE(Hm_train,H_est_train)
            train_losses_list.append(train_loss)

            # --- VALIDATION ---
            H_est_val = Ym_val - OMPnet(Ym_val, sigma2,iter_max=iter_max)[0]
            valid_loss = NMSE(Hm_val, H_est_val)
            valid_losses_list.append(valid_loss)

    #--------------- evaluate model after training ----------------------------
    OMPnet.eval()
    with torch.no_grad():
        H_test_MOMPnet = Ym_test - OMPnet(Ym_test, sigma2,iter_max=iter_max)[0]
        # Compute NMSEs
        NMSE_OMP=NMSE(Hm_test, H_test_MOMPnet)

    # --------------------------------------- MOD -----------------------------------------------------
    D0=nominal_BS_Dictionary
    # D0 = torch.randn(nominal_BS_Dictionary.shape, dtype=nominal_BS_Dictionary.dtype)
    # D0=D0/torch.norm(D0, dim=0, keepdim=True)
    D_MOD=MOD(Ym_train,D0,OMP_iter=iter_max,epsilon=2e-2,iter_max=2000,torchlstsq=lstsq)
    nmse_MOD=NMSE(Hm_test,Ym_test-OMP(Ym_test,D_MOD,iter_max=iter_max)[0])

    nmse0=NMSE(Hm_test,Ym_test)
    nmse1 = NMSE_nominal
    nmse2 = NMSE_OMP
    nmse3 = NMSE_real

    means_list.append([
        nmse0.mean().item(),
        nmse1.mean().item(),
        nmse_MOD.mean().item(),
        nmse2.mean().item(),
        nmse3.mean().item()
    ])

means_arr=np.stack(means_list,axis=1)
# NMSE means vs SNR OR vs Dataset size
labels = [
    'Observation error',
    'OMP with nominal Dict',
    'OMP with MOD',
    'OMPnet',
    'OMP with real Dict'
]
colors = [color_observation, color_nominal, color_MOD, color_MOMP, color_real]
markers = ['o', 's', 'D', '^', 'x']
plt.figure(figsize=(8, 5))

# means_arr must have shape (5, len(nb_obs_list))
# one row per method

for i in range(len(labels)):
    plt.plot(snr_list,
             means_arr[i],
             color=colors[i],
             label=labels[i],
             marker=markers[i])

plt.yscale('log')
plt.xlabel('average SNR')
plt.xticks(ticks=snr_list, labels=snr_list_dB)
plt.xlim(left=0)
plt.ylabel('NMSE (mean)')
plt.title('Mean NMSE vs SNR')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


#%% Save data
save_dict = {
    'model_state_dict': OMPnet.state_dict(),
    'NMSE0': NMSE_nominal,
    'NMSEZ': NMSE_OMP,
    'NMSE_real': NMSE_real,
    'train_losses': train_losses_list,
    'valid_losses': valid_losses_list
}

# Save to a file
torch.save(save_dict, 'OMPnet_1D.pth')

print("Model and lists saved successfully!")

#%%#############################################################################################################################################################################
##################################################################  plot evaluation ############################################################################################
################################################################################################################################################################################
#%% ------------------------ Learned parameters -------------------------------------------------------------
learned_BS_ant_pos=torch.stack([torch.tensor(nominal_BS_ant_position[:,0]), list(OMPnet.parameters())[0].detach(), torch.tensor(nominal_BS_ant_position[:,2])], dim=1)
learned_D=steering_vect_dict(BS_DoA,learned_BS_ant_pos,antenna_gains=list(OMPnet.parameters())[1].detach(),antenna_coupling_coeff=list(OMPnet.parameters())[2].detach(),lambda_=lambda_)

learned_BS_pos=list(OMPnet.parameters())[0].detach().numpy()  # first parameter tensor
learned_gains=list(OMPnet.parameters())[1].detach().numpy()  # 2nd parameter tensor
learned_coupling=list(OMPnet.parameters())[2].detach().numpy()  # 3rd parameter tensor
nominal_BS_gains = np.asarray(BS_gains['nominal_BS_gains'])
nominal_BS_coupling_coeff = np.asarray(BS_coupling['nominal_BS_coupling_coeff'],dtype=np.complex128)
real_BS_ant_position = np.asarray(real_BS_ant_position)
nominal_BS_ant_position = np.asarray(nominal_BS_ant_position)
real_BS_gains = np.asarray(real_BS_gains)
l=2/lambda_
real_BS_gains_normalized = real_BS_gains / np.sqrt(np.sum((np.abs(real_BS_gains)**2)))
nominal_BS_gains_normalized = nominal_BS_gains / np.sqrt(np.sum((np.abs(nominal_BS_gains)**2)))
learned_gains_normalized= learned_gains / np.sqrt(np.sum((np.abs(learned_gains)**2)))
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
#--------------------------------------------------Plotting testing NMSE------------------------------------------------------------------
# Filter and slice data
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
    'OMP with nominal Dicts',
    'OMPnet',
    'OMP with real Dicts'
]
colors = [color_observation, color_nominal, color_OMP, color_real]
width=0.5
# Plot
plt.figure(figsize=(8, 5))
x = np.arange(len(means))
plt.bar(x, means,width=width, color=colors, alpha=0.7)
# plt.yscale('log')
plt.semilogy()
plt.xticks(x, labels, rotation=20)
plt.ylabel('NMSE (mean)')
plt.title('Mean NMSE comparison')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ALL BS parameters in one fig
nominal_min=nominal_BS_ant_position[:,1].min()
scaled_positions = [(pos-nominal_min) * l for pos in [real_BS_ant_position[:,1], learned_BS_pos, nominal_BS_ant_position[:,1]]]
fig, ax = plot_multiple_parameter_sets(
    scaled_positions,
    [real_BS_gains_normalized, learned_gains_normalized, nominal_BS_gains_normalized],
    [real_BS_coupling_coeff, learned_coupling, nominal_BS_coupling_coeff],
    colors=[color_real,color_OMP,color_nominal],labels=["Real ", "Learned", "Nominal"],
    y_spacing=2.0,positions_scale=0.8,mag_scale=1,
    figsize=(12,8)
)
plt.show()

# %% dictionaries comparison
# plt.imshow(torch.abs(torch.conj(nominal_BS_Dictionary).T@nominal_BS_Dictionary))
# plt.show()
# plt.imshow(torch.abs(torch.conj(D_MOD).T@D_MOD))
# plt.show()
# plt.imshow(torch.abs(torch.conj(learned_D).T@learned_D))
# plt.show()
# plt.imshow(torch.abs(torch.conj(real_BS_Dictionary).T@real_BS_Dictionary))
# plt.show()
# print(torch.norm(real_BS_Dictionary-nominal_BS_Dictionary))
# print(torch.norm(real_BS_Dictionary-D_MOD))
# print(torch.norm(real_BS_Dictionary-learned_D))
#%%
# Compute means
means = [
    nmse0.mean().item(),
    nmse1.mean().item(),
    nmse_MOD.mean().item(),
    nmse2.mean().item(),
    nmse3.mean().item()
]

# Labels and colors
labels = [
    'Observation error',
    'OMP with nominal Dict',
    'OMP with MOD',
    'OMPnet',
    'OMP with real Dict'
]
colors = [color_observation, color_nominal, color_MOD, color_MOMP, color_real]
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



# %%
