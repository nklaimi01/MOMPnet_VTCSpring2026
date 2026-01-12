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
# H=channels[:Umax,:Pmax] #([Umax,Pmax, 16, 8, 128])
# Y=observations[SNR][:Umax,:Pmax] #temporarily 
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

# nb_obs_list = np.arange(200, 20001, 2500, dtype=int)
nb_obs_list = np.array([200, 500, 1000, 2000, 5000, 7500, 10000, 15000],dtype=int)
means_list1=[]
for nb_obs in nb_obs_list:
    #train data
    Hm_train=H[:nb_obs]
    Ym_train=Y[:nb_obs]
    Hm_test=H[-1000:]
    Ym_test=Y[-1000:]
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
    train_losses_list = []
    with torch.no_grad():
        train_losses_list.append(NMSE(Hm_train,Ym_train - OMPnet(Ym_train,iter_max=iter_max)[0]))
    #---------------------------------------training-----------------------------------------------
    OMPnet.train()
    nb_epochs_dict = {
        200: 100,
        500: 100,
        1_000: 70,
        2_000: 50,
    }
    lstsq = nb_obs < 1_000
    nb_epochs = nb_epochs_dict.get(nb_obs, 20)  # default 20 if nb_obs not in dict

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

    means_list1.append([
        nmse0.mean().item(),
        nmse1.mean().item(),
        nmse_MOD.mean().item(),
        nmse2.mean().item(),
        nmse3.mean().item()
    ])
#%% figures
means_arr1=np.stack(means_list1,axis=1)
# NMSE means vs Dataset size
labels = [
    'Observation',
    'OMP with nominal Dict',
    'OMP with MOD',
    'OMPnet',
    'OMP with real Dict'
]
colors = [color_observation, color_nominal, color_MOD, color_MOMP, color_real]
markers = ['o', 's', 'D', '^', 'x']
linestyles=['--','--','-','-','--']
markersizes=[6,6,6,8,6]
fontsize=16
plt.figure(figsize=(8, 5))

# means_arr must have shape (5, len(nb_obs_list))
# one row per method

for i in range(len(labels)):
    plt.plot(nb_obs_list,
             means_arr1[i],
             color=colors[i],
             label=labels[i],
             marker=markers[i],linestyle=linestyles[i],markersize=markersizes[i])
    # plt.plot(
    #     [nb_obs_list[j] for j in range(len(nb_obs_list)) if j != 1],  # skip 2nd point
    #     [means_arr1[i][j] for j in range(len(nb_obs_list)) if j != 1],
    #     color=colors[i],
    #     label=labels[i],
    #     marker=markers[i]
    # )
plt.yscale('log')
xlabels = list(nb_obs_list)
xlabels[1] = ''
xlabels[2] = ''
plt.xticks(ticks=nb_obs_list, labels=xlabels, fontsize=fontsize)
plt.xlabel('training dataset size', fontsize=fontsize)
plt.xlim(left=0)
plt.ylabel('NMSE (logscale)',fontsize=fontsize)
plt.yticks(fontsize=fontsize)
# plt.title('Mean NMSE vs number of observations')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=fontsize)
plt.tight_layout()
# plt.savefig("MODobs.pdf", bbox_inches="tight")
plt.show()

#%% ######################################### for different SNR ##############################################################
means_list=[]
nb_obs=7_500
snr_list_dB=np.linspace(0,15,8, dtype=int)
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
    Hm_test=H[-1000:]
    Ym_test=Y[-1000:]
    #train data
    Hm_train=H[:nb_obs]
    Ym_train=Y[:nb_obs]
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
    train_losses_list = []
    with torch.no_grad():
        train_losses_list.append(NMSE(Hm_train,Ym_train - OMPnet(Ym_train,iter_max=iter_max)[0]))
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

#%% figure
means_arr=np.stack(means_list,axis=1)
# NMSE means vs SNR 
labels = [
    'Observation',
    'OMP with nominal Dict',
    'OMP with MOD',
    'OMPnet',
    'OMP with real Dict'
]
colors = [color_observation, color_nominal, color_MOD, color_MOMP, color_real]
markers = ['o', 's', 'D', '^', 'x']
linestyles=['--','--','-','-','--']
markersizes=[6,6,6,10,6]
plt.figure(figsize=(8, 5))

# means_arr must have shape (5, len(nb_obs_list))
# one row per method

for i in range(len(labels)):
    plt.plot(snr_list_dB,
             means_arr[i],
             color=colors[i],
             label=labels[i],
             marker=markers[i],markersize=markersizes[i])#,linestyle=linestyles[i]

plt.yscale('log')
plt.xlabel('average SNR',fontsize=fontsize)
plt.xticks(ticks=snr_list_dB, labels=snr_list_dB,fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlim(left=0)
# plt.ylabel('NMSE (logscale)',fontsize=fontsize)
# plt.title('Mean NMSE vs SNR')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
# plt.legend(fontsize=fontsize)
plt.tight_layout()
plt.savefig("MODsnr.pdf", bbox_inches="tight")
plt.show()


# #%% Save data
# save_dict = {
#     'model_state_dict': OMPnet.state_dict(),
#     'NMSE0': NMSE_nominal,
#     'NMSEZ': NMSE_OMP,
#     'NMSE_real': NMSE_real,
#     'train_losses': train_losses_list,
# }

# # Save to a file
# torch.save(save_dict, 'OMPnet_1D.pth')

# print("Model and lists saved successfully!")

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




