#%% importing libraries
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from models.OMP_model import OMP_1D_model,OMP_uncnstrd_model
from utils.dictionary_gen_utils import *
import matplotlib.pyplot as plt
from saved_data_loader import *
from utils.training_utils import *
#%%
torch.manual_seed(42)   # any integer you like
nb_atoms=len(BS_DoA)
total_nb_obs=31_000
a_idx=torch.randint(nb_atoms,(total_nb_obs,4))
a_val=torch.randn(a_idx.shape,dtype=torch.complex128)+1j*torch.randn(a_idx.shape,dtype=torch.complex128)
alpha=torch.zeros((total_nb_obs,nb_atoms),dtype=a_val.dtype)
alpha.scatter_(1,a_idx,a_val)
real_D=real_BS_Dictionary
H=alpha@real_D.T

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
Hm_test=H[-1000:]
Ym_test=Y[-1000:]

#train data
train_valid_ratio = 0.8
tv_split_index = int(H.shape[0] * train_valid_ratio)
H_train   = H [:tv_split_index].to(device)
Y_train   = Y [:tv_split_index].to(device)
# validation data 
H_val     = H [tv_split_index:].to(device)
Y_val     = Y[tv_split_index:].to(device)

#--------------------------- evaluate model BEFORE training and model with real dictionary----------------------------------
iter_max=4
H_test_nominaldict = Ym_test - OMP(Ym_test,nominal_BS_Dictionary, iter_max=iter_max)[0] #Y-r
H_test_realdict = Ym_test - OMP(Ym_test,real_BS_Dictionary, iter_max=iter_max)[0]
# Compute NMSEs
NMSE_nominal=NMSE(Hm_test, H_test_nominaldict)
NMSE_real=NMSE(Hm_test,H_test_realdict)
#%%
# ----------------------------------- Deep unfolding ------------------------------------------
# parameters defining
# model defining
OMPnet = OMP_1D_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff, BS_DoA)

optimizer = torch.optim.Adam([
    {'params': OMPnet.learnable_ant_pos_y, 'lr':1e-4},
    {'params': OMPnet.ant_gains, 'lr':1e-2},
    {'params': OMPnet.coupling_coeff, 'lr':1e-2},
])
# scheduler= torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.7)   

#---------------------------------------eval before training-----------------------------------------------
NMSE_OMPnet_list=[] 
train_losses_list, valid_losses_list = [], []
OMPnet.eval()
with torch.no_grad():
    H_test_MOMPnet = Ym_test - OMPnet(Ym_test,iter_max=iter_max)[0]
    NMSE_OMPnet_list.append(NMSE(Hm_test, H_test_MOMPnet).mean().item())
    train_losses_list.append(NMSE(H_train,Y_train - OMPnet(Y_train,iter_max=iter_max)[0]))
    valid_losses_list.append(NMSE(H_val,Y_val - OMPnet(Y_val,iter_max=iter_max)[0]))
#%%
#---------------------------------------training-----------------------------------------------
OMPnet.train()
# batch_size = 1 # batch size
batch_size = 100
train_size = Y_train.shape[0]
for i in tqdm(range(0, train_size, batch_size)):
    Y_batched =   Y_train[i:i + batch_size].to(device)
    H_batched  =   H_train[i:i + batch_size].to(device)
    Y_batched=Y_batched.squeeze()
    H_batched=H_batched.squeeze()
    ################################## channel estimation #####################################################
    res_batched=OMPnet(Y_batched,iter_max=iter_max)[0]
    H_est_batched=Y_batched-res_batched
    loss = torch.mean(NMSE(Y_batched,H_est_batched))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step() # Update the learning rate using the scheduler
    with torch.no_grad():
        # --- TRAIN ---
        H_est_train = Y_train - OMPnet(Y_train,iter_max=iter_max)[0] # Y - r
        train_loss = NMSE(H_train,H_est_train)
        train_losses_list.append(train_loss)
        # --- VALIDATION ---
        H_est_val = Y_val - OMPnet(Y_val,iter_max=iter_max)[0]
        valid_loss = NMSE(H_val, H_est_val)
        valid_losses_list.append(valid_loss)
    #--------------- evaluate model after training ----------------------------
    OMPnet.eval()
    with torch.no_grad():
        H_test_MOMPnet = Ym_test - OMPnet(Ym_test,iter_max=iter_max)[0]
        # Compute NMSEs
        NMSE_OMPnet_list.append(NMSE(Hm_test, H_test_MOMPnet).mean().item())
nmse_OMPnet=np.array(NMSE_OMPnet_list)


#%% ------------------------------------------------ ML: unconstrained OMPnet generated randomly --------------------------------------------------------------------------------
D0 = torch.randn(nominal_BS_Dictionary.shape, dtype=nominal_BS_Dictionary.dtype)
D0=D0/torch.norm(D0, dim=0, keepdim=True)
#%%
ML_OMPnet = OMP_uncnstrd_model(D0.clone())
optimizer = torch.optim.Adam(ML_OMPnet.parameters(), lr=1e-2)
# scheduler= torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.7)   

#---------------------------------------eval before training-----------------------------------------------
NMSE_MLnet_list=[] 
ML_train_losses_list, ML_valid_losses_list = [], []
ML_OMPnet.eval()
with torch.no_grad():
    H_test_MLnet = Ym_test - ML_OMPnet(Ym_test,iter_max=iter_max)[0]
    # Compute NMSEs
    NMSE_MLnet_list.append(NMSE(Hm_test, H_test_MLnet).mean().item())
    ML_train_losses_list.append(NMSE(H_train,Y_train - ML_OMPnet(Y_train,iter_max=iter_max)[0]))
    ML_valid_losses_list.append(NMSE(H_val,Y_val - ML_OMPnet(Y_val,iter_max=iter_max)[0]))
#---------------------------------------training-----------------------------------------------
ML_OMPnet.train()
# batch_size = 1 # batch size
batch_size = 100
train_size = Y_train.shape[0]
for i in tqdm(range(0, train_size, batch_size)):
    Y_batched =   Y_train[i:i + batch_size].to(device)
    H_batched  =   H_train[i:i + batch_size].to(device)
    Y_batched=Y_batched.squeeze()
    H_batched=H_batched.squeeze()
    ################################## channel estimation #####################################################
    res_batched=ML_OMPnet(Y_batched,iter_max=iter_max)[0]
    H_est_batched=Y_batched-res_batched
    loss = torch.mean(NMSE(Y_batched,H_est_batched))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step() # Update the learning rate using the scheduler
    with torch.no_grad():
        # --- TRAIN ---
        H_est_train = Y_train - ML_OMPnet(Y_train,iter_max=iter_max)[0] # Y - r
        train_loss = NMSE(H_train,H_est_train)
        ML_train_losses_list.append(train_loss)
        # --- VALIDATION ---
        H_est_val = Y_val - ML_OMPnet(Y_val,iter_max=iter_max)[0]
        valid_loss = NMSE(H_val, H_est_val)
        ML_valid_losses_list.append(valid_loss)
    #--------------- evaluate model after training ----------------------------
    ML_OMPnet.eval()
    with torch.no_grad():
        H_test_MLnet = Ym_test - ML_OMPnet(Ym_test,iter_max=iter_max)[0]
        # Compute NMSEs
        NMSE_MLnet_list.append(NMSE(Hm_test, H_test_MLnet).mean().item())
nmse_MLnet=np.array(NMSE_MLnet_list)


#%%--------------------------------------------------Plot NMSE on testing channels vs nb of seen channels: ------------------------------------------------------------------
nmse_obs=NMSE(Hm_test,Ym_test).mean().item()
nmse_nominal = NMSE_nominal.mean().item()
nmse_real = NMSE_real.mean().item()

means_arr=np.vstack([np.full_like(nmse_OMPnet, nmse_nominal),nmse_MLnet, nmse_OMPnet,np.full_like(nmse_OMPnet, nmse_real)])
# NMSE means vs SNR OR vs Dataset size
labels = [
    # 'Observation',
    'MB (inaccurate)',
    'AI',
    'MB-AI',
    'MB (perfect)'
]
colors = [ color_nominal,'orange', 'blue' , color_real]
markers = ['o','s','^', 'x']
linestyles=['--','-','-','--']
P=50
plt.figure(figsize=(8, 5))

# means_arr must have shape (5, len(nb_obs_list))
# one row per method
nb_seen_channels=np.arange(0, train_size+batch_size, batch_size)
for i in range(len(labels)):
    plt.plot(nb_seen_channels,
             means_arr[i],
             color=colors[i],
             label=labels[i], markevery=P,
             marker=markers[i], linestyle=linestyles[i])

# plt.yscale('log')
ticks = nb_seen_channels[::P]
plt.xticks(ticks=ticks, labels=(ticks/1e3).astype(int))
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
##################################################################  evaluation #################################################################################################
################################################################################################################################################################################
#--------------------------------------------------Plotting learning curve------------------------------------------------------------------
# Convert list of tensors -> average NMSE per epoch
train_losses_avg = [t.mean().item() for t in train_losses_list]
valid_losses_avg = [v.mean().item() for v in valid_losses_list]
ML_train_losses_avg = [t.mean().item() for t in ML_train_losses_list]
ML_valid_losses_avg = [v.mean().item() for v in ML_valid_losses_list]

epochs = range(0, len(train_losses_avg))

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses_avg, label='Train MB-ML', marker='o', color='blue',markevery=P)
plt.plot(epochs, valid_losses_avg, label='Validation MB-ML ', marker='s', color='orange',markevery=P)
plt.plot(epochs, ML_train_losses_avg, label='Train ML', marker='D',linestyle='--', color='blue',markevery=P)
plt.plot(epochs, ML_valid_losses_avg, label='Validation ML', marker='v',linestyle='--', color='orange',markevery=P)

plt.gca().spines['left'].set_position('zero')
plt.xlabel('training batches')
# plt.xticks(epochs)
plt.ylabel('NMSE')
plt.title('Learning Curve')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

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
#-------------------------------------------------- ALL BS parameters in one fig --------------------------------------------------
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
