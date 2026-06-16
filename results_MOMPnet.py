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
from utils.plot_utils import *
from matplotlib.ticker import MultipleLocator
#%%
def ticklabels_array(highest_int, spacing):
    result = []
    for i in range(highest_int + 1):
        result.append(str(i))
        if i < highest_int:
            result.extend([""] * (spacing-1))
    return result

#%%---------------------------------------------------------------------------------------------------
Umax=10
Pmax=150
SNR_av_list=[0,5,15]
models_list=[]
for SNR_av in SNR_av_list:
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

    # #%%############################### MOMP with real dictionary ################################################################
    # H_val_realdict=MOMP_estimation(Y_val,real_BS_Dictionary,real_MS_Dictionaries,FRV_Dictionary,sigma2)
    # NMSE_real=NMSE(H_val,H_val_realdict)

    # H_val_nominaldict=MOMP_estimation(Y_val,nominal_BS_Dictionary,nominal_MS_Dictionary,FRV_Dictionary,sigma2)
    # NMSE_nominal=NMSE(H_val,H_val_nominaldict)

    # LOAD TRAINED MODELS
    ############################ MOMP ############################
    nominal_MS_ant_position_stacked = torch.stack([nominal_MS_ant_position.clone() for _ in range(Umax)], dim=0)
    MOMPnet = MOMP_model(nominal_BS_ant_position, nominal_BS_gains, nominal_BS_coupling_coeff, nominal_MS_ant_position_stacked,
                    subcarriers, BS_DoA, MS_DoA, delays)  # replace with your model class
    # Load everything
    checkpoint = torch.load(f'.saved_data\.saved_models\MOMPnet_{SNR_av}_dB.pth')
    # checkpoint = torch.load(f'MOMPnet_{SNR_av}_dB_02.pth')
    models_list.append(checkpoint)

    # Load model weights
    MOMPnet.load_state_dict(checkpoint['model_state_dict'])
    # Load the lists
    
#%%#############################################################################################################################################################################
##################################################################  plot evaluation ############################################################################################
################################################################################################################################################################################
#------------------------------------- subplot --------------------------------------------------
from matplotlib.ticker import MultipleLocator
fontsize = 16  # set desired fontsize

# Number of checkpoints
num_checkpoints = len(models_list)

fig, axes = plt.subplots(num_checkpoints, 1, figsize=(8, 2.5*num_checkpoints), sharex=True)

for idx, checkpoint in enumerate(models_list):
    MOMPnet_NMSE_list = checkpoint['MOMPnet_NMSE']
    NMSE_nominal = checkpoint['nominal_NMSE']
    NMSE_real = checkpoint['real_NMSE']

    MOMPnet_NMSE_avg = [v.mean().item() for v in MOMPnet_NMSE_list]
    nb_epochs = int((len(MOMPnet_NMSE_avg)-1)/nb_users)
    epochs = range(0, len(MOMPnet_NMSE_avg))

    # Labels and colors
    labels = [
        'MOMP with nominal parameters',
        'MOMPnet',
        'MOMP with real parameters'
    ]
    colors = [color_nominal, color_MOMP, color_real]
    markers = ['','^', '']
    linestyles=['--','-','-.']
    titles = [r'$0~\mathrm{dB}$', r'$5~\mathrm{dB}$', r'$15~\mathrm{dB}$']    
    nmse_nominal = NMSE_nominal.mean().item()
    nmse_MOMPnet = MOMPnet_NMSE_avg[-1]
    nmse_real = NMSE_real.mean().item()

    # Prepare data for plotting
    MOMPnet_NMSE_arr = np.array(MOMPnet_NMSE_avg)
    means_arr = np.vstack([
        np.full_like(MOMPnet_NMSE_arr, nmse_nominal),
        MOMPnet_NMSE_arr,
        np.full_like(MOMPnet_NMSE_arr, nmse_real)
    ])

    ax = axes[idx] if num_checkpoints > 1 else axes  # handle single subplot case
    P = 10
    for i in range(len(labels)):
        ax.plot(epochs,
                means_arr[i],
                color=colors[i],
                label=labels[i],
                marker=markers[i],
                linestyle=linestyles[i],
                markevery=P)
    
    ax.set_xlim(left=0)
    ax.set_xticks(epochs)
    ax.set_xticklabels(ticklabels_array(nb_epochs, nb_users), fontsize=fontsize)
    ax.set_yticks([nmse_nominal, nmse_real])
    ax.set_yticklabels([f'{nmse_nominal:.2f}', f'{nmse_real:.2f}'], fontsize=fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    # ax.legend()
    ax.set_title(f'average SNR={titles[idx]}', fontsize=fontsize)
axes[1].set_ylabel('NMSE', fontsize=fontsize)
axes[0].legend(fontsize=fontsize)
plt.xlabel('epochs', fontsize=fontsize)
plt.tight_layout()
fig.savefig("figures\MOMPnet_subplots.pdf")
plt.show()
#%%
#--------------------------------------------------Plotting learning curve------------------------------------------------------------------
SNR_av=5
# checkpoint = torch.load(f'MOMPnet_{SNR_av}_dB_02.pth')
checkpoint = torch.load(f'.saved_data\.saved_models\MOMPnet_{SNR_av}_dB_new.pth')

train_NMSE_list = checkpoint['train_NMSE']
MOMPnet_NMSE_list = checkpoint['MOMPnet_NMSE']
NMSE_nominal = checkpoint['nominal_NMSE']
NMSE_real = checkpoint['real_NMSE']
# Convert list of tensors -> average NMSE per epoch
train_NMSE_avg = [t.mean().item() for t in train_NMSE_list]
MOMPnet_NMSE_avg = [v.mean().item() for v in MOMPnet_NMSE_list]
nb_epochs=int((len(train_NMSE_avg)-1)/nb_users)
epochs = range(0, len(train_NMSE_avg))
plt.figure(figsize=(8, 5))
P=5
plt.plot(epochs, train_NMSE_avg, label='Train NMSE', marker='o', color='blue',markevery=P)
plt.plot(epochs, MOMPnet_NMSE_avg, label='Validation NMSE', marker='s', color='orange',markevery=P)
plt.gca().spines['left'].set_position('zero')
plt.xlabel('Epoch')
plt.xticks(ticks=epochs,labels=ticklabels_array(nb_epochs,nb_users))
plt.ylabel('NMSE')
plt.title('Learning Curve')
plt.gca().xaxis.set_major_locator(MultipleLocator(5))
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
#%%--------------------------------------------------Plotting validation NMSE------------------------------------------------------------------
# Labels and colors
labels = [
    'MOMP with nominal Dicts',
    'MOMPnet',
    'MOMP with real Dicts'
]
colors = [color_nominal, color_MOMP, color_real]
markers = ['','s', '']
linestyles=['--','-','--']
nmse_nominal = NMSE_nominal.mean().item()
nmse_MOMPnet = MOMPnet_NMSE_avg[-1]
nmse_real = NMSE_real.mean().item()
#plot throughout training 
MOMPnet_NMSE_arr=np.array(MOMPnet_NMSE_avg)
means_arr=np.vstack([np.full_like(MOMPnet_NMSE_arr, nmse_nominal), MOMPnet_NMSE_arr,np.full_like(MOMPnet_NMSE_arr, nmse_real)])
P=10
fig=plt.figure(figsize=(8, 5))
# means_arr must have shape (5, len(nb_obs_list))
# one row per method
for i in range(len(labels)):
    plt.plot(epochs,
            means_arr[i],
            color=colors[i],
            label=labels[i],
            marker=markers[i], linestyle=linestyles[i],markevery=P)
# plt.yscale('log')
ticks = epochs
plt.xticks(ticks=ticks, labels=ticklabels_array(nb_epochs,nb_users))
plt.xlabel('epochs')
plt.xlim(left=0)
plt.ylabel('NMSE (mean)')
# plt.title('Mean NMSE vs number of seen channels')
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.legend()
plt.tight_layout()
# plt.ylim([0.14,0.5])
# fig.savefig(f"figures\MOMPnet_{SNR_av}.pdf")
plt.show()


#%%#############################################################################################################################################################################
##################################################################  learned parameters #########################################################################################
################################################################################################################################################################################
checkpoint = models_list[1]
MOMPnet.load_state_dict(checkpoint['model_state_dict'])
learned_BS_pos=list(MOMPnet.parameters())[0].detach().numpy()  # first parameter tensor
learned_gains=list(MOMPnet.parameters())[1].detach().numpy()  # 2nd parameter tensor
learned_coupling=list(MOMPnet.parameters())[2].detach().numpy()  # 3rd parameter tensor
learned_MS_pos=torch.stack([p.detach() for p in MOMPnet.MS_learnable_pos_list], 0).cpu().numpy()  # 4th parameter tensor
# to numpy
nominal_BS_gains = np.asarray(BS_gains['nominal_BS_gains'])
nominal_BS_coupling_coeff = np.asarray(BS_coupling['nominal_BS_coupling_coeff'],dtype=np.complex128)
real_BS_ant_position = np.asarray(real_BS_ant_position)
nominal_BS_ant_position = np.asarray(nominal_BS_ant_position)
real_MS_ant_position = np.asarray(real_MS_ant_position)
nominal_MS_ant_position = np.asarray(nominal_MS_ant_position)
real_BS_gains = np.asarray(real_BS_gains)


#%% 
############################################### ALL BS parameters in one fig ####################################
colors=[color_nominal,color_MOMP,color_real]
l=2/lambda_
nominal_min=nominal_BS_ant_position[:,1].min()
scaled_positions = [(pos-nominal_min) * l for pos in [nominal_BS_ant_position[:,1], learned_BS_pos, real_BS_ant_position[:,1]]]
fig, ax = plot_multiple_parameter_sets(
    scaled_positions,
    [nominal_BS_gains, learned_gains, real_BS_gains],
    [nominal_BS_coupling_coeff, learned_coupling, real_BS_coupling_coeff],
    colors=colors,labels=["Nominal ", "Learned", "Real"],
    y_spacing=1.5,positions_scale=0.8,mag_scale=1.2,
    figsize=(10,7),fontsize=16,c1_legend=False
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
# plt.savefig('figures\learned_params.pdf', bbox_inches="tight")
plt.show()

#%%
####################################### plot learned BS postions #############################################
fig=plot_antenna_positions(x=real_BS_ant_position[:, 0],y_list=[nominal_BS_ant_position[:, 1],learned_BS_pos,real_BS_ant_position[:,1]],colors=colors,title='Base Station Antenna Positions')
############################################ plot learned BS antenna Gains ####################################
gains_list=[nominal_BS_gains,learned_gains,real_BS_gains]
plot_antenna_gains(gains_list,colors)
####################################### plot learned MS postions #############################################
for u in range(Umax):
    y_nominal = nominal_MS_ant_position[:, 1]
    y_real = real_MS_ant_position[u, :, 1]
    y_MOMP = learned_MS_pos[u]
    plot_antenna_positions(x=real_MS_ant_position[0, :, 0],y_list=[y_nominal,y_MOMP,y_real],colors=colors,title=f'Mobile Station #{u} Antenna Positions ')
#%% -------------------------------------- Quantitatif evaluation: quadratic error --------------------------------
# BS antenna positions
real_gains_normalized = real_BS_gains / np.sqrt(np.sum((np.abs(real_BS_gains)**2)))
nominal_gains_normalized = nominal_BS_gains / np.sqrt(np.sum((np.abs(nominal_BS_gains)**2)))
learned_gains_normalized= learned_gains / np.sqrt(np.sum((np.abs(learned_gains)**2)))
print('---------- BS Antenna parameters ---------')
print(f"|g_real - g_nominal| = {np.mean(np.abs(real_gains_normalized - nominal_gains_normalized)):.4e}")
print(f"|g_real - g_learned| = {np.mean(np.abs(real_gains_normalized - learned_gains_normalized)):.4e}")

print(f"‖P_real - P_nominal‖²₂ = {np.mean(np.abs(real_BS_ant_position[:,1] - nominal_BS_ant_position[:,1])):.4e}")
print(f"‖P_real - P_learned‖²₂ = {np.mean(np.abs(real_BS_ant_position[:,1] - learned_BS_pos)):.4e}")

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


# %%
