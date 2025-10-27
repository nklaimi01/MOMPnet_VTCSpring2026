#%%
from pathlib import Path
import torch
import time
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import os
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
# Now you can safely import
from utils.data_gen_utils import *
from utils.dictionary_gen_utils import *
from saved_data_loader import *

#%%  functions
#OMP
def OMP(h, D1, D2, D3,iter_max=30, sigma2_est=None,stopping_criterion='SC1'):
    """
    Orthogonal Matching Pursuit (OMP) algorithm for sparse channel estimation
    using a Kronecker-structured dictionary.

    This implementation stores intermediate channel estimates at each iteration
    and supports two stopping criteria.

    Parameters:
        h : torch.Tensor
            True channel tensor to approximate. Shape can be arbitrary.
        D1, D2, D3 : torch.Tensor
            Factor matrices of the Kronecker dictionary. Shapes: (n1, m1), (n2, m2), (n3, m3)
        sigma2_est : float
            Estimated noise variance, used in the stopping criterion.
        iter_max : int, optional (default=30)
            Maximum number of OMP iterations.
        stopping_criterion : str, optional (default='SC1')
            Stopping criterion type:
            - 'SC1': stops when the squared residual norm is below N * sigma2_est.
            - 'SC2': stops when the squared magnitude of the last coefficient is below 2 * sigma2_est.

    Returns:
        estimations : torch.Tensor
            Stack of estimated channels at each iteration, including the initial channel.
            Shape: [iterations + 1, *h.shape]

    Notes:
        - The Kronecker-structured dictionary is constructed as D = kron(D1, kron(D2, D3)).
        - The algorithm flattens the channel tensor internally for least-squares projection.
        - The residual and intermediate estimations are reshaped back to the original channel shape.
    """
    N=h.numel()
    i = 0
    I_list=[]
    h_reshaped = h.reshape(-1)  # flatten channel
    D_I_list=[]
    r = h  # initialize residual
    estimations_list = []
    estimations_list.append(h)  # store initial channel (iteration 0)
    stop=False

    while not stop:
        # Step 1: compute correlations c = D^H r (tensor contractions with D1, D2, D3)
        AADM = torch.einsum('ab,bms->ams', torch.conj(D1).T, r) 
        AADM = torch.einsum('am,bms->bas', torch.conj(D2).T, AADM)
        AADM = torch.einsum('as,bms->bma', torch.conj(D3).T, AADM) 

        # Step 2: choose most correlated atom 
        i1, i2, i3 = torch.unravel_index(torch.argmax(torch.abs(AADM)), AADM.shape)
        I_list.append(torch.tensor([i1, i2, i3], device=device))

        # Step 3: construct D_I from chosen atoms so far
        vec1 = D1[:, i1]
        vec2 = D2[:, i2]
        vec3 = D3[:, i3]
        D_I_list.append(torch.kron(torch.kron(vec1, vec2), vec3))
        D_I=torch.stack(D_I_list,1)

        # Step 4: projection (solve least-squares to update coefficients)
        x = torch.linalg.lstsq(D_I, h_reshaped).solution
        proj_h = D_I @ x

        # Step 5: update residual
        r_reshaped = h_reshaped - proj_h
        r = r_reshaped.reshape(h.shape)

        # Store intermediate estimation
        estimations_list.append(h - r)

        i += 1
        if sigma2_est is None:
           SC=False
        else:
           SC= torch.sum(torch.abs(r)**2)<=N*sigma2_est # see mpnet paper   
        
        if SC or i>iter_max-1:
            stop=True

    # Stack all estimations along first dimension
    estimations = torch.stack(estimations_list, 0)
    return estimations

#MOMP
def MOMP(h,D1,D2,D3,iter_max=30,sigma2_est=None,refine_iter=None):
    N=h.numel()
    stop=False
    iter = 0
    I_list=[]
    h_reshaped = h.reshape(-1)
    D_I_list=[]
    r = h
    estimations_list=[]
    estimations_list.append(h)
    while not stop:
        corr1 = torch.einsum('ab,bms->ams', torch.conj(D1).T, r)
        i1 = torch.argmax((corr1.abs()**2).sum(dim=(1, 2)))

        corr2 = torch.conj(D2).T @ corr1[i1]
        i2 = torch.argmax((corr2.abs()**2).sum(dim=1))

        corr3 = torch.conj(D3).T @ corr2[i2]
        i3 = torch.argmax(torch.abs(corr3)**2)

        # Refinement of atom selection 
        if refine_iter is not None:
            # print('avant raffinement: ',i1,i2,i3)
            atom=[i1,i2,i3]
            D=[D1,D2,D3]
            for _ in range(refine_iter):
                for d in range(len(atom)):
                    other_idx1, other_idx2 = (set(range(len(atom))) - {d})
                    vec_0=D[other_idx1][:,atom[other_idx1]]
                    vec_1=D[other_idx2][:,atom[other_idx2]]
                    r_permuted=r.permute(other_idx1,other_idx2,d) #[N_o1,N_o2,N_d]
                    corr_d = torch.einsum('a,abc,b->c',torch.conj(vec_0),r_permuted,torch.conj(vec_1)) #[N_o1,N_o2,N_d] -> [N_o2,N_d] -> [N_d]
                    corr_d = torch.matmul(torch.conj(D[d]).T, corr_d) #[A_d]
                    i_d = torch.argmax(torch.abs(corr_d)**2) #refined selection of i_d
                    atom[d]=i_d
            i1,i2,i3=atom
            # print('après raffinement: ',i1,i2,i3)
        # else:
        #     print('chosen atom:',[i1,i2,i3])
        vec1 = D1[:, i1]
        vec2 = D2[:, i2]
        vec3 = D3[:, i3]
        I_list.append(torch.tensor([i1,i2,i3], device=device))
             
        D_I_list.append(torch.kron(torch.kron(vec1, vec2), vec3))
        D_I=torch.stack(D_I_list,1)

        x = torch.linalg.lstsq(D_I, h_reshaped).solution
        proj_h = D_I @ x
        r_reshaped = h_reshaped - proj_h
        r = r_reshaped.reshape(h.shape)
        estimations_list.append(h-r)
        iter += 1

        if sigma2_est is None:
            SC=False
        else:
            SC=torch.sum(torch.abs(r)**2)<=N*sigma2_est

        if SC or iter>iter_max-1:
            stop=True
    estimations=torch.stack(estimations_list,0)
    return estimations


def NMSE(channel,channel_estimation):
    if channel.dim() == 3:
        channel = channel.unsqueeze(0)  # [1, Nbs, Nms, Nsub]
    if channel_estimation.dim() == 3:  
        channel_estimation = channel_estimation.unsqueeze(0)  # add batch dimension
    return torch.sum(torch.abs(channel-channel_estimation)**2,dim=(1,2,3))/torch.sum(torch.abs(channel)**2)

def stack_with_padding(tensors,dim,length=None):
    """
    Pad a list of 1D tensors with their last value
    so they all have the same length.
    """
    if length is None:
        length = max(t.size(0) for t in tensors)
    padded = []
    for t in tensors:
        if t.size(0) < length:
            pad_len = length - t.size(0)
            last_val = t[-1].expand(pad_len)   # repeat last value
            t = torch.cat([t, last_val])
        padded.append(t)
    return torch.stack(padded, dim=dim)  # shape [max_len, n_tensors]

def nmse_animation(nmse_1,nmse_2,logscale,save=None,labels=['OMP','MOMP'],colors=["#E07F8C","#77DD77"],y_min=1e-6):

    n_frames=max(len(nmse_1),len(nmse_2))
    nb_channels=nmse_1.shape[1]
    channels_idx = np.arange(1, nb_channels + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))


    # Initial bars
    bars1 = ax.bar(channels_idx - width/2, nmse_1[0], width, label=labels[0], color=colors[0])
    bars2 = ax.bar(channels_idx + width/2, nmse_2[0], width, label=labels[1], color=colors[1])
    for xi, yi in zip(channels_idx, nmse_1[0]):
        ax.hlines(y=yi, xmin=xi-0.4, xmax=xi+0.4, colors="red", linestyles=":")


    # --- NEW: Add horizontal lines at the minimum NMSE per channel ---
    min_nmse_1,_ = nmse_1.min(axis=0)
    min_nmse_2,_ = nmse_2.min(axis=0)
    # Plot green dashed lines for minima
    for xi, yi in zip(channels_idx, min_nmse_1):
        ax.hlines(y=yi, xmin=xi-0.4, xmax=xi, colors="blue", linestyles="-", linewidth=1.5, alpha=0.8)
    for xi, yi in zip(channels_idx, min_nmse_2):
        ax.hlines(y=yi, xmin=xi, xmax=xi+0.4, colors="blue", linestyles="-", linewidth=1.5, alpha=0.8)


    ax.set_xlabel('Channel Index')
    ax.set_ylabel('NMSE')
    ax.set_title(f'NMSE Comparison: {labels[0]} vs {labels[1]}')
    ax.set_xticks(channels_idx)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Y-axis
    if logscale:
        ax.set_yscale('log')
        ymin = y_min
    else:
        ymin=0
    ymax = max(nmse_1.max(), nmse_2.max())
    ax.set_ylim(ymin, ymax * 1.1)

    # Iteration text
    iteration_text = ax.text(
        0.95, 0.95, '', transform=ax.transAxes,
        ha='right', va='top', fontsize=12, color='blue'
    )

    # min_nmse_1,_ = nmse_1.min(axis=0)
    # min_nmse_2,_ = nmse_2.min(axis=0)
    # min_lines_1 = [None] * nb_channels
    # min_lines_2 = [None] * nb_channels

    def update(frame):
        for b, h in zip(bars1, nmse_1[frame]):
            b.set_height(h)
        for b, h in zip(bars2, nmse_2[frame]):
            b.set_height(h)
            
        # # --- NEW: Add horizontal lines at the minimum NMSE per channel ---
        # for i, xi in enumerate(channels_idx):
        #     # NMSE 1 minima
        #     if (nmse_1[frame, i] == min_nmse_1[i]) and min_lines_1[i] is None:
        #         min_lines_1[i] = ax.hlines(
        #             y=min_nmse_1[i], xmin=xi-0.4, xmax=xi,
        #             colors="blue", linestyles="-", linewidth=1.5, alpha=0.8
        #         )

        #     # NMSE 2 minima
        #     if (nmse_2[frame, i] == min_nmse_2[i]) and min_lines_2[i] is None:
        #         min_lines_2[i] = ax.hlines(
        #             y=min_nmse_2[i], xmin=xi, xmax=xi+0.4,
        #             colors="blue", linestyles="-", linewidth=1.5, alpha=0.8
        #         )

        iteration_text.set_text(f'iteration {frame}/{n_frames-1}')
        return [*bars1, *bars2, iteration_text]#, *min_lines_1, *min_lines_2]

    ani = FuncAnimation(fig, update, frames=n_frames, interval=300, blit=False)

    if save=='gif':
        ani.save("nmse_animation.gif", writer="pillow", fps=5)
    elif save=='mp4':
        ani.save("nmse_animation.mp4", writer="ffmpeg", fps=5)

    plt.close(fig)  # avoid static plot

    return HTML(ani.to_jshtml()) 


#%% OMP vs MOMP on observations
D_B = real_BS_Dictionary
D_S = FRV_Dictionary.to(dtype=torch.complex128)

users = [0,16,44,63]
positions = [2,8,16,50,75]
nb_channels=len(users)*len(positions)
nmse_omp_list=[]
nmse_momp_list=[]
iter_max=80
sigma2_est=sigma2
i=0
MOMP_refine_iter=3
for user in users:
    for p in positions:
        print(f'---------------------example {i+1}:---------------------------')
        channel = channels[user, p]
        observation=observations[user, p]
        D_M = real_MS_Dictionaries[user]

        # start_omp = time.time()
        estimations_omp=OMP(observation,D_B,D_M,D_S,iter_max,sigma2_est)
        print(f'number of iterations OMP={len(estimations_omp)-1}')
        # end_omp = time.time()
        # print(f"OMP time: {end_omp - start_omp:.6f} seconds")

        # start_momp = time.time()
        estimations_momp=MOMP(observation,D_B,D_M,D_S,iter_max,sigma2_est)
        print(f'number of iterations MOMP={len(estimations_momp)-1}')

        # end_momp = time.time()
        # print(f"MOMP time: {end_momp - start_momp:.6f} seconds")
    
        nmse_omp_list.append(NMSE(channel,estimations_omp))
        nmse_momp_list.append(NMSE(channel,estimations_momp))

        i+=1

nmse_momp=stack_with_padding(nmse_momp_list,1)
nmse_omp=stack_with_padding(nmse_omp_list,1,length=len(nmse_momp))

#%%
nmse_animation(nmse_omp,nmse_momp,logscale=True,save='mp4',y_min=1e-4)

#%% quantitive eval
print("=== NMSE Comparison over independent examples ===")

nmse_method1=nmse_omp[-1]
nmse_method2=nmse_momp[-1]
# Per-example difference
diff = nmse_method2 - nmse_method1
improvements = 100 * diff / nmse_method2  # improvement percentage per example

print(f"Average NMSE:")
print(f"  { 'Method 1':<10}: {nmse_method1.mean():.4e}")
print(f"  { 'Method 2':<10}: {nmse_method2.mean():.4e}")

print("\nMinimum NMSE:")
print(f"  { 'Method 1':<10}: {nmse_method1.min():.4e}")
print(f"  { 'Method 2':<10}: {nmse_method2.min():.4e}")

print("\nAverage improvement (%):")
print(f"  Method 1 improves by {improvements.mean():.2f}% over Method 2 on average")

# Optional: fraction of examples where Method 1 is better
better_fraction = np.mean(nmse_method1 < nmse_method2) * 100
print(f"  Method 1 performs better in {better_fraction:.1f}% of examples")

#%% 2 methods comparison test
D_B = real_BS_Dictionary
D_S = FRV_Dictionary.to(dtype=torch.complex128)

users = [0,16,44,63]
positions = [2,8,16,50,75]
nb_channels=len(users)*len(positions)
nmse_momp_=[]
nmse_momp_refine=[]
sigma2_est=sigma2
iter_max=50
i=0
for user in users:
    for p in positions:
        print(f'---------------------example {i+1}:---------------------------')
        channel = channels[user, p]
        observation=observations[user, p]
        D_M = real_MS_Dictionaries[user]

        # start_omp = time.time()
        estimations_momp_=MOMP(observation,D_B,D_M,D_S,iter_max,sigma2_est)
        # print(f'number of iterations MOMP ={len(estimations_momp_)-1}')
        # end_omp = time.time()
        # print(f"OMP time: {end_omp - start_omp:.6f} seconds")

        # start_momp = time.time()
        estimations_momp_refine=MOMP(observation,D_B,D_M,D_S,iter_max,sigma2_est,refine_iter=2)
        # print(f'number of iterations MOMP refined={len(estimations_momp_refine)-1}')

        # end_momp = time.time()
        # print(f"MOMP time: {end_momp - start_momp:.6f} seconds")
    
        nmse_momp_.append(NMSE(channel,estimations_momp_))
        nmse_momp_refine.append(NMSE(channel,estimations_momp_refine))

        i+=1
nmse_momp_=stack_with_padding(nmse_momp_,1)
nmse_momp_refine=stack_with_padding(nmse_momp_refine,1,length=len(nmse_momp_))
#%%
nmse_animation(nmse_momp_,nmse_momp_refine,logscale=True,labels=['MOMP NO refine','MOMP refined'],colors=["#E4E266","#7F91E0"],save='mp4')

# %%
