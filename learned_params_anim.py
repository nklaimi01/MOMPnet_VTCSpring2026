#%%
import torch
import time
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from saved_data_loader import *
from utils.plot_utils import *

# learned_params_list=[]
# learned_BS_pos=MOMPnet.BS_learnable_pos_y.detach().numpy()  # first parameter tensor
# learned_gains=MOMPnet.BS_ant_gains.detach().numpy()  # 2nd parameter tensor
# learned_coupling=MOMPnet.BS_coupling_coeff.detach().numpy()  # 3rd parameter tensor
# learned_MS_pos=torch.stack([p.detach() for p in MOMPnet.MS_learnable_pos_list], 0).cpu().numpy()  # 4th parameter tensor
# learned_params_list.append({'learned_BS_pos':learned_BS_pos,'learned_gains':learned_gains,'learned_coupling':learned_coupling,'learned_MS_pos':learned_MS_pos})


# learned_params_list = []
# for _ in range(30):
#     learned_BS_pos = np.random.uniform(-0.04, 0.04, size=(16,)).astype(np.float32)
#     learned_gains = (np.random.randn(16) + 1j * np.random.randn(16)).astype(np.complex64)
#     learned_coupling = np.complex64(np.random.randn() + 1j * np.random.randn())
#     learned_MS_pos = None

#     learned_params_list.append({
#         'learned_BS_pos': learned_BS_pos,
#         'learned_gains': learned_gains,
#         'learned_coupling': learned_coupling,
#         'learned_MS_pos': learned_MS_pos
#     })
checkpoint = torch.load(f'.saved_data\.saved_models\MOMPnet_new_{SNR_av}_dB.pth')
learned_params_list = checkpoint['learned_params_list']

# to numpy
nominal_BS_gains = np.asarray(BS_gains['nominal_BS_gains'])
nominal_BS_coupling_coeff = np.asarray(BS_coupling['nominal_BS_coupling_coeff'],dtype=np.complex128)
real_BS_ant_position = np.asarray(real_BS_ant_position)
nominal_BS_ant_position = np.asarray(nominal_BS_ant_position)
real_MS_ant_position = np.asarray(real_MS_ant_position)
nominal_MS_ant_position = np.asarray(nominal_MS_ant_position)
real_BS_gains = np.asarray(real_BS_gains)



#%%
#! Animation:
y_spacing=1.5
positions_scale=0.8
mag_scale=1.2
fontsize=16
figsize=(10,7)
n_frames=len(learned_params_list)

colors=[color_nominal,color_MOMP,color_real]
labels=["Nominal ", "Learned", "Real"]
l=2/lambda_
nominal_min=nominal_BS_ant_position[:,1].min()
list_of_positions = [(pos-nominal_min) * l for pos in [nominal_BS_ant_position[:,1], np.zeros((16,)), real_BS_ant_position[:,1]]]
list_of_gains=[nominal_BS_gains, np.zeros((16,)), real_BS_gains]
list_of_c1=[nominal_BS_coupling_coeff, np.zeros(()), real_BS_coupling_coeff]
assert len(list_of_positions) == len(list_of_gains) == len(list_of_c1) 
fig, ax = plt.subplots(figsize=figsize)
if colors is None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for k, (p, g, c1) in enumerate(zip(list_of_positions, list_of_gains, list_of_c1)):
    color = colors[k % len(colors)]
    label = labels[k]
    y_offset = k * y_spacing
    g_normalized = g / np.sqrt(np.sum((np.abs(g)**2)))
    plot_antennas_with_parameters(
        ax, p, g_normalized, c1, color=color, label=label,
        positions_scale=positions_scale,mag_scale=mag_scale,
        coupling_line_scale=1.0, circle_alpha=0.25,
        y_offset=y_offset, zorder_base=10+k,
        center_marker=True, center_marker_size=25
    )
ax.set_aspect("equal", "box")
# ax.set_xlabel("x (position)")
# ax.set_xticklabels(range(16))
ax.set_xticklabels([])
ax.set_yticklabels([])
# ax.set_title("BS Antenna parameters")
ax.plot([],[],color='k',label='mutual coupling')
ax.legend(ncol=3,fontsize=fontsize,loc='lower center',bbox_to_anchor=(0.5, -0.2),frameon=False)


def update(frame):
    ax.cla()  # clear axes before redrawing

    ############################################### ALL BS parameters in one fig ####################################
    learned_BS_pos=learned_params_list[frame]['learned_BS_pos']
    learned_gains=learned_params_list[frame]['learned_gains']
    learned_coupling=learned_params_list[frame]['learned_coupling']

    list_of_positions = [(pos-nominal_min) * l for pos in [nominal_BS_ant_position[:,1], learned_BS_pos, real_BS_ant_position[:,1]]]
    list_of_gains=[nominal_BS_gains, learned_gains, real_BS_gains]
    list_of_c1=[nominal_BS_coupling_coeff, learned_coupling, real_BS_coupling_coeff]
    for k, (p, g, c1) in enumerate(zip(list_of_positions, list_of_gains, list_of_c1)):
        color = colors[k % len(colors)]
        label = labels[k]
        y_offset = k * y_spacing
        g_normalized = g / np.sqrt(np.sum((np.abs(g)**2)))
        plot_antennas_with_parameters(
            ax, p, g_normalized, c1, color=color, label=label,
            positions_scale=positions_scale,mag_scale=mag_scale,
            coupling_line_scale=1.0, circle_alpha=0.25,
            y_offset=y_offset, zorder_base=10+k,
            center_marker=True, center_marker_size=25
        )

    return ax#, 
ani = FuncAnimation(fig, update, frames=n_frames, interval=300, blit=False)
plt.tight_layout()
# ani.save("animation.gif", writer="pillow", fps=5)
plt.close(fig)
HTML(ani.to_jshtml()) 
# %%
