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
from matplotlib.patches import Circle

#%%
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
checkpoint = torch.load('.saved_data\.saved_models\MOMPnet5_bis_test.pth', weights_only=False)
learned_params_list = checkpoint['learned_params_list']

# to numpy
nominal_BS_gains = np.asarray(BS_gains['nominal_BS_gains'])
nominal_BS_coupling_coeff = np.asarray(BS_coupling['nominal_BS_coupling_coeff'],dtype=np.complex128)
real_BS_ant_position = np.asarray(real_BS_ant_position)
nominal_BS_ant_position = np.asarray(nominal_BS_ant_position)
real_MS_ant_position = np.asarray(real_MS_ant_position)
nominal_MS_ant_position = np.asarray(nominal_MS_ant_position)
real_BS_gains = np.asarray(real_BS_gains)




# %%
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

fig, ax = plt.subplots(figsize=figsize)
if colors is None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def update(frame):
    ax.cla()  # clear axes before redrawing

    ############################################### ALL BS parameters in one fig ####################################
    learned_BS_pos=learned_params_list[frame]['learned_BS_pos']
    learned_gains=learned_params_list[frame]['learned_gains']
    learned_coupling=learned_params_list[frame]['learned_coupling']

    if frame>0:
        learned_BS_pos=learned_BS_pos*0.95

    list_of_positions = [(pos-nominal_min) * l for pos in [nominal_BS_ant_position[:,1], learned_BS_pos, real_BS_ant_position[:,1]]]
    list_of_gains=[nominal_BS_gains, learned_gains, real_BS_gains]
    list_of_c1=[nominal_BS_coupling_coeff, learned_coupling, real_BS_coupling_coeff]
    assert len(list_of_positions) == len(list_of_gains) == len(list_of_c1) 
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
    ax.axis("off")
    ax.set_xlim(-1,13)
    ax.plot([],[],color='k',label='mutual coupling')
    ax.legend(ncol=4,fontsize=14,loc='lower center',bbox_to_anchor=(0.5, -0.35),frameon=False)
    
    return ax#, 
ani = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
plt.tight_layout()
plt.close(fig)

ani.save(
    "antenna_animation.mp4",
    writer="ffmpeg",
    fps=5,
    dpi=200,
)
# ani.save("animation.gif", writer="pillow", fps=5)

HTML(ani.to_jshtml()) 

# %%
