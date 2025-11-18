import torch 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


import torch

def stack_with_padding(tensors, dim=0, length=None, zero_padding=False):
    """
    Pad a list of 1D tensors so they all have the same length, 
    then stack them along the given dimension.
    
    Args:
        tensors (list[torch.Tensor]): List of 1D tensors.
        dim (int): Dimension along which to stack.
        length (int, optional): Target length. Defaults to the max length.
        zero_padding (bool): If True, pad with zeros; else with the last value of each tensor.
    """
    if not tensors:
        raise ValueError("Input list 'tensors' cannot be empty.")

    # Check all tensors are 1D
    if not all(t.ndim == 1 for t in tensors):
        raise ValueError("All tensors must be 1D.")

    if length is None:
        length = max(t.size(0) for t in tensors)

    padded = []
    for t in tensors:
        pad_len = length - t.size(0)
        if pad_len > 0:
            # Ensure padding value has the same dtype and device
            if zero_padding:
                pad_value = torch.zeros(1, dtype=t.dtype, device=t.device)
            else:
                pad_value = t[-1:].clone()  # keep same dtype/device safely

            last_vals = pad_value.expand(pad_len)
            t = torch.cat([t, last_vals])

        padded.append(t)

    return torch.stack(padded, dim=dim)


def NMSE(channel,channel_estimation):
    if channel.dim() == 3:
        channel = channel.unsqueeze(0)  # [1, Nbs, Nms, Nsub]
    if channel_estimation.dim() == 3:  
        channel_estimation = channel_estimation.unsqueeze(0)  # add batch dimension
    return torch.sum(torch.abs(channel-channel_estimation)**2,dim=(-3,-2,-1))/torch.sum(torch.abs(channel)**2,dim=(-3,-2,-1))

def model_estimation(Y, model, sigma2):
            H_est = torch.zeros_like(Y)
            for u in range(Y.shape[0]):
                for p in range(Y.shape[1]):
                    y = Y[u, p]
                    y = y.squeeze()

                    res, _, _ = model.forward(y, u, sigma2)
                    H_est[u, p] = y - res
            return H_est



def plot_antennas_with_parameters(
    ax, positions_y, gains, coupling_c1, color="C0", label=None,
    positions_scale=0.8,mag_scale=1.2, coupling_line_scale=1.0,

    circle_alpha=0.25, zorder_base=10, y_offset=0.0,
    center_marker=True, center_marker_size=20
):
    median_spacing = np.median(np.diff(positions_y))  # in λ
    mag_scale = mag_scale * median_spacing                 # circles ≈ 30% of spacing

    """Plot one set of antenna parameters (offset by y_offset)."""
    gains = np.asarray(gains).astype(np.complex128)
    positions_y=np.asarray(positions_y)
    coupling_c1=np.asarray(coupling_c1)
    N = positions_y.shape[0]
    positions = np.zeros((N,2))
    positions[:, 0]=positions_y*positions_scale
    assert gains.shape[0] == N, "gains must have same length as positions"

    # Apply vertical offset
    positions[:, 1] -= y_offset

    mags = np.abs(gains)
    phases = np.angle(gains)
    radii = (mags + 1e-8) * mag_scale

    # Plot antennas (circles + phase tick + optional center point)
    for pos, r, phi in zip(positions, radii, phases):
        circ = Circle(pos, radius=r, facecolor=color, alpha=circle_alpha,
                      edgecolor=color, linewidth=1.0, zorder=zorder_base + 2)
        ax.add_patch(circ)

        # phase tick
        vec = np.array([np.cos(phi), np.sin(phi)])
        end = pos + vec * r
        ax.plot([pos[0], end[0]], [pos[1], end[1]],
                color=color, linewidth=1.5, zorder=zorder_base + 3)

        # center marker
        if center_marker:
            ax.scatter(pos[0], pos[1], s=center_marker_size,
                       color=color, zorder=zorder_base + 4)

    # Coupling between adjacent antennas
    if N >= 2:
        c_abs, c_ang = np.abs(coupling_c1), np.angle(coupling_c1)
        for i in range(N - 1):
            # fixed tick length (absolute, not dependent on seg_vec)
            tick_len = coupling_line_scale * (c_abs + 1e-8)

            p0, p1 = positions[i], positions[i + 1]
            mid = 0.5 * (p0 + p1)
            seg_vec = p1 - p0
            seg_norm = np.linalg.norm(seg_vec)
            if seg_norm == 0:
                continue

            # unit vector along the segment
            seg_unit = seg_vec / seg_norm

            # rotate the unit vector by coupling phase to get tick direction
            rot = np.array([[np.cos(c_ang), -np.sin(c_ang)],
                            [np.sin(c_ang),  np.cos(c_ang)]])
            tick_dir = rot @ seg_unit  # now magnitude = 1

            # draw tick of fixed length
            start = mid - 0.5 * tick_len * tick_dir
            end   = mid + 0.5 * tick_len * tick_dir
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color='k', linewidth=1.8, zorder=zorder_base+4)


    if label is not None:
        ax.scatter([], [], color=color, alpha=0.9, label=label)
        


def plot_multiple_parameter_sets(
    list_of_positions, list_of_gains, list_of_c1,colors=None, labels=None, y_spacing=2.0,positions_scale=0.8,mag_scale=1.2, figsize=(10,6)
):
    """Plot multiple parameter sets with vertical offset."""
    assert len(list_of_positions) == len(list_of_gains) == len(list_of_c1) 

    fig, ax = plt.subplots(figsize=figsize)
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for k, (p, g, c1) in enumerate(zip(list_of_positions, list_of_gains, list_of_c1)):
        color = colors[k % len(colors)]
        label = None if labels is None else labels[k]
        y_offset = k * y_spacing
        plot_antennas_with_parameters(
            ax, p, g, c1, color=color, label=label,
            positions_scale=positions_scale,mag_scale=mag_scale,
            coupling_line_scale=1.0, circle_alpha=0.25,
            y_offset=y_offset, zorder_base=10+k,
            center_marker=True, center_marker_size=25
        )

    ax.set_aspect("equal", "box")
    # ax.set_xlabel("x (position)")
    ax.set_xticklabels(range(16))
    ax.set_yticklabels([])
    ax.set_title("BS Antenna parameters")
    ax.plot([],[],color='k',label='mutual coupling')


    if labels is not None:
        # ax.legend(loc="upper right")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    return fig, ax