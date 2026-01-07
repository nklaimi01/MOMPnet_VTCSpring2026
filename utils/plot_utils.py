import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

def plot_antennas_with_parameters(
    ax, positions_y, gains, coupling_c1, color="C0", label=None,
    positions_scale=0.8,mag_scale=1.2, coupling_line_scale=1.0,

    circle_alpha=0.25, zorder_base=10, y_offset=0.0,
    center_marker=True, center_marker_size=20
):
    # median_spacing = np.median(np.diff(positions_y))  # in λ
    # mag_scale = mag_scale * median_spacing                 # circles ≈ 30% of spacing

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
    list_of_positions, list_of_gains, list_of_c1,colors=None, labels=None, y_spacing=2.0,positions_scale=0.8,mag_scale=1.2, figsize=(10,6),c1_legend=True,fontsize=10
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
    if c1_legend:
        ax.plot([],[],color='k',label='mutual coupling')


    if labels is not None:
        ax.legend(ncol=3,fontsize=fontsize,loc='lower center',bbox_to_anchor=(0.5, -0.2),frameon=False)

    plt.tight_layout()
    return fig, ax


def plot_antenna_positions(x,y_list,colors,title):
    # --- X and Y coordinates ---
    y_nominal,y_MOMP,y_real=y_list


    # --- Apply small horizontal offsets for visibility ---
    offset = 0.03  # adjust if antennas are close
    x_nominal = x - offset *1.5
    x_MOMP    = x - offset * 0.5
    x_real     = x + offset *0.5

    # --- Plot ---
    fig= plt.figure(figsize=(6,5))

    plt.scatter(x_nominal, y_nominal, label='Nominal BS', marker='x', color=colors[0], s=50, linewidths=1)
    plt.scatter(x_MOMP, y_MOMP, label='Learned BS (MOMP)', marker='d',color=colors[1], s=70, edgecolors='k', alpha=0.8)
    plt.scatter(x_real, y_real, label='Real BS', color=colors[2], s=70, edgecolors='k', alpha=0.8)

    # --- Optional: connect each antenna index with dotted lines ---
    for i in range(len(x)):
        plt.plot([x_nominal[i], x_MOMP[i], x_real[i]],
                [y_nominal[i], y_MOMP[i], y_real[i]],
                color='gray', linestyle='--', alpha=0.4, linewidth=1)

    # --- Labels and style ---
    plt.title(title, fontsize=14)
    plt.xlabel('X-axis [m]')
    plt.ylabel('Y-axis [m]')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    return fig

def plot_antenna_gains(gains_list,colors):
    # --- Prepare data for plotting ---
    nominal_gains,learned_gains,real_gains=gains_list
    real_gains_normalized = real_gains / np.sqrt(np.sum((np.abs(real_gains)**2)))
    nominal_gains_normalized = nominal_gains / np.sqrt(np.sum((np.abs(nominal_gains)**2)))
    learned_gains_normalized= learned_gains / np.sqrt(np.sum((np.abs(learned_gains)**2)))
    
    idx = np.arange(len(real_gains_normalized))
    mag_real = np.abs(real_gains_normalized)
    mag_nominal = np.abs(nominal_gains_normalized)
    mag_MOMP = np.abs(learned_gains_normalized)

    phase_real = np.angle(real_gains_normalized)
    phase_nominal = np.angle(nominal_gains_normalized)
    phase_MOMP = np.angle(learned_gains_normalized)

    # --- Plot magnitude comparison ---
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(idx, mag_real, 'o-', label='Real', color=colors[0])
    plt.plot(idx, mag_MOMP, 'd-', label='MOMP Learned', color=colors[1])
    plt.plot(idx, mag_nominal, 'x--', label='Nominal', color=colors[2])
    plt.title('Antenna Gain Magnitudes')
    plt.xlabel('Antenna Index')
    plt.ylabel('|Gain|')
    plt.legend()
    plt.grid(True)

    # --- Plot phase comparison ---
    plt.subplot(1,2,2)
    plt.plot(idx, phase_real, 'o-', label='Real', color=colors[0])
    plt.plot(idx, phase_MOMP, 'd-', label='MOMP Learned', color=colors[1])
    plt.plot(idx, phase_nominal, 'x--', label='Nominal', color=colors[2])
    plt.title('Antenna Gain Phases')
    plt.xlabel('Antenna Index')
    plt.ylabel('Phase [rad]')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()