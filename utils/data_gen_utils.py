from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
import sionna
import tensorflow as tf
import numpy as np
import torch
from tqdm import tqdm
import os

def init_scene_ULA(save_dir,BS_position,f0,nb_BS_antennas,nb_MS_antennas,delta_p_BS,delta_g_BS,coupling_coeff_BS):
    '''
    Generate a scene containing the positions of a Base Station (BS) and Mobile Station (MS),
    where both the BS and MS are arranged as Uniform Linear Arrays (ULAs). using sionna.rt.loadscene.

    Parameters:
        save_dir (str): Local directory where the generated data will be saved (created if it does not exist).
        BS_position (array-like): Position of the Base Station (BS). Uplink direction is assumed.
        f0 (float): Carrier frequency of the system.
        nb_BS_antennas (int): Number of antennas at the Base Station.
        nb_MS_antennas (int): Number of antennas at the Mobile Station.
        sigma_p (float): Standard deviation of noise in BS antenna positions.
        sigma_g (float): Standard deviation of noise in BS antenna gains.
        coupling_strength (float): Off-diagonal magnitude of the mutual coupling matrix.

    Returns:
        scene (sionna.rt.Scene): The generated scene object.

    Notes:
        - This function saves the BS antenna positions, gains, and mutual coupling matrix in `save_dir`.
    '''
    delta_g_mag,delta_g_phase=delta_g_BS
    scene = load_scene(sionna.rt.scene.etoile)

    # change frequency
    scene.frequency=f0
    lambda_ = 3e8/f0

    # Configuration of transmitters
    scene.tx_array = PlanarArray(num_rows=1, num_cols=nb_MS_antennas,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")
    
    # Configuration of recievers 
    # the antenna by default is located in the y-z plane 
    # This config creates antennas that are aligned over y axis 
    scene.rx_array = PlanarArray(num_rows=1, num_cols=nb_BS_antennas,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")
    
    
    #generate real and nominal positions at the BS
    nominal_BS_ant_positions = tf.Variable(scene.rx_array.positions)
    scene.rx_array.positions=tf.Variable(scene.rx_array.positions)
    scene.rx_array.positions[:,1].assign(scene.rx_array.positions[:,1]+ delta_p_BS * lambda_ * (2 * np.random.rand(nb_BS_antennas) - 1))
    real_BS_ant_positions = scene.rx_array.positions

    # Add BS to the scene
    rx = Receiver("rx", position=BS_position, orientation=[0,0,0])
    scene.add(rx)

    #Save nominal and real positions
    BS_ant_position_dict={}
    BS_ant_position_dict['nominal_BS_ant_position']=nominal_BS_ant_positions
    BS_ant_position_dict['real_BS_ant_position']=real_BS_ant_positions

    #antenna gains at the BS
    nominal_BS_gains=np.ones(nb_BS_antennas,dtype=complex)
    real_g_mag=np.abs(nominal_BS_gains) + delta_g_mag * (np.random.rand(nb_BS_antennas) - 1)
    real_g_phase=np.angle(nominal_BS_gains) + delta_g_phase * (2 * np.random.rand(nb_BS_antennas) - 1)
    real_BS_gains=real_g_mag * np.exp(1j * real_g_phase)

    #Save nominal and real antenna gains
    BS_gains_dict={}
    BS_gains_dict['nominal_BS_gains']=nominal_BS_gains
    BS_gains_dict['real_BS_gains']=real_BS_gains

    #mutual coupling matrix at the BS
    # Diagonal = 1 (self-coupling) + Upper diagonal + Lower diagonal
    nominal_BS_coupling_coeff = np.complex128(0 + 0j)
    real_BS_coupling_coeff = np.complex128(coupling_coeff_BS)

    #Save nominal and real antenna gains
    BS_coupling_coeff_dict={}
    BS_coupling_coeff_dict['nominal_BS_coupling_coeff']=nominal_BS_coupling_coeff
    BS_coupling_coeff_dict['real_BS_coupling_coeff']=real_BS_coupling_coeff

    if save_dir is not None:
        np.savez( save_dir/"BS_ant_position.npz", **BS_ant_position_dict)
        np.savez( save_dir/"BS_gains.npz", **BS_gains_dict)
        np.savez( save_dir/"BS_coupling.npz", **BS_coupling_coeff_dict)


    return scene

def generate_channels(save_dir, scene, nb_users, nb_positions_per_user, subcarriers,delta_p_MS=0.1):
    '''
    Generate the dataset required for the project based on a given scene.
    The dataset includes channel realizations for multiple users at multiple positions,
    considering individual hardware impairments for each user.

    Parameters:
        save_dir (str): Local directory where the generated data will be saved (created if it does not exist).
        scene (sionna.rt.Scene): Scene generated using `init_scene_ULA`.
        nb_users (int): Number of users in the scene, each with their own hardware impairments.
        nb_positions_per_user (int): Number of positions each user will visit in the scene. 
                                     Positions are randomly generated within a 200m radius from the BS.
        subcarriers (array-like): Array of pilot subcarrier frequencies.

    Returns:
        channels (numpy.ndarray): Generated clean channel realizations of shape 
                                  [nb_users, nb_positions_per_user, nb_BS_antennas, nb_MS_antennas, nb_subcarriers].

    Notes:
        - This function saves the generated channels, user positions, and MS antenna positions in `save_dir`.
        - It should be called right after `init_scene_ULA`.
    '''
    subcarriers=tf.cast(subcarriers,tf.float32) 
    channels_list=[]
    users_pos_list=[]
    #Generate UE positions 
    rx=list(scene.receivers.values())[0]
    BS_position = rx.position.numpy().tolist()

    # Load data
    #load antenna gains at the BS:
    BS_gains=np.load(save_dir/'BS_gains.npz')
    real_BS_gains=BS_gains['real_BS_gains']

    #load mutual coupling matrix at the BS
    BS_coupling=np.load(save_dir/'BS_coupling.npz')
    real_BS_coupling_coeff=BS_coupling['real_BS_coupling_coeff']
    nb_BS_antennas=len(real_BS_gains)
    off_diag = np.full(nb_BS_antennas - 1, real_BS_coupling_coeff, dtype=np.complex128)
    real_BS_coupling = np.eye(nb_BS_antennas, dtype=np.complex128) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    
    #antenna positions at the MS
    MS_ant_position_dict={}
    nominal_MS_ant_positions = tf.Variable(scene.tx_array.positions)
    MS_ant_position_dict['nominal_MS_ant_position']=nominal_MS_ant_positions
    expanded_positions_per_user =nb_positions_per_user *3 # Expand margin for positions (to account for channels with zero norm being discarded)
    nb_positions=nb_users*expanded_positions_per_user  # number of random positions in the grid
    rayon=200
    rng = np.random.default_rng(seed=None)
    centre=BS_position[:2]
    lambda_ = 0.010706874
    u = rng.random(nb_positions)
    v = rng.random(nb_positions)
    r = rayon * np.sqrt(u)
    theta=2*np.pi*v
    x = centre[0] + r * np.cos(theta)
    y = centre[1] + r * np.sin(theta)
    z = np.full(nb_positions,1.5)
    random_positions_grid=np.column_stack([x, y, z])  # combine into table (nb_positions, 3)

    #Generate random UE antenna array orientation : A 3D rotation with yaw, pitch, and roll angles
    yaw = 2*np.pi*rng.random(nb_positions)
    pitch= 2*np.pi*rng.random(nb_positions)
    roll= np.zeros(nb_positions)
    random_orientation=np.column_stack([yaw, pitch, roll])  # combine into table (nb_positions, 3)

    pbar = tqdm(total=nb_users, desc=f"Building Dataset", unit="user")

    real_MS_ant_position_list=[]
    for user in range(nb_users):

        #generate real and nominal positions at the BS
        scene.tx_array.positions=tf.Variable(nominal_MS_ant_positions)
        scene.tx_array.positions[:,1].assign(scene.tx_array.positions[:,1]+ delta_p_MS * lambda_ * (2*np.random.rand(scene.tx_array.num_ant)-1))
        real_MS_ant_position_list.append(scene.tx_array.positions)

        #generate realistic synthetic channels using sionna RT:

        # choose user's position and orientation from the random grids
        cur_positions=random_positions_grid[user*expanded_positions_per_user :(user+1)*expanded_positions_per_user ,:]
        cur_orientation=random_orientation[user*expanded_positions_per_user :(user+1)*expanded_positions_per_user ]
        
        #Add UE 
        for idx in range(cur_positions.shape[0]):
            tx = Transmitter(name='tx'+f'_{idx}',
                    position=cur_positions[idx],
                        orientation=cur_orientation[idx])
            scene.add(tx)
        paths=scene.compute_paths()
        paths.normalize_delays=False
        # Extract path delays and gains (single TX, multiple paths)
        a,tau=paths.cir()
        #built in function method : cir
        H_sionna = sionna.channel.cir_to_ofdm_channel(subcarriers,a,tau,normalize=False)
        H_sionna = H_sionna.numpy().squeeze() #[nb_BS_antenna , nb_chan_train]
        channel=H_sionna.transpose(1,0,2,3)
        channel=np.einsum('a,uamk->uamk',real_BS_gains,channel) #consider antennas gains at BS
        channel=np.einsum('ab,ubmk->uamk',real_BS_coupling,channel) #consider antennas coupling at BS
        
        # remove 0 norm paths and save the rest
        norms=(np.abs(channel).sum(axis=(1,2,3)))
        idx_to_delete=np.where(norms==0)
        channel=np.delete(channel,idx_to_delete,axis=0)
        tx_positions=np.delete(cur_positions,idx_to_delete,axis=0) #save UE positions

        pbar.update(1)  # manually update progress
        pbar.set_postfix()
        # pbar.set_postfix(valid=batch_idx)
        #remove UE 
        for idx in range(cur_positions.shape[0]):
            scene.remove(f'tx_{idx}')         

        if channel.size>0:
            channels_list.append(channel)
            #saving users positions
            users_pos_list.append(tx_positions)

    # Truncate to given number of positions per user and stack:
    min_nb_positions = min(c.shape[0] for c in channels_list)
    min_nb_positions = min(min_nb_positions, nb_positions_per_user)

    channels = np.stack([v[:min_nb_positions] for v in channels_list], axis=0) #[user, user_position, BS_antennas, MS_antennas, subcarriers]
    channels_dict={}
    channels_dict['channels']=channels

    users_position=np.stack([v[:min_nb_positions] for v in users_pos_list], axis=0) #[user, user_position, position]
    users_positions_dict={}
    users_positions_dict['users_position']=users_position

    real_MS_ant_position=np.stack(real_MS_ant_position_list, axis=0) #[user, MS_antennas, position]
    MS_ant_position_dict['real_MS_ant_position']=real_MS_ant_position

    if save_dir is not None:
        os.makedirs(save_dir,exist_ok=True)
        np.savez( save_dir/ "channels.npz", **channels_dict)
        np.savez( save_dir/ "users_position.npz", **users_positions_dict)
        np.savez( save_dir/ "MS_ant_position.npz", **MS_ant_position_dict)
    return channels


def add_UEs_to_scene(my_scene,my_positions):
    ''' positions: numpy array of shape (Nb_UE, 3)
    '''
    for tx in my_scene.transmitters.copy():   # copy to avoid modifying while iterating
        my_scene.remove(tx)
    for i, pos in enumerate(my_positions):
        tx = Transmitter(name=f"tb_{i}", position=pos.tolist())  
        my_scene.add(tx)

def generate_observations(save_dir,H,SNR_avg_dB):
    """
    Generate observation Y=H+N
    by adding complex Gaussian noise to channels to achieve an average SNR.
    Variations across users, positions, and subcarriers will naturally occur.
    Parameters:
        channels: [nb_users, nb_positions, nb_BS_ant, nb_MS_ant, nb_subcarriers]
        SNR_avg_dB: target SNR in dB
        
    Returns:
        Y: Noisy channel observations, same shape as H
        sigma2 : Estimated noise variance.
            
    """
    # Convert SNR from dB to linear scale
    snr_avg_lin = 10.0 ** (SNR_avg_dB / 10.0)

    # Compute noise variance
    nb_elements = np.prod(H.shape[2:])
    sigma2 = np.mean(np.sum(np.abs(H)**2, axis=(2, 3, 4))) / (nb_elements * snr_avg_lin)

    # Generate complex Gaussian noise
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))

    # Add noise to the channel
    observations = H + noise

    # Prepare dictionary
    observations_dict = {'observations': observations, 'sigma2': sigma2}

    # Save to .npz if directory is specified
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, "observations.npz"), **observations_dict)

    return observations, sigma2