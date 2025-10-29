from sionna.rt import load_scene,PlanarArray,Transmitter
import sionna
nb_BS_antennas=16
nb_MS_antennas=8
f0=28e9 #HZ 
c = 3e8  # m/s
lambda_ =  c/f0
BS_position=[60, -90, 30]
#%%############################################################
############### coverage map using  sionna  ###################
###############################################################
scene = load_scene(sionna.rt.scene.etoile)

# change frequency
scene.frequency=f0
# Configuration of transmitters
scene.rx_array = PlanarArray(num_rows=1, num_cols=nb_MS_antennas,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")

# Configuration of recievers 
# the antenna by default is located in the y-z plane 
# This config creates antennas that are aligned over y axis 
scene.tx_array = PlanarArray(num_rows=1, num_cols=nb_BS_antennas,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")

# Add BS to the scene
tx = Transmitter("tx", position=BS_position, orientation=[0,0,0])
# rx = Receiver("rx", position=BS_position, orientation=[0,0,0])
scene.add(tx)
# scene.add(rx)
cm=scene.coverage_map(max_depth=8)
cm.show()
#%%
# #%% Memory allocation
# def sizeof(shape, dtype=torch.complex128):
#     """Compute memory size (in bytes) for a tensor with given shape and dtype."""
#     return int(torch.tensor(shape).prod().item()) * torch.tensor([], dtype=dtype).element_size()

# def pretty_size(nbytes):
#     """Format memory size in MB and GB."""
#     mb = nbytes / (1024**2)
#     gb = nbytes / (1024**3)
#     return f"{mb:.3f} MB ({gb:.4f} GB)"

# # List of arrays (using PyTorch tensors now)
# arrays = {
#     "channel": h,
#     "D_1": D_B,
#     "D_2": D_M,
#     "D_S": D_S
# }

# # Display shapes and dtypes
# for name, arr in arrays.items():
#     print(f"{name}: shape={tuple(arr.shape)}, dtype={arr.dtype}")

# print("=== Version 1 : OMP ===")
# s5 = sizeof((160, 8, 128))   # intermediate step 1
# s6 = sizeof((80, 160, 128))  # intermediate step 2
# s7 = sizeof((80, 160, 1280)) # final step
# total_v1 = s5 + s6 + s7
# print("1st corr (D1):", pretty_size(s5))
# print("2nd corr (D2):", pretty_size(s6))
# print("3rd corr (Ds):", pretty_size(s7))
# print("TOTAL:", pretty_size(total_v1))

# print("\n=== Version 2 : MOMP ===")
# s2 = sizeof((160, 8, 128))   # nominal_angle_1_map
# s3 = sizeof((80, 128))       # nominal_angle_2_map
# s4 = sizeof((1280, 128))     # nominal_delay_map
# total_v2 = s2 + s3 + s4
# print("1st corr (D1):", pretty_size(s2))
# print("2nd corr (D2):", pretty_size(s3))
# print("3rd corr (Ds):", pretty_size(s4))
# print("TOTAL:", pretty_size(total_v2))

# #%% Verifying torch.reshape and torch.kron follow the same order
# # Initialize tensor
# ex_h = torch.zeros(4, 2, 3, dtype=torch.int32)

# # Fill it with your pattern
# for i in range(4):       # first dimension
#     for j in range(2):   # second dimension
#         for k in range(3):  # third dimension
#             ex_h[i, j, k] = (i+1)*100 + (j+1)*10 + (k+1)

# ex_h_reshaped=ex_h.reshape(-1)
# print(ex_h_reshaped)

# ex_vec_B=torch.arange(1,5) #[1,2,3,4]
# ex_vec_M=torch.arange(1,3) #[1,2]
# ex_vec_S=torch.arange(1,4) #[1,2,3]
# ex_D_I=torch.kron(torch.kron(ex_vec_B,ex_vec_M),ex_vec_S)
# print(ex_D_I) 
# #expected output: h_reshape ([111, 112, 113, 121, 122, 123, 211, 212, 213, 221, 222, 223, 311, 312, 313, 321, 322, 323, 411, 412, 413, 421, 422, 423])
# #expected output: D_I       ([ 1,  2,  3,  2,  4,  6,  2,  4,  6,  4,  8, 12,  3,  6,  9,  6, 12, 18, 4,  8, 12,  8, 16, 24]) =product of digits in h_reshaped


