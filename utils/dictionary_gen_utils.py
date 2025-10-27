import torch
def generate_DoA(nb_DoA: int):
    """
    Generate directions of arrival (DoA) vectors and corresponding angles (PyTorch version).
    
    Returns:
        DoA : [nb_DoA, 3] tensor
        angles : [nb_DoA] tensor
    """
    DoA = torch.zeros((nb_DoA, 3), dtype=torch.float64)

    cos_vals = torch.linspace(-1, 1, nb_DoA, dtype=torch.float64)
    angles = torch.flip(torch.arccos(cos_vals), dims=[0])

    DoA[:, 0] = torch.sin(angles)
    DoA[:, 1] = torch.cos(angles)
    # DoA[:, 2] remains 0

    return DoA, angles

def generate_delays(nb_delays: int,delta_f):
    """
    Generate uniformly spaced delay values based on frequency resolution.

    Args:
        nb_delays (int): Number of delay values to generate.
        delta_f (float): Frequency spacing in Hz.

    Returns:
        torch.Tensor: Delay values in seconds.
    """
    c=3e8
    max_distance = c / delta_f
    delays = torch.linspace(0, max_distance, nb_delays) / c
    return delays

def steering_vect_dict(DoA: torch.Tensor,antenna_pos: torch.Tensor,antenna_gains: torch.Tensor,antenna_coupling_coeff: torch.Tensor,lambda_: float,) -> torch.Tensor:
    """
    Compute normalized steering vector dictionary (PyTorch version).

    DoA: [A, 3] tensor of directions of arrival
    antenna_pos: [N, 3] tensor of antenna positions
    antenna_gains: [N] tensor of complex gains
    antenna_coupling: [N, N] tensor (mutual coupling matrix)
    lambda_: wavelength (float)
    """
    Nb_antenna=len(antenna_pos)
    antenna_coupling=torch.eye(Nb_antenna, dtype=torch.complex128)+ torch.diag(antenna_coupling_coeff * torch.ones(Nb_antenna-1, dtype=torch.complex128), diagonal=1)+ torch.diag(antenna_coupling_coeff * torch.ones(Nb_antenna-1, dtype=torch.complex128), diagonal=-1)
    print(antenna_coupling)
    # Exponential term: [N, A]
    expo = torch.exp(-1j * 2 * torch.pi * (1.0 / lambda_) * (antenna_pos @ DoA.T))

    # Apply gains
    dict_ = antenna_gains[:, None] * expo

    # Apply antenna coupling
    dict_ = antenna_coupling @ dict_

    # Normalize each column
    norm_factor = torch.sqrt(torch.sum(torch.abs(dict_) ** 2, dim=0))
    dict_ = dict_ / norm_factor

    return dict_

def frequency_response_vect_dict(Delays: torch.Tensor,subcarrier_freq: torch.Tensor,antenna_gains: torch.Tensor = None,) -> torch.Tensor:
    """
    Compute frequency response vector dictionary (PyTorch version).
    
    Delays : [A] tensor
        Array of delays.
    subcarrier_freq : [subc] tensor
        Array of subcarrier frequencies.
    antenna_gains : optional [A] tensor
        Complex gains (if needed, else ignored).
    """

    # Outer product: [subc, A]
    exponent = -1j * 2 * torch.pi * torch.outer(subcarrier_freq, Delays)
    dict_ = torch.exp(exponent)

    # Optionally apply gains
    if antenna_gains is not None:
        dict_ = antenna_gains[None, :] * dict_

    # Normalize each column
    norm_factor = torch.sqrt(torch.sum(torch.abs(dict_) ** 2, dim=0))
    dict_ = dict_ / norm_factor

    return dict_