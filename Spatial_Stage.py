import numpy as np

def spherical_to_cartesian(azimuth_deg, elevation_deg, radius):
    """Converts spherical coordinates (degrees) to 3D Cartesian vectors."""
    azi = np.radians(azimuth_deg)
    ele = np.radians(elevation_deg)
    x = radius * np.cos(ele) * np.cos(azi)
    y = radius * np.cos(ele) * np.sin(azi)
    z = radius * np.sin(ele)
    return np.array([x, y, z])

def generate_steering_vectors(azimuth_deg, elevation_deg, freqs, mics_spherical, source_distance=2.0, sound_speed=343.0):
    """
    Generates complex steering vectors for a 32-channel array given an estimated DoA.
    
    Parameters:
    -----------
    azimuth_deg, elevation_deg : float
        The DoA coordinates estimated by SRP-PHAT.
    freqs : ndarray
        1D array containing the center frequencies for each STFT bin (from np.fft.rfftfreq).
    mics_spherical : dict
        The dictionary containing mic coordinates from World.py.
    source_distance : float
        Assumed distance of the target source in meters (default matches World.py).
        
    Returns:
    --------
    a_vec : ndarray
        Steering vector matrix of shape (F_dim, M) where M=32.
    """
    M = len(mics_spherical)
    F_dim = len(freqs)
    a_vec = np.zeros((F_dim, M), dtype=np.complex64)
    
    # 1. Compute hypothetical 3D coordinate of the target source
    src_pos = spherical_to_cartesian(azimuth_deg, elevation_deg, source_distance)
    
    # 2. Calculate the physical distance from the source to each microphone
    distances = []
    for name, v in mics_spherical.items():
        mic_pos = spherical_to_cartesian(v[0], v[1], v[2])
        dist = np.linalg.norm(src_pos - mic_pos)
        distances.append(dist)
    distances = np.array(distances)
    
    # 3. Compute relative propagation delay to the array origin (center)
    relative_delays = (distances - source_distance) / sound_speed
    
    # 4. Generate frequency-dependent phase transformations (Vectorized)
    # Phase lag = exp(-j * 2 * pi * f * tau)
    # Using outer product to apply across all frequencies and channels simultaneously
    phase_exponent = -1j * 2 * np.pi * np.outer(freqs, relative_delays)
    a_vec = np.exp(phase_exponent).astype(np.complex64)
    
    return a_vec

def compute_spatial_target_mask(Y_in, a_vec, sensitivity=15.0, threshold=0.45):
    """
    Computes a soft Time-Frequency Spatial Target Mask using spatial coherence similarity.
    
    Parameters:
    -----------
    Y_in : ndarray
        Input multi-channel STFT spectrum of shape (M, F_dim, T_dim)
    a_vec : ndarray
        Steering vector matrix of shape (F_dim, M)
    sensitivity : float
        Controls the steepness/sharpness of the mask boundary.
    threshold : float
        Coherence value below which a bin is categorized as noise/interference.
        
    Returns:
    --------
    mask : ndarray
        Time-frequency soft mask of shape (F_dim, T_dim) with values between [0.0, 1.0].
    """
    M, F_dim, T_dim = Y_in.shape
    
    # Transpose input for vectorized dot products: (M, F, T) -> (F, T, M)
    X = np.moveaxis(Y_in, 0, -1)
    
    # Compute instantaneous power norm for normalization
    X_norm = np.linalg.norm(X, axis=-1) + 1e-10
    
    # Expand steering vector to match frames: (F, M) -> (F, 1, M)
    a_exp = np.expand_dims(a_vec, axis=1)
    
    # Compute the projection (spatial coherence dot-product) of empirical field onto target vector
    # inner_prod results in shape (F, T)
    inner_prod = np.sum(X * np.conj(a_exp), axis=-1)
    
    # Normalize coherence metric between 0.0 and 1.0
    coherence = (np.abs(inner_prod) ** 2) / (M * (X_norm ** 2))
    
    # Map raw coherence to a smooth soft assignment mask using a generalized Sigmoid
    mask = 1.0 / (1.0 + np.exp(-sensitivity * (coherence - threshold)))
    
    return mask.astype(np.float32)