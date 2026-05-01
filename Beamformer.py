import numpy as np

def get_steering_vector(delay, freqs):
    a_mic1 = np.ones_like(freqs, dtype=complex)
    a_mic2 = np.exp(1j * 2 * np.pi * freqs * delay)
    return np.array([a_mic1, a_mic2])

def calculate_mvdr_power(frame_2ch, steering_vector, diag_load=1e-3, sr=16000):
    window = np.hanning(frame_2ch.shape[0])
    frame_windowed = frame_2ch * window[:, np.newaxis]
    
    X = np.fft.rfft(frame_windowed, axis=0).T
    total_power = 0.0
    num_bins = X.shape[1]
    
    # Calculate actual frequencies for each bin
    freqs = np.fft.rfftfreq(frame_2ch.shape[0], d=1.0/sr)
    
    for f in range(num_bins):
        # NEW: Skip frequencies outside the reliable speech/array geometry band
        if freqs[f] < 300 or freqs[f] > 2500:
            continue
            
        x_f = X[:, f].reshape(2, 1)
        a_f = steering_vector[:, f].reshape(2, 1)
        
        R = x_f @ x_f.conj().T
        R += diag_load * np.eye(2)
        
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            continue
            
        denominator = (a_f.conj().T @ R_inv @ a_f)[0, 0]
        if np.abs(denominator) > 1e-10:
            power_f = np.real(1.0 / denominator)
            total_power += power_f
            
    return total_power