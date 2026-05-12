import numpy as np
import librosa
import os

sound_speed = 343.0
mic_radius_m = 0.042

mics_spherical = {
    'M01': [0, 21, mic_radius_m],    'M02': [32, 0, mic_radius_m],
    'M03': [0, -21, mic_radius_m],   'M04': [328, 0, mic_radius_m],
    'M05': [0, 58, mic_radius_m],    'M06': [45, 35, mic_radius_m],
    'M07': [69, 0, mic_radius_m],    'M08': [45, -35, mic_radius_m],
    'M09': [0, -58, mic_radius_m],   'M10': [315, -35, mic_radius_m],
    'M11': [291, 0, mic_radius_m],   'M12': [315, 35, mic_radius_m],
    'M13': [91, 69, mic_radius_m],   'M14': [90, 32, mic_radius_m],
    'M15': [90, -31, mic_radius_m],  'M16': [89, -69, mic_radius_m],
    'M17': [180, 21, mic_radius_m],  'M18': [212, 0, mic_radius_m],
    'M19': [180, -21, mic_radius_m], 'M20': [148, 0, mic_radius_m],
    'M21': [180, 58, mic_radius_m],  'M22': [225, 35, mic_radius_m],
    'M23': [249, 0, mic_radius_m],   'M24': [225, -35, mic_radius_m],
    'M25': [180, -58, mic_radius_m], 'M26': [135, -35, mic_radius_m],
    'M27': [111, 0, mic_radius_m],   'M28': [135, 35, mic_radius_m],
    'M29': [269, 69, mic_radius_m],  'M30': [270, 32, mic_radius_m],
    'M31': [270, -32, mic_radius_m], 'M32': [271, -69, mic_radius_m],
}

def coordinates(azimuth_deg, elevation_deg, distance):
    azi = np.radians(azimuth_deg)
    ele = np.radians(elevation_deg)
    x = distance * np.cos(ele) * np.cos(azi)
    y = distance * np.cos(ele) * np.sin(azi)
    z = distance * np.sin(ele)
    return np.array([x, y, z])

def compute_gcc_phat(sig1, sig2, n_fft=2048):
    """Computes the GCC-PHAT cross-correlation array for two signals."""
    X1 = np.fft.rfft(sig1, n=n_fft)
    X2 = np.fft.rfft(sig2, n=n_fft)
    
    cross_spec = X1 * np.conj(X2)
    phat_weight = np.maximum(np.abs(cross_spec), 1e-10) # Prevent divide by zero
    gcc_phat_freq = cross_spec / phat_weight
    
    gcc_phat_time = np.fft.irfft(gcc_phat_freq, n=n_fft)
    
    return np.fft.fftshift(gcc_phat_time)

def srp_phat_doa(audio_path, mics_cartesian, sr=16000):
    audio, _ = librosa.load(audio_path, sr=sr, mono=False)
    
    ref_mic_idx = 0
    ref_mic_pos = mics_cartesian[ref_mic_idx]
    
    n_fft = 2048
    center_idx = n_fft // 2
    
    gcc_phat_arrays = []
    mic_vectors = []
    
    for m in range(1, 32):
        gcc = compute_gcc_phat(audio[ref_mic_idx], audio[m], n_fft)
        gcc_phat_arrays.append(gcc)
        mic_vectors.append(mics_cartesian[m] - ref_mic_pos)
        
    # Define our Search Grid (Resolution: 5 degrees)
    azimuths = np.arange(0, 360, 5)
    elevations = np.arange(-90, 91, 5)
    
    best_power = -np.inf
    best_doa = (0, 0)
    
    for azi in azimuths:
        for ele in elevations:
            target_vec = coordinates(azi, ele, 1.0) 
            
            total_power = 0
            for i in range(len(gcc_phat_arrays)):
                dist_diff = np.dot(target_vec, mic_vectors[i])
                
                delay_sec = dist_diff / sound_speed
                delay_samples = delay_sec * sr
                
                # Find the exact index in the GCC-PHAT array
                # (Rounding to nearest integer sample)
                lookup_idx = int(np.round(center_idx + delay_samples))
                
                # Add the power at this exact delay to our total
                total_power += gcc_phat_arrays[i][lookup_idx]
                
            if total_power > best_power:
                best_power = total_power
                best_doa = (azi, ele)
                
    return best_doa

if __name__ == "__main__":
    wav_file = "Simulated_Environment_32Ch.wav"
    gt_file = "Ground_Truth.npz"
    
    if not os.path.exists(wav_file):
        print("Error: Run World.py first to generate the 32-channel audio.")
    else:
        mics_cart = [coordinates(v[0], v[1], v[2]) for v in mics_spherical.values()]
        
        est_azi, est_ele = srp_phat_doa(wav_file, mics_cart)
        
        try:
            gt = np.load(gt_file)
            true_azi, true_ele = gt['azimuth'], gt['elevation']
            
            azi_error = min(abs(est_azi - true_azi), 360 - abs(est_azi - true_azi))
            ele_error = abs(est_ele - true_ele)
            
            print("--- DoA Estimation Results (SRP-PHAT) ---")
            print(f"Estimated:  Azimuth: {est_azi:3.1f}°, Elevation: {est_ele:3.1f}°")
            print(f"Actual:     Azimuth: {float(true_azi):3.1f}°, Elevation: {float(true_ele):3.1f}°")
            print(f"Error:      Azimuth Error: {azi_error:3.1f}°, Elevation Error: {ele_error:3.1f}°")
            
        except FileNotFoundError:
            print(f"\nEstimated Azimuth: {est_azi}°, Elevation: {est_ele}° (Ground truth not found)")