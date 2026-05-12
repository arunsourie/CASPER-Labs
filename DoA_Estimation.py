import numpy as np
import librosa
import soundfile as sf

def estimate_doa(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000, mono=False)
    
    # Ambisonic Encoding (First Order - FOA)
    # In a real scenario, you use an "Encoding Matrix" specific to your mic geometry.
    # For a simple simulation, we approximate the W, X, Y, Z components:
    # W = Sum of all mics (Pressure)
    # X, Y, Z = Weighted difference based on mic positions
    
    W = np.sum(audio, axis=0)
    
    X = np.mean(audio[:16, :], axis=0) - np.mean(audio[16:, :], axis=0)
    Y = np.mean(audio[8:24, :], axis=0) - np.mean(np.concatenate((audio[:8, :], audio[24:, :])), axis=0)
    Z = np.mean(audio[::2, :], axis=0) - np.mean(audio[1::2, :], axis=0)

    I_x = np.sum(W * X)
    I_y = np.sum(W * Y)
    I_z = np.sum(W * Z)

    azimuth = np.degrees(np.arctan2(I_y, I_x))
    
    horizontal_dist = np.sqrt(I_x**2 + I_y**2)
    elevation = np.degrees(np.arctan2(I_z, horizontal_dist))

    return azimuth, elevation

if __name__ == "__main__":
    wav_file = "Simulated_Environment_32Ch.wav"
    
    try:
        est_azi, est_ele = estimate_doa(wav_file)
        
        # Load Ground Truth to compare
        gt = np.load('Ground_Truth.npz')
        
        print("--- DoA Estimation Results ---")
        print(f"Estimated: Azimuth: {est_azi:2.1f}°, Elevation: {est_ele:2.1f}°")
        print(f"Actual:    Azimuth: {gt['azimuth']:2.1f}°, Elevation: {gt['elevation']:2.1f}°")
        print(f"Error:     {abs(est_azi - gt['azimuth']):2.1f}°")
    except FileNotFoundError:
        print("Please run the simulator (World.py) first to generate the input file.")