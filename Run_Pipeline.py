import numpy as np
import librosa
import soundfile as sf
import os

# Import modules and configurations directly from your current pipeline files
from World import mics_spherical, sample_rate, sound_speed, source_distance, coordinates
from SRP_PHAT import srp_phat_doa
from Spatial_Stage import generate_steering_vectors, compute_spatial_target_mask
from TFLC_Beamformer import tflc_beamformer

def main():
    wav_file = "Simulated_Environment_32Ch.wav"
    gt_file = "Ground_Truth.npz"
    
    if not os.path.exists(wav_file):
        print(f"Error: '{wav_file}' not found. Please run 'World.py' first to simulate the environment.")
        return

    print("\n=== STEP 1: Loading 32-Channel Audio Signal ===")
    # Load multi-channel audio (mono=False preserves all 32 channels)
    audio, sr = librosa.load(wav_file, sr=sample_rate, mono=False)
    print(f"Loaded audio array shape: {audio.shape} (Channels: {audio.shape[0]}, Samples: {audio.shape[1]})")
    
    # Cross-verify with ground truth if available
    if os.path.exists(gt_file):
        gt = np.load(gt_file)
        print(f"Ground Truth Location -> Azimuth: {gt['azimuth']}°, Elevation: {gt['elevation']}°")

    print("\n=== STEP 2: Running DoA Estimation (SRP-PHAT) ===")
    # Convert spherical mic coordinates to Cartesian vectors as required by SRP-PHAT
    mics_cart = [coordinates(v[0], v[1], v[2]) for v in mics_spherical.values()]
    
    # Estimate Direction of Arrival (DoA) using the SRP-PHAT grid search
    est_azi, est_ele = srp_phat_doa(wav_file, mics_cart)
    print(f"Estimated Location -> Azimuth: {est_azi:.2f}°, Elevation: {est_ele:.2f}°")

    print("\n=== STEP 3: Transforming Audio to STFT (Time-Frequency) Domain ===")
    n_fft = 2048
    hop_length = 512
    
    # Compute Short-Time Fourier Transform (STFT) across all 32 channels
    stft_list = [librosa.stft(audio[m], n_fft=n_fft, hop_length=hop_length) for m in range(32)]
    Y_in = np.stack(stft_list, axis=0)  # Shape: (32, F_dim, T_dim)
    print(f"STFT Matrix Tensor Shape: {Y_in.shape} (Mics, Freq Bins, Time Frames)")
    
    # Calculate frequency bins map for steering vector phase math
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    print("\n=== STEP 4: Spatial Stage Processing ===")
    print("Generating steering vectors and creating spatial target mask...")
    # Map the estimated DoA angle onto frequency-dependent phase transformations across the 32 microphones
    a_vec = generate_steering_vectors(
        azimuth_deg=est_azi, 
        elevation_deg=est_ele, 
        freqs=freqs, 
        mics_spherical=mics_spherical, 
        source_distance=source_distance, 
        sound_speed=sound_speed
    )
    
    # Compute the soft time-frequency target mask based on spatial coherence alignment
    target_mask = compute_spatial_target_mask(Y_in, a_vec, sensitivity=15.0, threshold=0.45)
    print(f"Spatial Target Mask created successfully! Shape: {target_mask.shape}")

    print("\n=== STEP 5: Running TFLC Adaptive Beamformer ===")
    # Execute the optimization loops to isolate the sound coming from look-direction
    Y_enhanced, debug_data = tflc_beamformer(Y_in, target_mask, a_vec=a_vec, iterations=15)

    print("\n=== STEP 6: Reconstructing Enhanced Signal ===")
    # Reconstruct the audio track from enhanced complex spectrum back into a 1D waveform
    enhanced_audio = librosa.istft(Y_enhanced, hop_length=hop_length)
    
    # Normalize maximum amplitude to prevent digital clipping
    if np.max(np.abs(enhanced_audio)) > 0:
        enhanced_audio = (enhanced_audio / np.max(np.abs(enhanced_audio))) * 0.9

    output_path = "Isolated_Output.wav"
    sf.write(output_path, enhanced_audio, sample_rate)
    print(f"\nSUCCESS: Beamformed sound extracted and saved to: '{output_path}'")

if __name__ == "__main__":
    main()