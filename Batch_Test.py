import numpy as np
import pandas as pd
import random
import soundfile as sf
import time
import os

# Import your existing scripts as modules
import World
import Ambisonics
import SRP_PHAT

def run_batch_test(num_iterations=50):
    results = []

    print("Fetching base audio signal...")
    # Fetch the audio once using World.py's logic
    fetched = World.audio_files('ljspeech', World.n_needed, World.min_duration)
    raw_signals = World.load_and_format_signals(fetched, World.sample_rate, World.min_duration)
    source_sig = raw_signals[0]

    # Pre-compute microphone coordinates for both generation and SRP-PHAT
    print("Preparing microphone arrays...")
    mics_cartesian_dict = {k: World.coordinates(v[0], v[1], v[2]) for k, v in World.mics_spherical.items()}
    
    # SRP-PHAT expects a list of coordinates instead of a dictionary
    mics_cart_list = [World.coordinates(v[0], v[1], v[2]) for v in SRP_PHAT.mics_spherical.values()]

    print(f"Starting {num_iterations}-batch DoA estimation test...")

    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        
        # 1. GENERATE NEW GROUND TRUTH
        a_deg = random.randint(0, 360)
        e_deg = random.randint(-90, 90)
        src_pos = World.coordinates(a_deg, e_deg, World.source_distance)
        
        # 2. SIMULATE 32-CHANNEL AUDIO
        total_samples = World.sample_rate * World.min_duration
        multichannel_out = np.zeros((total_samples, 32))
        
        for idx, (name, mic_xyz) in enumerate(mics_cartesian_dict.items()):
            dist = np.linalg.norm(src_pos - mic_xyz)
            delay = dist / World.sound_speed
            delayed = World.apply_fractional_delay(source_sig, delay, World.sample_rate)
            multichannel_out[:, idx] = delayed[:total_samples]

        # Normalize and save the temporary audio file for this batch
        max_val = np.max(np.abs(multichannel_out))
        if max_val > 0:
            multichannel_out = (multichannel_out / max_val) * 0.9

        wav_file = "Batch_Temp_32Ch.wav"
        sf.write(wav_file, multichannel_out, World.sample_rate, subtype='PCM_16')
        
        # 3. RUN AMBISONICS ESTIMATION
        start_time = time.time()
        amb_azi, amb_ele = Ambisonics.estimate_doa(wav_file)
        amb_time = time.time() - start_time
        
        # Calculate Ambisonics Error (accounting for 360-degree wrap-around)
        amb_azi_err = min(abs(amb_azi - a_deg), 360 - abs(amb_azi - a_deg))
        amb_ele_err = abs(amb_ele - e_deg)

        # 4. RUN SRP-PHAT ESTIMATION
        start_time = time.time()
        srp_azi, srp_ele = SRP_PHAT.srp_phat_doa(wav_file, mics_cart_list, sr=World.sample_rate)
        srp_time = time.time() - start_time
        
        # Calculate SRP-PHAT Error
        srp_azi_err = min(abs(srp_azi - a_deg), 360 - abs(srp_azi - a_deg))
        srp_ele_err = abs(srp_ele - e_deg)

        # Print quick feedback to console
        print(f"Ground Truth -> Az: {a_deg}°, El: {e_deg}°")
        print(f"Ambisonics   -> Az: {amb_azi:.1f}° (Err: {amb_azi_err:.1f}°)")
        print(f"SRP-PHAT     -> Az: {srp_azi:.1f}° (Err: {srp_azi_err:.1f}°)")

        # 5. STORE RESULTS IN DICTIONARY
        results.append({
            "Iteration": i + 1,
            "True_Azimuth": a_deg,
            "True_Elevation": e_deg,
            "Ambisonics_Azimuth": round(amb_azi, 2),
            "Ambisonics_Elevation": round(amb_ele, 2),
            "Ambisonics_Azi_Error": round(amb_azi_err, 2),
            "Ambisonics_Ele_Error": round(amb_ele_err, 2),
            "Ambisonics_Time_s": round(amb_time, 4),
            "SRP_PHAT_Azimuth": round(srp_azi, 2),
            "SRP_PHAT_Elevation": round(srp_ele, 2),
            "SRP_PHAT_Azi_Error": round(srp_azi_err, 2),
            "SRP_PHAT_Ele_Error": round(srp_ele_err, 2),
            "SRP_PHAT_Time_s": round(srp_time, 4)
        })

    # 6. EXPORT LOGS TO CSV
    df = pd.DataFrame(results)
    csv_filename = "DoA_Estimation.csv"
    df.to_csv(csv_filename, index=False)
    
    # Cleanup the temporary wave file
    if os.path.exists("Batch_Temp_32Ch.wav"):
        os.remove("Batch_Temp_32Ch.wav")

    print(f"\nBatch test completed! Results saved to {csv_filename}")
    
    # Print Final Summary Statistics
    print("\n--- Summary Statistics (Average Errors) ---")
    print(f"Ambisonics -> Azimuth: {df['Ambisonics_Azi_Error'].mean():.2f}°, Elevation: {df['Ambisonics_Ele_Error'].mean():.2f}°")
    print(f"SRP-PHAT   -> Azimuth: {df['SRP_PHAT_Azi_Error'].mean():.2f}°, Elevation: {df['SRP_PHAT_Ele_Error'].mean():.2f}°")
    print(f"Avg Time   -> Ambisonics: {df['Ambisonics_Time_s'].mean():.4f}s, SRP-PHAT: {df['SRP_PHAT_Time_s'].mean():.4f}s")

if __name__ == "__main__":
    # Ensure pandas is installed: pip install pandas
    run_batch_test(50)