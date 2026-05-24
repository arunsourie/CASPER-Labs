import os
import csv
import random
import numpy as np
import librosa
import soundfile as sf
import scipy.signal

# Standardize perceptual library fallbacks if not locally available
try:
    from pystoi import stoi
except ImportError:
    stoi = None

try:
    from pesq import pesq
except ImportError:
    pesq = None

# Import core pipeline architecture elements directly from your local modules
from World import mics_spherical, sample_rate, sound_speed, source_distance, coordinates, apply_fractional_delay, audio_files, load_and_format_signals
from SRP_PHAT import srp_phat_doa
from Spatial_Stage import generate_steering_vectors, compute_spatial_target_mask
from TFLC_Beamformer import tflc_beamformer

def updated_evaluate_func(y_est, y_tgt, y_int, fs=16000):
    """
    Calculates physical separation metrics (SNR, SIR, SINR) via signal vector 
    decomposition alongside perceptual attributes (STOI, PESQ).
    """
    metrics = {}

    # Standardize arrays to 1D Mono signatures
    if y_est.ndim > 1: y_est = y_est[:, 0]
    if y_tgt.ndim > 1: y_tgt = y_tgt[:, 0]
    if y_int.ndim > 1: y_int = y_int[:, 0]

    # Time Alignment via Cross-Correlation Cross-Check
    correlation = scipy.signal.correlate(y_tgt, y_est, mode='full')
    lags = scipy.signal.correlation_lags(len(y_tgt), len(y_est), mode='full')
    delay = lags[np.argmax(np.abs(correlation))]

    if delay > 0:
        y_est = y_est[delay:]
        y_tgt = y_tgt[:len(y_est)]
        y_int = y_int[:len(y_est)]
    elif delay < 0:
        abs_delay = abs(delay)
        y_tgt = y_tgt[abs_delay:]
        y_int = y_int[abs_delay:]
        y_est = y_est[:len(y_tgt)]

    min_len = min(len(y_est), len(y_tgt), len(y_int))
    y_est, y_tgt, y_int = y_est[:min_len], y_tgt[:min_len], y_int[:min_len]

    # Energy Component Vector Projections
    eps = 1e-10
    tgt_n = y_tgt / (np.linalg.norm(y_tgt) + eps)
    int_n = y_int / (np.linalg.norm(y_int) + eps)

    alpha = np.dot(y_est, tgt_n)
    beta = np.dot(y_est, int_n)

    e_target = alpha * tgt_n
    e_interf = beta * int_n
    e_noise = y_est - e_target - e_interf  # Uncorrelated processing artifacts/sensor noise

    P_t = np.sum(e_target ** 2)
    P_i = np.sum(e_interf ** 2)
    P_n = np.sum(e_noise ** 2)

    # Core Mathematical Audio Quality Formulations
    metrics['SIR'] = 10 * np.log10(P_t / (P_i + eps))
    metrics['SNR'] = 10 * np.log10(P_t / (P_n + eps))  # Target power vs pure noise/distortion
    metrics['SINR'] = 10 * np.log10(P_t / (P_i + P_n + eps))

    # Perceptual Intelligibility (STOI)
    metrics['STOI'] = np.nan
    if stoi is not None:
        try: metrics['STOI'] = stoi(y_tgt, y_est, fs, extended=False)
        except Exception: pass

    # Perceptual Quality Evaluation Index (PESQ)
    metrics['PESQ_WB'] = np.nan
    metrics['PESQ_NB'] = np.nan
    if pesq is not None:
        try:
            metrics['PESQ_WB'] = pesq(fs, y_tgt, y_est, 'wb')
            metrics['PESQ_NB'] = pesq(fs, y_tgt, y_est, 'nb')
        except Exception: pass

    return metrics

def main():
    num_runs = 50
    csv_filename = "pipeline_batch_test_results.csv"
    temp_wav = "temp_batch_mixed_32ch.wav"
    min_duration = 5
    
    mics_cartesian = [coordinates(v[0], v[1], v[2]) for v in mics_spherical.values()]
    
    headers = [
        "Run", "Target_Azi", "Target_Ele", "Interferer_Azi", "Interferer_Ele",
        "Est_Azi", "Est_Ele", "Azimuth_Error", "Elevation_Error",
        "SIR_dB", "SNR_dB", "SINR_dB", "STOI", "PESQ_WB", "PESQ_NB"
    ]
    
    with open(csv_filename, mode="w", newline="") as csv_file:
        csv.writer(csv_file).writerow(headers)
    
    print(f"Beginning 50-run batch validation suite. Logging targets to: {csv_filename}")
    results_accumulator = []

    for run in range(1, num_runs + 1):
        # 1. Coordinate Generation
        tgt_azi = random.randint(0, 359)
        tgt_ele = random.randint(-60, 60)
        
        while True:
            int_azi = random.randint(0, 359)
            int_ele = random.randint(-60, 60)
            if np.sqrt((tgt_azi - int_azi)**2 + (tgt_ele - int_ele)**2) > 25:
                break
                
        tgt_pos = coordinates(tgt_azi, tgt_ele, source_distance)
        int_pos = coordinates(int_azi, int_ele, source_distance)
        
        # 2. Fetch Audio Inputs
        try:
            fetched_paths = audio_files('ljspeech', n_needed=2, min_duration=min_duration)
            if len(fetched_paths) < 2:
                fetched_paths = [librosa.ex('trumpet'), librosa.ex('nutcracker')]
        except Exception:
            fetched_paths = [librosa.ex('trumpet'), librosa.ex('nutcracker')]
            
        raw_signals = load_and_format_signals(fetched_paths, sample_rate, min_duration)
        target_sig, interferer_sig = raw_signals[0], raw_signals[1]
        
        # 3. Simulate Spatial Propagation Matrix (32 channels)
        total_samples = sample_rate * min_duration
        multichannel_tgt = np.zeros((total_samples, 32))
        multichannel_int = np.zeros((total_samples, 32))
        
        mics_cart_dict = {k: coordinates(v[0], v[1], v[2]) for k, v in mics_spherical.items()}
        for i, (name, mic_xyz) in enumerate(mics_cart_dict.items()):
            multichannel_tgt[:, i] = apply_fractional_delay(target_sig, np.linalg.norm(tgt_pos - mic_xyz) / sound_speed, sample_rate)[:total_samples]
            multichannel_int[:, i] = apply_fractional_delay(interferer_sig, np.linalg.norm(int_pos - mic_xyz) / sound_speed, sample_rate)[:total_samples]
            
        mixed_array = multichannel_tgt + multichannel_int
        mixed_array += np.random.normal(0, 0.001, mixed_array.shape)  # Inject sensor floor noise
        
        max_val = np.max(np.abs(mixed_array))
        if max_val > 0:
            mixed_array = (mixed_array / max_val) * 0.9
            multichannel_tgt = (multichannel_tgt / max_val) * 0.9
            multichannel_int = (multichannel_int / max_val) * 0.9

        sf.write(temp_wav, mixed_array, sample_rate, subtype='PCM_16')
        
        # 4. DoA Estimation using the local SRP_PHAT.py module
        est_azi, est_ele = srp_phat_doa(temp_wav, mics_cartesian, sr=sample_rate)
        
        azi_error = min(abs(est_azi - tgt_azi), 360 - abs(est_azi - tgt_azi))
        ele_error = abs(est_ele - tgt_ele)
        
        # 5. Transform to TF STFT Representation Domain
        n_fft, hop_length = 2048, 512
        mixed_array_t = mixed_array.T
        stft_channels = [librosa.stft(mixed_array_t[m], n_fft=n_fft, hop_length=hop_length) for m in range(32)]
        Y_in = np.stack(stft_channels, axis=0)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        
        # 6. Spatial Mask Construction & Optimization Beamforming Enhancement
        a_vec = generate_steering_vectors(est_azi, est_ele, freqs, mics_spherical, source_distance, sound_speed)
        target_mask = compute_spatial_target_mask(Y_in, a_vec, sensitivity=15.0, threshold=0.45)
        Y_enhanced, _ = tflc_beamformer(Y_in, target_mask, a_vec=a_vec, iterations=15)
        y_est = librosa.istft(Y_enhanced, hop_length=hop_length)
        
        # 7. Metrics Extraction
        metrics = updated_evaluate_func(y_est, multichannel_tgt[:, 0], multichannel_int[:, 0], fs=sample_rate)
        
        print(f"Run {run:02d} -> Error: {azi_error:.1f}° | SIR: {metrics['SIR']:5.2f}dB | SNR: {metrics['SNR']:5.2f}dB | STOI: {metrics['STOI']:.3f}")
        
        row_data = [
            run, tgt_azi, tgt_ele, int_azi, int_ele, est_azi, est_ele, azi_error, ele_error,
            round(metrics['SIR'], 2), round(metrics['SNR'], 2), round(metrics['SINR'], 2),
            round(metrics['STOI'], 4) if not np.isnan(metrics['STOI']) else "NaN",
            round(metrics['PESQ_WB'], 4) if not np.isnan(metrics['PESQ_WB']) else "NaN",
            round(metrics['PESQ_NB'], 4) if not np.isnan(metrics['PESQ_NB']) else "NaN"
        ]
        results_accumulator.append(row_data)
        
        with open(csv_filename, mode="a", newline="") as csv_file:
            csv.writer(csv_file).writerow(row_data)

    if os.path.exists(temp_wav):
        os.remove(temp_wav)
        
    print("\n=== BATCH RUN FINISHED ===")
    summary_matrix = np.array(results_accumulator, dtype=object)
    print(f"Mean Spatial Tracking Error: {np.mean(summary_matrix[:, 7]):.2f}°")
    print(f"Mean Performance Ratios    : SIR = {np.mean(summary_matrix[:, 9]):.2f} dB | SNR = {np.mean(summary_matrix[:, 10]):.2f} dB")

if __name__ == "__main__":
    main()