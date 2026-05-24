import numpy as np
import scipy.signal

# Standardize perceptual library fallbacks if not installed
try:
    from pystoi import stoi
except ImportError:
    stoi = None

try:
    from pesq import pesq
except ImportError:
    pesq = None

def evaluate_func(y_est, y_tgt, y_int, fs=16000):
    """
    Calculates SIR, SINR, STOI, and PESQ between isolated and reference tracks.
    
    Parameters:
    -----------
    y_est : ndarray - The isolated output waveform from the TFLC beamformer.
    y_tgt : ndarray - The original clean target source waveform (Ground Truth).
    y_int : ndarray - The interference/noise source waveform.
    fs    : int     - Audio sample rate (default 16000 matches pipeline).
    
    Returns:
    --------
    metrics : dict  - Dictionary containing all calculated performance scores.
    """
    metrics = {}

    # 1. Ensure signals are 1D arrays (Mono)
    if y_est.ndim > 1: y_est = y_est[:, 0]
    if y_tgt.ndim > 1: y_tgt = y_tgt[:, 0]
    if y_int.ndim > 1: y_int = y_int[:, 0]

    # 2. Time Alignment via Cross-Correlation
    # Equivalent to MATLAB's [c, lags] = xcorr(y_tgt, y_est)
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

    # Truncate to matching minimum length
    min_len = min(len(y_est), len(y_tgt), len(y_int))
    y_est = y_est[:min_len]
    y_tgt = y_tgt[:min_len]
    y_int = y_int[:min_len]

    # 3. Physics Metrics (Decomposition & Power Analysis)
    eps = 1e-10
    tgt_n = y_tgt / (np.linalg.norm(y_tgt) + eps)
    int_n = y_int / (np.linalg.norm(y_int) + eps)

    # Spatial-acoustic scaling projections
    alpha = np.dot(y_est, tgt_n)
    beta = np.dot(y_est, int_n)

    e_target = alpha * tgt_n
    e_interf = beta * int_n
    e_noise = y_est - e_target - e_interf

    P_t = np.sum(e_target ** 2)
    P_i = np.sum(e_interf ** 2)
    P_n = np.sum(e_noise ** 2)

    # Signal-to-Interference & Signal-to-Interference-plus-Noise Ratios
    metrics['SIR'] = 10 * np.log10(P_t / (P_i + eps))
    metrics['SINR'] = 10 * np.log10(P_t / (P_i + P_n + eps))

    # 4. Perceptual Evaluation (STOI)
    metrics['STOI'] = np.nan
    if stoi is not None:
        try:
            metrics['STOI'] = stoi(y_tgt, y_est, fs, extended=False)
        except Exception:
            pass

    # 5. Perceptual Evaluation (PESQ - Wideband & Narrowband)
    metrics['PESQ_WB'] = np.nan
    metrics['PESQ_NB'] = np.nan
    if pesq is not None:
        try:
            metrics['PESQ_WB'] = pesq(fs, y_tgt, y_est, 'wb')
            metrics['PESQ_NB'] = pesq(fs, y_tgt, y_est, 'nb')
        except Exception:
            pass

    # Print summary output keeping original formatting style
    print(f"SIR: {metrics['SIR']:5.2f} dB | SINR: {metrics['SINR']:5.2f} dB | "
          f"STOI: {metrics['STOI']:.4f} | PESQ_WB: {metrics['PESQ_WB']:.4f} | PESQ_NB: {metrics['PESQ_NB']:.4f}")

    return metrics