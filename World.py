import numpy as np
import soundfile as sf
import os
import glob
import random
import librosa
import kagglehub

# --- Physical Constants ---
sample_rate = 16000
sound_speed = 343.0
source_distance = 2.0  # Meters
mic_radius_m = 0.042   # 4.2cm converted to meters
min_duration = 5
n_needed = 1

def coordinates(azimuth_deg, elevation_deg, distance):
    """Converts spherical coordinates (degrees) to Cartesian meters."""
    azi = np.radians(azimuth_deg)
    ele = np.radians(elevation_deg)
    x = distance * np.cos(ele) * np.cos(azi)
    y = distance * np.cos(ele) * np.sin(azi)
    z = distance * np.sin(ele)
    return np.array([x, y, z])

def audio_files(dataset, n_needed, min_duration):
    """Fetches clean mono audio for the source signal."""
    try:
        path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
        wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
        files = glob.glob(os.path.join(wav_path, "*.wav"))
        random.shuffle(files)
        return files[:n_needed]
    except Exception:
        return [librosa.ex('trumpet')]

def apply_fractional_delay(signal, delay_sec, sr):
    """Applies high-precision sub-sample delays via FFT phase shifting."""
    X = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/sr)
    # Apply phase shift: e^(-j * 2pi * f * tau)
    X_shifted = X * np.exp(-1j * 2 * np.pi * freqs * delay_sec)
    return np.fft.irfft(X_shifted, n=len(signal))

def load_and_format_signals(file_paths, sr, duration):
    """Standardizes audio duration and sample rate."""
    target_len = sr * duration
    signals = []
    for path in file_paths:
        audio, _ = librosa.load(path, sr=sr)
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
        signals.append(audio)
    return signals

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

if __name__ == "__main__":
    a_deg = random.randint(0, 359)
    e_deg = random.randint(-90, 90)
    src_pos = coordinates(a_deg, e_deg, source_distance)
    
    mics_cartesian = {k: coordinates(v[0], v[1], v[2]) for k, v in mics_spherical.items()}
    
    fetched = audio_files('ljspeech', n_needed, min_duration)
    raw_signals = load_and_format_signals(fetched, sample_rate, min_duration)
    source_sig = raw_signals[0]

    total_samples = sample_rate * min_duration
    multichannel_out = np.zeros((total_samples, 32))
    
    print(f"Simulating source at Azimuth: {a_deg}°, Elevation: {e_deg}°...")
    for i, (name, mic_xyz) in enumerate(mics_cartesian.items()):
        dist = np.linalg.norm(src_pos - mic_xyz)
        delay = dist / sound_speed
        
        delayed_channel = apply_fractional_delay(source_sig, delay, sample_rate)
        multichannel_out[:, i] = delayed_channel[:total_samples]

    max_val = np.max(np.abs(multichannel_out))
    if max_val > 0:
        multichannel_out = (multichannel_out / max_val) * 0.9

    output_filename = "Simulated_Environment_32Ch.wav"
    print(f"Writing {output_filename}...")
    sf.write(output_filename, multichannel_out, sample_rate, subtype='PCM_16')

    np.savez('Ground_Truth.npz', azimuth=a_deg, elevation=e_deg)

    print("Successfully Executed!")