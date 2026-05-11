import numpy as np
import soundfile as sf
import os
import glob
import random
import librosa
import kagglehub
from scipy.signal import medfilt

sample_rate = 16000
sound_speed = 343.0
mic_spacing = 0.08
source_distance = 1
min_duration = 5
n_needed = 1

def coordinates(azimuth_deg, elevation_deg, distance):
    azi = np.radians(azimuth_deg)
    ele = np.radians(elevation_deg)
    x = distance * np.cos(ele) * np.cos(azi)
    y = distance * np.cos(ele) * np.sin(azi)
    z = distance * np.sin(ele)
    return np.array([x, y, z])

def audio_files(dataset, n_needed, min_duration):
    files = []
    try:
        if dataset == 'librispeech':
            path = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
            files = glob.glob(os.path.join(path, "**", "*.flac"), recursive=True)
        elif dataset == 'musan':
            path = kagglehub.dataset_download("dogrose/musan-dataset")
            files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
        else: 
            path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
            wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
            files = glob.glob(os.path.join(wav_path, "*.wav"))
        
        if len(files) == 0: 
            raise ValueError()

        random.shuffle(files)
        valid_files = []
        
        for f in files:
            if len(valid_files) >= n_needed:
                break
            try:
                info = sf.info(f)
                if info.duration >= min_duration:
                    valid_files.append(f)
            except Exception:
                continue

        if len(valid_files) < n_needed:
            if len(valid_files) == 0:
                 raise ValueError()
            while len(valid_files) < n_needed:
                valid_files += valid_files
            valid_files = valid_files[:n_needed]
        
        return valid_files
    
    except Exception:
        return []

def apply_fractional_delay(signal, delay_sec, sr):
    X = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/sr)
    X_shifted = X * np.exp(-1j * 2 * np.pi * freqs * delay_sec)
    return np.fft.irfft(X_shifted, n=len(signal))

def load_and_format_signals(file_paths, sr, duration):
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

mics = {
    'M01': np.array([0,   21, 4.2]),
    'M02': np.array([32,   0, 4.2]),
    'M03': np.array([0,  -21, 4.2]),
    'M04': np.array([328,  0, 4.2]),
    'M05': np.array([0,   58, 4.2]),
    'M06': np.array([45,  35, 4.2]),
    'M07': np.array([69,   0, 4.2]),
    'M08': np.array([45, -35, 4.2]),
    'M09': np.array([0,  -58, 4.2]),
    'M10': np.array([315, -35, 4.2]),
    'M11': np.array([291,   0, 4.2]),
    'M12': np.array([315,  35, 4.2]),
    'M13': np.array([91,  69, 4.2]),
    'M14': np.array([90,  32, 4.2]),
    'M15': np.array([90, -31, 4.2]),
    'M16': np.array([89, -69, 4.2]),
    'M17': np.array([180,  21, 4.2]),
    'M18': np.array([212,   0, 4.2]),
    'M19': np.array([180, -21, 4.2]),
    'M20': np.array([148,   0, 4.2]),
    'M21': np.array([180,  58, 4.2]),
    'M22': np.array([225,  35, 4.2]),
    'M23': np.array([249,   0, 4.2]),
    'M24': np.array([225, -35, 4.2]),
    'M25': np.array([180, -58, 4.2]),
    'M26': np.array([135, -35, 4.2]),
    'M27': np.array([111,   0, 4.2]),
    'M28': np.array([135,  35, 4.2]),
    'M29': np.array([269,  69, 4.2]),
    'M30': np.array([270,  32, 4.2]),
    'M31': np.array([270, -32, 4.2]),
    'M32': np.array([271, -69, 4.2]),
}

if __name__ == "__main__":
    a_deg, e_deg = random.randint(0, 360), random.randint(-90, 90)
    src_pos = coordinates(a_deg, e_deg, source_distance)
    
    mics_cartesian = {k: coordinates(v[0], v[1], v[2]) for k, v in mics.items()}
    
    fetched = audio_files('ljspeech', n_needed, min_duration)
    raw_signals = load_and_format_signals(fetched, sample_rate, min_duration)
    source_sig = raw_signals[0]

    total_samples = sample_rate * min_duration
    multichannel_out = np.zeros((total_samples, 32))
    
    print(f"Simulating source at Az:{a_deg}, El:{e_deg}...")
    for i, (name, mic_xyz) in enumerate(mics_cartesian.items()):
        dist = np.linalg.norm(src_pos - mic_xyz)
        delay = dist / sound_speed
        delayed = apply_fractional_delay(source_sig, delay, sample_rate)
        multichannel_out[:, i] = delayed[:total_samples]

    max_val = np.max(np.abs(multichannel_out))
    if max_val > 0:
        multichannel_out = (multichannel_out / max_val) * 0.9

    output_filename = "Simulated_Environment_32Ch.wav"
    print(f"Writing {output_filename}...")
    # info = sf.info(output_filename)
    # print(info)
    sf.write(output_filename, multichannel_out, sample_rate, subtype='PCM_16')
    print("Successfully Executed!")