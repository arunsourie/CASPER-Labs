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
n_needed = 3

def coordinates(angle_deg, distance):
    angle_rad = np.radians(angle_deg)
    x = distance * np.sin(angle_rad)
    y = distance * np.cos(angle_rad)
    return np.array([x, y])

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

def generate_ground_truth(sig_t, sig_a, sig_b, frame_size=80, margin_db=3.0, kernel_size=21):
    num_frames = len(sig_a) // frame_size
    raw_gt = []
    p_t_list, p_a_list, p_b_list = [], [], []

    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        
        f_t = sig_t[start:end]
        f_a = sig_a[start:end]
        f_b = sig_b[start:end]
        
        p_t = np.sum(f_t**2) / frame_size
        p_a = np.sum(f_a**2) / frame_size
        p_b = np.sum(f_b**2) / frame_size
        
        p_t = max(p_t, 1e-10)
        p_a = max(p_a, 1e-10)
        p_b = max(p_b, 1e-10)
        
        p_t_list.append(p_t)
        p_a_list.append(p_a)
        p_b_list.append(p_b)
        
        diff_db = 10 * np.log10(p_a / p_b)
        
        if diff_db > margin_db:
            raw_gt.append(1)
        elif diff_db < -margin_db:
            raw_gt.append(2)
        else:
            raw_gt.append(0)
            
    smoothed_gt = medfilt(raw_gt, kernel_size=kernel_size).astype(int)
    return smoothed_gt, np.array(p_t_list), np.array(p_a_list), np.array(p_b_list)

mics = {
    'M1': np.array([-mic_spacing/2, 0.0]),
    'M2': np.array([mic_spacing/2, 0.0])
}

sources = {
    'Target': coordinates(0, source_distance),
    'Intf_1': coordinates(60, source_distance),
    'Intf_2': coordinates(-60, source_distance)
}

if __name__ == "__main__":
    dataset_choice = 'ljspeech' 
    fetched_files = audio_files(dataset_choice, n_needed, min_duration)
    
    if len(fetched_files) == n_needed:
        raw_signals = load_and_format_signals(fetched_files, sample_rate, min_duration)
        
        signal_map = {
            'Target': raw_signals[0],
            'Intf_1': raw_signals[1],
            'Intf_2': raw_signals[2]
        }

        total_samples = sample_rate * min_duration
        mix_m1 = np.zeros(total_samples)
        mix_m2 = np.zeros(total_samples)

        for name, sig in signal_map.items():
            pos = sources[name]
            
            dist1 = np.linalg.norm(pos - mics['M1'])
            dist2 = np.linalg.norm(pos - mics['M2'])
            t1 = dist1 / sound_speed
            t2 = dist2 / sound_speed
            
            mix_m1 += apply_fractional_delay(sig, t1, sample_rate)
            mix_m2 += apply_fractional_delay(sig, t2, sample_rate)

        stereo_out = np.vstack((mix_m1, mix_m2)).T
        output_filename = "Simulated_Environment.wav"
        
        print(f"Writing {output_filename}...")
        sf.write(output_filename, stereo_out, sample_rate)
        
        print("Writing Ground_Truth.npz...")
        gt_labels, pt, pa, pb = generate_ground_truth(signal_map['Target'], signal_map['Intf_1'], signal_map['Intf_2'])
        np.savez('Ground_Truth.npz', labels=gt_labels, p_t=pt, p_a=pa, p_b=pb)
        
        print("Successfully Executed!")