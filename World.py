import numpy as np
import soundfile as sf
import os
import glob
import random
import argparse
import librosa
import kagglehub
import pyroomacoustics as pra
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

#Parameters
sample_rate = 16000
sound_speed = 343.0
mic_spacing = 0.08
source_distance = 1
min_duration = 5
n_needed = 3

#Functions
def coordinates(angle_deg, distance):
    angle_rad = np.radians(angle_deg)
    x = distance*np.sin(angle_rad)
    y = distance*np.cos(angle_rad)
    return np.array([x, y])

def audio_files(dataset, n_needed, min_duration):
    print(f"--- Fetching {n_needed} files (>= {min_duration}s) from: {dataset} ---")
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
            raise ValueError(f"No files found for {dataset}")

        random.shuffle(files)
        valid_files = []
        print("Scanning files for duration requirements...")
        
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
            print(f"Warning: Only found {len(valid_files)} valid files >= {min_duration}s. Duplicating.")
            if len(valid_files) == 0:
                 raise ValueError("No files found meeting the duration requirement.")
            while len(valid_files) < n_needed:
                valid_files += valid_files
            valid_files = valid_files[:n_needed]
        
        return valid_files
    
    except Exception as e:
        print(f"Error getting data: {e}")
        return []

def sample_delay(source_pos, mic_pos):
    distance = np.linalg.norm(source_pos - mic_pos)
    time_delay = distance/sound_speed
    return int(np.round(time_delay*sample_rate))

def apply_delay(signal, delay_samples):
    if delay_samples == 0:
        return signal
    padded_signal = np.pad(signal, (delay_samples, 0), mode='constant')
    return padded_signal[:len(signal)]

def load_and_format_signals(file_paths, sr, duration):
    target_len = sr*duration
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
    'M1':np.array([-mic_spacing/2, 0.0]),
    'M2':np.array([mic_spacing/2, 0.0])
    }

sources = {
    'Target':coordinates(0, source_distance),
    'Intf_1':coordinates(60, source_distance),
    'Intf_2':coordinates(-60, source_distance)
    }

#Main Module
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
            d1 = sample_delay(pos, mics['M1'])
            d2 = sample_delay(pos, mics['M2'])
            
            mix_m1 += apply_delay(sig, d1)
            mix_m2 += apply_delay(sig, d2)

        stereo_out = np.vstack((mix_m1, mix_m2)).T
        output_filename = "Simulated_Environment.wav"
        
        sf.write(output_filename, stereo_out, sample_rate)
        print(f"Simulation complete. Mixed audio saved to {output_filename}")
    else:
        print("Failed to fetch required audio files. Simulation aborted.")