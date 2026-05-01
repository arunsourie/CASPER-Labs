import sys
import numpy as np
import soundfile as sf
from scipy.signal import medfilt

# Import from our custom modules
from Beamformer import get_steering_vector, calculate_mvdr_power
from Metrics import generate_outputs

# Environment Parameters
SAMPLE_RATE = 16000
SOUND_SPEED = 343.0
MIC_SPACING = 0.08
FRAME_SIZE = 80      # 5ms window
MARGIN_DB = 3.0      # Margin of Equivalence

def process_file(filepath):
    audio, sr = sf.read(filepath)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected sample rate {SAMPLE_RATE}Hz, got {sr}Hz")
    
    num_frames = len(audio) // FRAME_SIZE
    
    # Calculate physical TDOA delays
    delay_a = (MIC_SPACING * np.sin(np.radians(60))) / SOUND_SPEED
    delay_b = (MIC_SPACING * np.sin(np.radians(-60))) / SOUND_SPEED
    
    # Setup frequencies for steering vectors
    freqs = np.fft.rfftfreq(FRAME_SIZE, d=1.0/SAMPLE_RATE)
    freqs[0] = 1e-6 # Avoid divide by zero
    
    steer_a = get_steering_vector(delay_a, freqs)
    steer_b = get_steering_vector(delay_b, freqs)
    
    raw_predictions = []
    
    # Frame-by-Frame Processing Loop
    for i in range(num_frames):
        start = i * FRAME_SIZE
        end = start + FRAME_SIZE
        frame = audio[start:end]
        
        # Calculate raw spatial power using MVDR module
        p_a = calculate_mvdr_power(frame, steer_a)
        p_b = calculate_mvdr_power(frame, steer_b)
        
        if p_a < 1e-10 and p_b < 1e-10:
            raw_predictions.append(0)
            continue
            
        p_a = max(p_a, 1e-10)
        p_b = max(p_b, 1e-10)
        
        # Calculate Relative Power Distribution
        diff_db = 10 * np.log10(p_a / p_b)
        
        # Apply 3dB Margin of Equivalence
        if diff_db > MARGIN_DB:
            raw_predictions.append(1)
        elif diff_db < -MARGIN_DB:
            raw_predictions.append(2)
        else:
            raw_predictions.append(0)
            
    # Apply Temporal Smoothing (Median Filter) to defeat Jitter
    smoothed_predictions = medfilt(raw_predictions, kernel_size=21).astype(int)
    return smoothed_predictions

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python interferer_profile.py <input_wav_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    print(f"Processing {input_file}...")
    final_preds = process_file(input_file)
    
    print("Generating output files...")
    generate_outputs(final_preds)
    
    print("Done! Check predictions.csv and metrics.json")