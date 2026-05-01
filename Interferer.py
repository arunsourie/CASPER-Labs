import sys
import os
import numpy as np
import soundfile as sf
from scipy.signal import medfilt
from Beamformer import get_steering_vector, calculate_mvdr_power
from Metrics import generate_outputs

SAMPLE_RATE = 16000
SOUND_SPEED = 343.0
MIC_SPACING = 0.08
FRAME_SIZE = 80      
MARGIN_DB = 3.0      

def process_file(filepath):
    audio, sr = sf.read(filepath)
    if sr != SAMPLE_RATE:
        raise ValueError()
    
    num_frames = len(audio) // FRAME_SIZE
    
    delay_t = 0.0
    delay_a = (MIC_SPACING * np.sin(np.radians(60))) / SOUND_SPEED
    delay_b = (MIC_SPACING * np.sin(np.radians(-60))) / SOUND_SPEED
    
    freqs = np.fft.rfftfreq(FRAME_SIZE, d=1.0/SAMPLE_RATE)
    freqs[0] = 1e-6 
    
    steer_t = get_steering_vector(delay_t, freqs)
    steer_a = get_steering_vector(delay_a, freqs)
    steer_b = get_steering_vector(delay_b, freqs)
    
    raw_predictions = []
    p_t_calc, p_a_calc, p_b_calc = [], [], []
    
    for i in range(num_frames):
        start = i * FRAME_SIZE
        end = start + FRAME_SIZE
        frame = audio[start:end]
        
        p_t = calculate_mvdr_power(frame, steer_t)
        p_a = calculate_mvdr_power(frame, steer_a)
        p_b = calculate_mvdr_power(frame, steer_b)
        
        p_t_calc.append(p_t)
        p_a_calc.append(p_a)
        p_b_calc.append(p_b)
        
        if p_a < 1e-10 and p_b < 1e-10:
            raw_predictions.append(0)
            continue
            
        p_a = max(p_a, 1e-10)
        p_b = max(p_b, 1e-10)
        
        diff_db = 10 * np.log10(p_a / p_b)
        
        if diff_db > MARGIN_DB:
            raw_predictions.append(1)
        elif diff_db < -MARGIN_DB:
            raw_predictions.append(2)
        else:
            raw_predictions.append(0)
            
    smoothed_predictions = medfilt(raw_predictions, kernel_size=21).astype(int)
    
    calc_powers = {
        'p_t': np.array(p_t_calc),
        'p_a': np.array(p_a_calc),
        'p_b': np.array(p_b_calc)
    }
    
    return smoothed_predictions, calc_powers

if __name__ == "__main__":
    if len(sys.argv) == 2:
        input_file = sys.argv[1]
    else:
        input_file = "Simulated_Environment.wav"
        
    if not os.path.exists(input_file):
        sys.exit(1)
        
    final_preds, calc_powers = process_file(input_file)
    
    gt_data = None
    if os.path.exists('ground_truth.npz'):
        gt_data = np.load('ground_truth.npz')
    
    generate_outputs(final_preds, gt_data=gt_data, calc_powers=calc_powers)