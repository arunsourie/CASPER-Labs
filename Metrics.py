import csv
import json

def generate_outputs(predictions, gt_data=None, calc_powers=None, csv_filename='predictions.csv', json_filename='metrics.json'):
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        if gt_data is not None and calc_powers is not None:
            writer.writerow([
                'Frame_Index', 'Predicted_Class', 'Ground_Truth', 'Is_Correct', 'Running_Accuracy',
                'P_Target_Calc', 'P_Target_GT', 
                'P_Intf1_Calc', 'P_Intf1_GT', 
                'P_Intf2_Calc', 'P_Intf2_GT'
            ])
            
            correct_count = 0
            
            for idx in range(len(predictions)):
                pred = predictions[idx]
                truth = gt_data['labels'][idx]
                is_correct = 1 if pred == truth else 0
                correct_count += is_correct
                running_acc = (correct_count / (idx + 1)) * 100
                
                writer.writerow([
                    idx, pred, truth, is_correct, f"{running_acc:.2f}%",
                    f"{calc_powers['p_t'][idx]:.6e}", f"{gt_data['p_t'][idx]:.6e}",
                    f"{calc_powers['p_a'][idx]:.6e}", f"{gt_data['p_a'][idx]:.6e}",
                    f"{calc_powers['p_b'][idx]:.6e}", f"{gt_data['p_b'][idx]:.6e}"
                ])
        else:
            writer.writerow(['Frame_Index', 'Predicted_Class'])
            for idx, pred in enumerate(predictions):
                writer.writerow([idx, pred])
            
    metrics = {
        "target_mitigation_strategy": "Minimum Variance Distortionless Response (MVDR) formulation inherently constructs a spatial covariance matrix. This explicitly suppresses the highest-variance uncorrelated noise source (the continuous broadside target) to mathematically mitigate its presence prior to power evaluation.",
        "processing_type": "Non-Causal",
        "smoothing_strategy": "A median filter (kernel size = 21, approx 105ms) was applied globally over the output array to effectively eliminate jitter while preserving abrupt source transitions."
    }
    
    with open(json_filename, 'w') as f:
        json.dump(metrics, f, indent=4)