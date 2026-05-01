import csv
import json

def generate_outputs(predictions, csv_filename='predictions.csv', json_filename='metrics.json'):
    """Saves the final classifications to a CSV and the methodology to a JSON."""
    
    # Write Predictions
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame_Index', 'Class_Label'])
        for idx, label in enumerate(predictions):
            writer.writerow([idx, label])
            
    # Write Technical Metrics
    metrics = {
        "target_mitigation_strategy": "Minimum Variance Distortionless Response (MVDR) formulation inherently constructs a spatial covariance matrix. This explicitly suppresses the highest-variance uncorrelated noise source (the continuous broadside target) to mathematically mitigate its presence prior to power evaluation.",
        "processing_type": "Non-Causal",
        "smoothing_strategy": "A median filter (kernel size = 21, approx 105ms) was applied globally over the output array to effectively eliminate jitter while preserving abrupt source transitions."
    }
    
    with open(json_filename, 'w') as f:
        json.dump(metrics, f, indent=4)