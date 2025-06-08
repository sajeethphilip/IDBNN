import csv
import numpy as np
from collections import defaultdict

def normalize_power(power_array):
    """Normalize power spectrum using L2 norm"""
    norm = np.linalg.norm(power_array)
    if norm == 0:
        return power_array
    return power_array / norm

# Read and parse powerspec4.csv
sources = []
with open('powspec4.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Clean and parse the power feature string
        power_str = row['Power'].strip()
        if power_str.startswith('[') and power_str.endswith(']'):
            power_str = power_str[1:-1]
        power_str = power_str.replace('\n', ' ').replace('  ', ' ')
        parts = [x.strip() for x in power_str.split() if x.strip()]

        try:
            # Remove DC component (first element) and convert to floats
            power_vals = [float(x) for x in parts[1:]]  # Skip first element (DC)
        except (ValueError, IndexError):
            continue

        if len(power_vals) != 94:  # Should have 94 features after DC removal
            continue

        # Normalize the power spectrum
        normalized_power = normalize_power(np.array(power_vals))

        sources.append({
            'power': normalized_power,
            'label': int(row['label']),
            's_name': row['s_name']
        })

# Read and parse predictions.csv
predictions = []
with open('predictions.csv', 'r') as f:
    reader = csv.DictReader(f)
    power_columns = [f'Power_{i}' for i in range(1, 95)]  # Power_1 to Power_94

    for row in reader:
        try:
            # Get power features and normalize
            power_vals = [float(row[col]) for col in power_columns]
            normalized_power = normalize_power(np.array(power_vals))

            label_val = int(row['label'])
            pred_class = int(row['predicted_class'])
            confidence_val = float(row['confidence'])
        except (ValueError, KeyError):
            continue

        predictions.append({
            'power': normalized_power,
            'label': label_val,
            'predicted_class': pred_class,
            'confidence': confidence_val
        })

# Group predictions by label for faster matching
grouped_predictions = defaultdict(list)
for pred in predictions:
    grouped_predictions[pred['label']].append(pred)

# Tolerance for floating point comparison
TOLERANCE = 1e-5

# Prepare output data structure
source_predictions = defaultdict(list)
header = ['s_name', 'true_label'] + [f'Power_{i}' for i in range(1, 95)] + ['confidence', 'Winner']

# For confusion matrix
true_labels = []
predicted_labels = []

# Match sources to predictions
for source in sources:
    source_label = source['label']
    source_power = source['power']

    # Skip if no predictions with matching label
    if source_label not in grouped_predictions:
        continue

    # Find all matching predictions
    for pred in grouped_predictions[source_label]:
        # Compare normalized power spectra
        if np.allclose(source_power, pred['power'], atol=TOLERANCE):
            source_predictions[source['s_name']].append({
                'predicted_class': pred['predicted_class'],
                'confidence': pred['confidence']
            })

# Prepare final output rows
output_rows = []
for source in sources:
    s_name = source['s_name']

    # Skip if no predictions for this source
    if s_name not in source_predictions or not source_predictions[s_name]:
        continue

    preds = source_predictions[s_name]

    # Count votes for each class
    vote_count = defaultdict(int)
    for pred in preds:
        vote_count[pred['predicted_class']] += 1

    # Determine winner (majority vote)
    winner = max(vote_count.items(), key=lambda x: x[1])[0]

    # Calculate average confidence for winner class
    winner_confidences = [p['confidence'] for p in preds if p['predicted_class'] == winner]
    avg_confidence = sum(winner_confidences) / len(winner_confidences) if winner_confidences else 0.0

    # Store labels for confusion matrix
    true_labels.append(source['label'])
    predicted_labels.append(winner)

    # Prepare output row: source name, true label, power features, confidence, winner
    output_row = [s_name, source['label']] + source['power'].tolist() + [avg_confidence, winner]
    output_rows.append(output_row)

# Write results to file
with open('Source_wise_predictions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(output_rows)

print(f"Successfully saved {len(output_rows)} matched sources to Source_wise_predictions.csv")

# Generate confusion matrix and metrics
if true_labels and predicted_labels:
    # Initialize confusion matrix
    cm = np.zeros((2, 2), dtype=int)

    # Populate confusion matrix
    for true, pred in zip(true_labels, predicted_labels):
        cm[true][pred] += 1

    # Calculate metrics
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Write confusion matrix to file
    with open('confusion_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['', 'Predicted 0', 'Predicted 1'])
        writer.writerow(['Actual 0', tn, fp])
        writer.writerow(['Actual 1', fn, tp])

    # Write metrics to file
    with open('evaluation_metrics.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    print("Confusion Matrix:")
    print(f"          Predicted 0  Predicted 1")
    print(f"Actual 0   {tn:11}   {fp:11}")
    print(f"Actual 1   {fn:11}   {tp:11}")
    print(f"\nEvaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
else:
    print("No matched sources found for evaluation metrics")
