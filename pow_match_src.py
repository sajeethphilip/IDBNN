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

    for row_idx, row in enumerate(reader):
        try:
            # Get power features and normalize
            power_vals = [float(row[col]) for col in power_columns]
            original_power = power_vals  # Store original values for output
            normalized_power = normalize_power(np.array(power_vals))

            label_val = int(row['label'])
            pred_class = int(row['predicted_class'])
            confidence_val = float(row['confidence'])
        except (ValueError, KeyError):
            continue

        predictions.append({
            'original_power': original_power,
            'normalized_power': normalized_power,
            'label': label_val,
            'predicted_class': pred_class,
            'confidence': confidence_val,
            'row_id': row_idx  # Track row for reference
        })

# Group predictions by label for faster matching
grouped_predictions = defaultdict(list)
for pred in predictions:
    grouped_predictions[pred['label']].append(pred)

# Tolerance for floating point comparison
TOLERANCE = 1e-5

# Prepare data structures
source_matches = defaultdict(list)  # {s_name: [list of matched prediction rows]}
source_winners = {}  # {s_name: winner_class}
source_true_labels = {}  # {s_name: true_label}
source_winner_confidences = {}  # {s_name: winner_confidence (fraction)}

# Match sources to predictions
for source in sources:
    source_label = source['label']
    source_power = source['power']
    s_name = source['s_name']

    # Store true label for this source
    source_true_labels[s_name] = source_label

    # Skip if no predictions with matching label
    if source_label not in grouped_predictions:
        continue

    # Find all matching predictions
    for pred in grouped_predictions[source_label]:
        # Compare normalized power spectra
        if np.allclose(source_power, pred['normalized_power'], atol=TOLERANCE):
            source_matches[s_name].append(pred)

# Determine winner and winner confidence for each source
for s_name, matches in source_matches.items():
    if not matches:
        continue

    # Count votes for each class
    vote_count = defaultdict(int)
    for pred in matches:
        vote_count[pred['predicted_class']] += 1

    # Determine winner (majority vote)
    winner, winner_votes = max(vote_count.items(), key=lambda x: x[1])
    total_votes = len(matches)
    winner_confidence = winner_votes / total_votes  # Fraction of votes for winner

    source_winners[s_name] = winner
    source_winner_confidences[s_name] = winner_confidence

# Prepare detailed output rows
detailed_header = ['s_name', 'true_label', 'row_id'] + \
                  [f'Power_{i}' for i in range(1, 95)] + \
                  ['confidence', 'predicted_class', 'Winner', 'winner_confidence']

detailed_rows = []
for s_name, matches in source_matches.items():
    true_label = source_true_labels[s_name]
    winner = source_winners[s_name]
    winner_confidence = source_winner_confidences[s_name]

    for pred in matches:
        row = [
            s_name,
            true_label,
            pred['row_id'],
            *pred['original_power'],  # Original power values from predictions.csv
            pred['confidence'],
            pred['predicted_class'],
            winner,
            winner_confidence
        ]
        detailed_rows.append(row)

# Write detailed results to file
with open('Source_wise_predictions_detailed.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(detailed_header)
    writer.writerows(detailed_rows)

# Prepare summary output (one row per source)
summary_header = ['s_name', 'true_label', 'Winner', 'total_votes',
                  'votes_0', 'votes_1', 'winner_confidence']
summary_rows = []
for s_name in source_matches.keys():
    matches = source_matches[s_name]
    true_label = source_true_labels[s_name]
    winner = source_winners[s_name]
    winner_confidence = source_winner_confidences[s_name]

    # Count votes
    vote_count = defaultdict(int)
    for pred in matches:
        vote_count[pred['predicted_class']] += 1

    summary_rows.append([
        s_name,
        true_label,
        winner,
        len(matches),
        vote_count.get(0, 0),
        vote_count.get(1, 0),
        winner_confidence
    ])

# Write summary results to file
with open('Source_wise_predictions_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(summary_header)
    writer.writerows(summary_rows)

# Generate confusion matrix and metrics
true_labels = []
predicted_winners = []
for s_name in source_matches.keys():
    true_labels.append(source_true_labels[s_name])
    predicted_winners.append(source_winners[s_name])

if true_labels and predicted_winners:
    # Initialize confusion matrix
    cm = np.zeros((2, 2), dtype=int)

    # Populate confusion matrix
    for true, pred in zip(true_labels, predicted_winners):
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

    print(f"\nDetailed output saved to Source_wise_predictions_detailed.csv ({len(detailed_rows)} rows)")
    print(f"Summary output saved to Source_wise_predictions_summary.csv ({len(summary_rows)} rows)")
else:
    print("No matched sources found for evaluation metrics")

print("Processing complete!")
