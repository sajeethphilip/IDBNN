import csv
import matplotlib.pyplot as plt
import numpy as np

# User inputs
input_filename = input("Enter the input filename: ")
base_name = input_filename.rsplit('.', 1)[0]
output_filename = f"{base_name}.csv"

normalize = input("Perform feature normalization (yes/no)? ").strip().lower() == 'yes'
exclude_power0 = input("Exclude Power_0 (yes/no)? ").strip().lower() == 'yes'
generate_graphs = input("Generate coadded graphs (yes/no)? ").strip().lower() == 'yes'
display_plots = False
if generate_graphs:
    display_plots = input("Display plots interactively (yes/no)? ").strip().lower() == 'yes'

start_index = 1 if exclude_power0 else 0
label_data = {}  # To store raw data for coadding

with open(input_filename, 'r', newline='') as infile, open(output_filename, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    next(reader)  # Skip header

    writer = csv.writer(outfile)
    num_columns = None

    for row in reader:
        if len(row) != 2:
            print(f"Skipping invalid row: {row}")
            continue

        # Original data parsing
        array_str = row[0].strip('[]"')
        elements_str = array_str.split()
        label = row[1].strip()

        # Process for CSV
        csv_elements = elements_str.copy()
        if exclude_power0:
            csv_elements = csv_elements[1:] if len(csv_elements) > 0 else []

        if normalize:
            elements = list(map(float, csv_elements))
            if len(elements) > 0:
                min_val, max_val = min(elements), max(elements)
                if max_val != min_val:
                    elements = [(x - min_val)/(max_val - min_val) for x in elements]
                else:
                    elements = [0.0]*len(elements)
                csv_elements = [f"{x:.8f}" for x in elements]

        # Process for coadded data (raw values, no normalization)
        if generate_graphs:
            coadd_elements = list(map(float, elements_str))
            if exclude_power0 and len(coadd_elements) > 0:
                coadd_elements = coadd_elements[1:]

            if label not in label_data:
                label_data[label] = []
            label_data[label].append(coadd_elements)

        # CSV writing
        if num_columns is None:
            num_columns = len(csv_elements)
            headers = [f'Power_{i+start_index}' for i in range(num_columns)] + ['label']
            writer.writerow(headers)
        elif len(csv_elements) != num_columns:
            raise ValueError(f"Row {reader.line_num} has {len(csv_elements)} elements, expected {num_columns}")

        writer.writerow(csv_elements + [label])

# Generate coadded spectra
if generate_graphs and label_data:
    for label, spectra in label_data.items():
        if not spectra:
            continue

        # Calculate statistics
        arr = np.array(spectra)
        mean_power = np.mean(arr, axis=0)
        std_power = np.std(arr, axis=0)
        n_spectra = arr.shape[0]

        # Create plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(mean_power)) + start_index
        plt.plot(x, mean_power, 'b-', label='Mean Power')
        plt.fill_between(x,
                        mean_power - std_power,
                        mean_power + std_power,
                        color='blue', alpha=0.2, label='±1 SD')

        plt.title(f"Coadded Spectrum ({label})\n{n_spectra} spectra")
        plt.xlabel("Frequency Index")
        plt.ylabel("Power")
        plt.legend()
        plt.grid(True)

        # Save and optionally display
        safe_label = ''.join(c if c.isalnum() else '_' for c in label)
        plt.savefig(f"{base_name}_{safe_label}_coadd.png", dpi=150, bbox_inches='tight')
        if display_plots:
            plt.show()
        else:
            plt.close()
