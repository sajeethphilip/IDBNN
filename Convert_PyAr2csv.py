import csv

input_filename = input("Enter the input filename: ")
base_name = input_filename.rsplit('.', 1)[0]
output_filename = f"{base_name}.csv"

normalize_input = input("Perform feature normalization (yes/no)? ").strip().lower()
normalize = normalize_input == 'yes'

with open(input_filename, 'r', newline='') as infile, open(output_filename, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    next(reader)  # Skip the original header row "Power,label"

    writer = csv.writer(outfile)
    num_columns = None

    for row in reader:
        if len(row) != 2:
            print(f"Skipping invalid row: {row}")
            continue

        array_str = row[0].strip('[]"')
        elements_str = array_str.split()
        label = row[1].strip()

        if normalize:
            # Convert elements to floats and normalize
            elements = list(map(float, elements_str))
            min_val = min(elements)
            max_val = max(elements)

            if max_val != min_val:
                elements = [(x - min_val) / (max_val - min_val) for x in elements]
            else:
                # Handle case where all elements are the same
                elements = [0.0 for _ in elements]

            # Format to 8 decimal places as strings
            elements = [f"{x:.8f}" for x in elements]
        else:
            # Use original string elements without normalization
            elements = elements_str

        # Determine number of columns and write headers
        if num_columns is None:
            num_columns = len(elements)
            headers = [f'Power_{i}' for i in range(num_columns)] + ['label']
            writer.writerow(headers)
        else:
            if len(elements) != num_columns:
                raise ValueError(f"Row {reader.line_num} has {len(elements)} elements, expected {num_columns}")

        # Write the processed row
        writer.writerow(elements + [label])
