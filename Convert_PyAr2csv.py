import csv

input_filename = input("Enter the input filename: ")
base_name = input_filename.rsplit('.', 1)[0]
output_filename = f"{base_name}.csv"

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
        elements = array_str.split()
        label = row[1].strip()

        if num_columns is None:
            num_columns = len(elements)
            headers = [f'Power_{i}' for i in range(num_columns)] + ['label']
            writer.writerow(headers)
        else:
            if len(elements) != num_columns:
                raise ValueError(f"Row {reader.line_num} has {len(elements)} elements, expected {num_columns}")

        writer.writerow(elements + [label])
