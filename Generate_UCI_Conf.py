import requests
from bs4 import BeautifulSoup
import json
import os
import io
import sys
from ucimlrepo import fetch_ucirepo, list_available_datasets
import pandas as pd

def get_available_datasets():
    # Redirect stdout to capture the output
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Call the function that prints the datasets
    list_available_datasets()

    # Get the printed output and restore stdout
    output = new_stdout.getvalue()
    sys.stdout = old_stdout

    # Process the output to extract dataset names
    # Split by newlines and filter out empty lines
    lines = [line.strip() for line in output.split('\n') if line.strip()]

    # Extract dataset names (everything before the trailing spaces and numbers)
    datasets = []
    for line in lines:
        if line != "Available datasets:":  # Skip the header line
            # Split on whitespace and take everything except the last item (which is the number)
            name = ' '.join(line.split()[:-1]).strip()
            if name:  # Only add non-empty names
                datasets.append(name)

    return datasets

def fetch_dataset_info(dataset_name):
    try:
        dataset = fetch_ucirepo(name=dataset_name)

        # Determine the actual separator by checking the file extension and content
        file_url = dataset.metadata.data_url
        response = requests.get(file_url)
        if response.ok:
            first_line = response.text.split('\n')[0]
            if ',' in first_line:
                separator = ','
            elif ';' in first_line:
                separator = ';'
            elif '\t' in first_line:
                separator = '\t'
            else:
                separator = ' '
        else:
            separator = ','  # default if unable to determine

        # Handle target column - ensure it's a single value
        if isinstance(dataset.metadata.target_col, list):
            target = dataset.metadata.target_col[0]  # take first if multiple
        else:
            target = dataset.metadata.target_col

        dataset_info = {
            "name": dataset.metadata.name.lower(),
            "url": dataset.metadata.data_url,
            "columns": list(dataset.data.original.columns),
            "target": target,
            "instances": dataset.metadata.num_instances,
            "separator": separator,
            "has_header": True if dataset.metadata.feature_types else False
        }

        dataset_info["accuracy"] = 0.0  # placeholder for accuracy

        return dataset_info
    except Exception as e:
        print(f"Error fetching dataset {dataset_name}: {str(e)}")
        return None

def create_config_file(dataset_name, output_dir="data"):
    dataset_info = fetch_dataset_info(dataset_name)

    if dataset_info is None:
        print(f"Could not fetch information for dataset '{dataset_name}'")
        return

    config = {
        "file_path":dataset_name.lower().replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '_'),
        "file_url": dataset_info["url"],
        "column_names": dataset_info["columns"],
        "target_column": dataset_info["target"],  # Now a single value
        "separator": dataset_info["separator"],   # Determined from actual file
        "has_header": dataset_info["has_header"],
         "likelihood_config": {
        "feature_group_size": 2,
        "max_combinations": 1000,
        "bin_sizes": [20]
        },
        "training_params": {
            "trials": 100,
            "cardinality_threshold": 0.9,
            "learning_rate": 0.001,
            "random_seed": 42,
            "epochs": 1000,
            "test_fraction": 0.2,
            "enable_adaptive": True
        }
    }

    final_path=f"{output_dir}/{dataset_info['name'].lower().replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '_')}"
    os.makedirs(final_path, exist_ok=True)
    filename = f"{dataset_info['name'].lower().replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '_')}.conf"
    with open(os.path.join(final_path, filename), 'w') as f:
        #f.write(f"# {filename} best_accuracy: {dataset_info['accuracy']}, instances: {dataset_info['instances']}\n")
        json.dump(config, f, indent=4)
        print(f"Created configuration file: {final_path}/{filename}")
    return final_path


def download_uci_dataset(config_path,destination='data/'):
    # Read the configuration file
    print(f"Opening file {config_path}")
    with open(config_path, 'r') as f:
        # Skip the first line with accuracy info
        #next(f)
        config = json.load(f)

    # Extract file info from config
    file_url = config['file_url']
    dataset_name = os.path.basename(config_path).replace('.conf', '.csv')
    separator = config['separator']
    has_header = config['has_header']

    try:
        # Download the data
        response = requests.get(file_url)
        response.raise_for_status()

        # Read the data into a pandas DataFrame
        if has_header:
            df = pd.read_csv(io.StringIO(response.text), sep=separator)
        else:
            df = pd.read_csv(io.StringIO(response.text), sep=separator,
                           names=config['column_names'])

        # Save to local CSV file
        pth_name = os.path.basename(config_path).replace('.conf', '')
        dest_path= os.path.join(destination, pth_name,dataset_name)
        df.to_csv(dest_path, index=False)
        print(f"Successfully downloaded dataset to {dest_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {str(e)}")
    except pd.errors.EmptyDataError:
        print("The downloaded file is empty")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import os

    def ensure_directory_exists(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created.")
        else:
            print(f"Exisiting Directory '{directory_path}' will be used to update configure files from UCI data.")

    # Example usage
    directory_path = 'data/'
    ensure_directory_exists(directory_path)

    # Get the list of available datasets
    available_datasets = get_available_datasets()
    print("Available datasets:")
    for i, dataset in enumerate(available_datasets, 1):
        print(f"{i}. {dataset}")

    dataset_name = input("\nEnter dataset name (press Enter to process all datasets): ")
    if dataset_name == '':
        # Process all datasets
        for dataset_name in available_datasets:
            print(f"\nProcessing dataset: {dataset_name}")
            directory_path=create_config_file(dataset_name)
    else:
        # Process single dataset
        directory_path=create_config_file(dataset_name)

    # List all configuration files
    config_files = [f for f in os.listdir(directory_path)
                   if f.endswith('.conf')]

    print("Available configuration files:")
    for i, conf in enumerate(config_files, 1):
        print(f"{i}. {conf}")

    choice = input("\nEnter config file name (press Enter for all): ")

    print(f"Your choice is {choice} for conf in {config_files}")
    if choice == '':
        # Process all configuration files
        for conf in config_files:
            print(f"Processsing {conf}")
            config_path = os.path.join(directory_path, conf)
            print(f"\nProcessing: {conf}")
            download_uci_dataset(config_path)
    else:
        # Process single configuration file
        config_path = os.path.join(directory_path, choice)
        if os.path.exists(config_path):
            download_uci_dataset(config_path)
        else:
            print("Configuration file not found")
