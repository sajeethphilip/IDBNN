import os
import shutil
import tarfile
from torchvision import datasets
import torchvision
from typing import List, Dict, Tuple
import argparse

def list_available_datasets() -> List[str]:
    """List all available datasets in torchvision.datasets"""
    dataset_classes = []
    for name in dir(torchvision.datasets):
        if name[0].isupper() and name not in ['VisionDataset', 'DatasetFolder', 'ImageFolder']:
            dataset_classes.append(name)
    return sorted(dataset_classes)

def get_dataset_info(dataset_name: str) -> Dict:
    """Get information about a specific dataset"""
    try:
        dataset_class = getattr(torchvision.datasets, dataset_name)
        return {
            'name': dataset_name,
            'has_train_test_split': hasattr(dataset_class, 'train') and hasattr(dataset_class, 'test'),
            'default_root': os.path.join('data', dataset_name.lower())
        }
    except AttributeError:
        raise ValueError(f"Dataset {dataset_name} not found in torchvision.datasets")

def ensure_directory_exists(path: str) -> None:
    """Ensure a directory exists, create if it doesn't"""
    os.makedirs(path, exist_ok=True)

def extract_tar_file(tar_path: str, extract_to: str) -> None:
    """Extract a tar file to the specified directory"""
    ensure_directory_exists(extract_to)
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_to)

def download_dataset(dataset_name: str, root: str, merge_train_test: bool = True, **kwargs) -> Tuple[str, List[str]]:
    """
    Download a torchvision dataset and organize it according to the specified structure.
    """
    dataset_class = getattr(torchvision.datasets, dataset_name)
    dataset_info = get_dataset_info(dataset_name)

    # Create the target directory structure
    final_path = os.path.join(root, dataset_name.lower(), 'train')
    ensure_directory_exists(final_path)

    try:
        # Download the dataset (this may download a tar file)
        dataset = dataset_class(root=root, download=True, **kwargs)

        # Special handling for Caltech256 which comes as a tar file
        if dataset_name.lower() == 'caltech256':
            tar_path = os.path.join(root, '256_ObjectCategories.tar')
            if os.path.exists(tar_path):
                extract_tar_file(tar_path, os.path.join(root, dataset_name.lower()))
                # Move the extracted content to our train directory
                extracted_dir = os.path.join(root, dataset_name.lower(), '256_ObjectCategories')
                if os.path.exists(extracted_dir):
                    for item in os.listdir(extracted_dir):
                        shutil.move(os.path.join(extracted_dir, item), final_path)
                    shutil.rmtree(extracted_dir)
                os.remove(tar_path)
            return final_path, _infer_class_names(final_path)

        if dataset_info['has_train_test_split']:
            # Handle datasets with train/test splits
            train_path = os.path.join(root, dataset_name.lower(), 'train')
            test_path = os.path.join(root, dataset_name.lower(), 'test')

            if merge_train_test:
                _copy_contents(train_path, final_path)
                _copy_contents(test_path, final_path)
                shutil.rmtree(train_path, ignore_errors=True)
                shutil.rmtree(test_path, ignore_errors=True)
            else:
                final_path = train_path

            class_names = dataset_class(root=root, train=True, download=False).classes

        else:
            # Handle datasets without splits
            dataset_path = os.path.join(root, dataset_name.lower())

            if os.path.exists(os.path.join(dataset_path, 'train')):
                _copy_contents(os.path.join(dataset_path, 'train'), final_path)
                shutil.rmtree(os.path.join(dataset_path, 'train'), ignore_errors=True)
            elif os.path.exists(os.path.join(dataset_path, 'test')):
                _copy_contents(os.path.join(dataset_path, 'test'), final_path)
                shutil.rmtree(os.path.join(dataset_path, 'test'), ignore_errors=True)
            else:
                # Move all class directories to train folder
                for item in os.listdir(dataset_path):
                    item_path = os.path.join(dataset_path, item)
                    if os.path.isdir(item_path) and not item.endswith('.tar'):
                        dest_path = os.path.join(final_path, item)
                        shutil.move(item_path, dest_path)

            class_names = getattr(dataset, 'classes', None) or _infer_class_names(final_path)

        return final_path, class_names

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        raise

def _copy_contents(src: str, dst: str) -> None:
    """Copy contents from src to dst, maintaining directory structure"""
    ensure_directory_exists(dst)
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dst_path = os.path.join(dst, item)

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)

def _infer_class_names(directory: str) -> List[str]:
    """Infer class names from directory structure"""
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def interactive_mode():
    """Interactive mode for dataset selection and downloading"""
    available_datasets = list_available_datasets()

    print("Available datasets:")
    for i, dataset in enumerate(available_datasets, 1):
        print(f"{i}. {dataset}")

    dataset_name = input("\nEnter dataset name (press Enter to process all datasets): ").strip()

    if not dataset_name:
        # Process all datasets
        for dataset in available_datasets:
            process_dataset(dataset)
    else:
        # Process single dataset
        process_dataset(dataset_name)

def process_dataset(dataset_name: str) -> None:
    """Process a single dataset"""
    if dataset_name not in list_available_datasets():
        print(f"Dataset '{dataset_name}' not found.")
        return

    print(f"\nProcessing dataset: {dataset_name}")
    try:
        dataset_info = get_dataset_info(dataset_name)
        merge = True  # Default to merging

        if dataset_info['has_train_test_split']:
            response = input(f"Dataset {dataset_name} has train/test split. Merge them? (y/n, default y): ").lower()
            merge = response != 'n'

        final_path, class_names = download_dataset(
            dataset_name=dataset_name,
            root='data',
            merge_train_test=merge
        )

        print(f"Successfully processed dataset. Files saved to: {final_path}")
        print(f"Class names: {class_names}")

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")

if __name__ == "__main__":
    if len(os.sys.argv) > 1:
        # Command line mode
        parser = argparse.ArgumentParser(description='Download and organize torchvision image datasets')
        parser.add_argument('--dataset', type=str, default='', help='Name of dataset to download')
        parser.add_argument('--root', type=str, default='data', help='Root directory for downloaded datasets')
        parser.add_argument('--merge', action='store_true', help='Merge train and test sets (if available)')
        parser.add_argument('--all', action='store_true', help='Download all available datasets')
        args = parser.parse_args()

        if args.dataset or args.all:
            available_datasets = list_available_datasets()

            if args.all:
                datasets_to_process = available_datasets
            else:
                if args.dataset not in available_datasets:
                    print(f"Dataset '{args.dataset}' not found.")
                    exit()
                datasets_to_process = [args.dataset]

            for dataset_name in datasets_to_process:
                process_dataset(dataset_name)
        else:
            interactive_mode()
    else:
        # Interactive mode
        interactive_mode()
