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

def move_all_to_train(dataset_root: str, train_path: str) -> List[str]:
    """
    Move all image class subfolders to the train directory.
    Cleans the target directory first if it exists.
    Handles cases where:
    - There are train/test folders (move their contents to train)
    - Only raw class folders exist (move them to train)
    - There's a containing folder like 256_ObjectCategories (move its contents to train)
    """
    # Clean the train directory if it exists
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    os.makedirs(train_path, exist_ok=True)

    class_names = []

    # First check for train/test folders
    for split in ['train', 'test']:
        split_path = os.path.join(dataset_root, split)
        if os.path.exists(split_path):
            # Move all class folders from this split to train
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    dest_path = os.path.join(train_path, class_name)

                    # If class already exists in train (from previous split), merge the contents
                    if os.path.exists(dest_path):
                        for item in os.listdir(class_path):
                            src_item = os.path.join(class_path, item)
                            if os.path.isdir(src_item):
                                # For nested directories, merge recursively
                                dest_item = os.path.join(dest_path, item)
                                if os.path.exists(dest_item):
                                    shutil.rmtree(dest_item)
                                shutil.move(src_item, dest_path)
                            else:
                                # For files, overwrite if they exist
                                dest_file = os.path.join(dest_path, item)
                                if os.path.exists(dest_file):
                                    os.remove(dest_file)
                                shutil.move(src_item, dest_path)
                    else:
                        shutil.move(class_path, dest_path)

                    if class_name not in class_names:
                        class_names.append(class_name)

            # Remove the now-empty split directory
            shutil.rmtree(split_path)

    # If no train/test folders found, look for direct class folders or containing folders
    if not class_names:
        for item in os.listdir(dataset_root):
            item_path = os.path.join(dataset_root, item)

            # Skip hidden files and non-directories
            if not os.path.isdir(item_path) or item.startswith('.'):
                continue

            # If this is a directory that contains class folders (like 256_ObjectCategories)
            if any(os.path.isdir(os.path.join(item_path, subitem)) for subitem in os.listdir(item_path)):
                # Move all its subdirectories to train
                for class_name in os.listdir(item_path):
                    class_path = os.path.join(item_path, class_name)
                    if os.path.isdir(class_path):
                        dest_path = os.path.join(train_path, class_name)
                        if os.path.exists(dest_path):
                            shutil.rmtree(dest_path)
                        shutil.move(class_path, dest_path)
                        class_names.append(class_name)
                # Remove the now-empty containing directory
                shutil.rmtree(item_path)
            else:
                # It's a class folder itself, move it to train
                dest_path = os.path.join(train_path, item)
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.move(item_path, dest_path)
                class_names.append(item)

    return sorted(class_names)

def download_dataset(dataset_name: str, root: str, merge_train_test: bool = True, **kwargs) -> Tuple[str, List[str]]:
    """
    Download and organize a torchvision dataset.
    Always creates data/<dataset>/train/<class_folders> structure.
    """
    dataset_class = getattr(torchvision.datasets, dataset_name)
    dataset_path = os.path.join(root, dataset_name.lower())
    train_path = os.path.join(dataset_path, 'train')

    ensure_directory_exists(train_path)

    try:
        # Download the dataset (this may download a tar file)
        dataset = dataset_class(root=root, download=True, **kwargs)

        # Special handling for Caltech256 which comes as a tar file
        if dataset_name.lower() == 'caltech256':
            tar_path = os.path.join(root, '256_ObjectCategories.tar')
            if os.path.exists(tar_path):
                extract_tar_file(tar_path, dataset_path)
                os.remove(tar_path)

        # Handle all datasets (standard and special cases)
        class_names = move_all_to_train(dataset_path, train_path)

        # Get class names from dataset if available
        if hasattr(dataset, 'classes') and dataset.classes:
            return train_path, dataset.classes
        return train_path, class_names

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        raise

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

        # For datasets with splits, ask about merging (though we'll merge everything to train)
        merge = True
        if dataset_info['has_train_test_split']:
            response = input(f"Dataset {dataset_name} has train/test split. Merge them to train? (y/n, default y): ").lower()
            merge = response != 'n'

        final_path, class_names = download_dataset(
            dataset_name=dataset_name,
            root='data',
            merge_train_test=merge
        )

        print(f"Successfully processed dataset. Files saved to: {final_path}")
        print(f"Found {len(class_names)} classes: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")

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
