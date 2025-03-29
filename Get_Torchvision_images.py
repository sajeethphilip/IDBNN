import os
import shutil
from torchvision import datasets
import torchvision
from typing import Optional, List, Dict, Tuple
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
            'default_root': os.path.join('Data', dataset_name.lower())
        }
    except AttributeError:
        raise ValueError(f"Dataset {dataset_name} not found in torchvision.datasets")

def download_dataset(dataset_name: str, root: str, merge_train_test: bool = True, **kwargs) -> Tuple[str, List[str]]:
    """
    Download a torchvision dataset and organize it according to the specified structure.

    Args:
        dataset_name: Name of the dataset to download
        root: Root directory where the dataset will be stored
        merge_train_test: Whether to merge train and test sets
        **kwargs: Additional arguments to pass to the dataset constructor

    Returns:
        Tuple of (final_path, class_names)
    """
    dataset_class = getattr(torchvision.datasets, dataset_name)
    dataset_info = get_dataset_info(dataset_name)

    # Create the target directory structure
    final_path = os.path.join(root, dataset_name.lower(), 'train')
    os.makedirs(final_path, exist_ok=True)

    if dataset_info['has_train_test_split']:
        # Download both train and test sets
        train_set = dataset_class(root=root, train=True, download=True, **kwargs)
        test_set = dataset_class(root=root, train=False, download=True, **kwargs)

        # Get the original paths
        train_path = os.path.join(root, dataset_name.lower(), 'train')
        test_path = os.path.join(root, dataset_name.lower(), 'test')

        if merge_train_test:
            # Copy all files from train and test to our final train directory
            _copy_contents(train_path, final_path)
            _copy_contents(test_path, final_path)
        else:
            # Keep separate train and test directories
            final_path = train_path  # Just use the original structure

        # Clean up if we merged
        if merge_train_test:
            shutil.rmtree(train_path, ignore_errors=True)
            shutil.rmtree(test_path, ignore_errors=True)

        class_names = train_set.classes

    else:
        # Dataset doesn't have train/test split - treat all as training data
        dataset = dataset_class(root=root, download=True, **kwargs)

        # Check if the dataset has a standard structure
        dataset_path = os.path.join(root, dataset_name.lower())

        if os.path.exists(os.path.join(dataset_path, 'train')):
            # Has train folder (but no test folder)
            _copy_contents(os.path.join(dataset_path, 'train'), final_path)
            shutil.rmtree(os.path.join(dataset_path, 'train'), ignore_errors=True)
        elif os.path.exists(os.path.join(dataset_path, 'test')):
            # Has test folder (but no train folder)
            _copy_contents(os.path.join(dataset_path, 'test'), final_path)
            shutil.rmtree(os.path.join(dataset_path, 'test'), ignore_errors=True)
        else:
            # Has raw class folders - copy them to our train directory
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    dest_path = os.path.join(final_path, item)
                    shutil.move(item_path, dest_path)

        class_names = getattr(dataset, 'classes', None) or _infer_class_names(final_path)

    return final_path, class_names

def _copy_contents(src: str, dst: str) -> None:
    """Copy contents from src to dst, maintaining directory structure"""
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dst_path = os.path.join(dst, item)

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
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
            print(f"\nProcessing dataset: {dataset}")
            try:
                dataset_info = get_dataset_info(dataset)
                merge = True  # Default to merging

                if dataset_info['has_train_test_split']:
                    response = input(f"Dataset {dataset} has train/test split. Merge them? (y/n, default y): ").lower()
                    merge = response != 'n'

                final_path, class_names = download_dataset(
                    dataset_name=dataset,
                    root='Data',
                    merge_train_test=merge
                )

                print(f"Successfully processed dataset. Files saved to: {final_path}")
                print(f"Class names: {class_names}")

            except Exception as e:
                print(f"Error processing dataset {dataset}: {str(e)}")
    else:
        # Process single dataset
        if dataset_name not in available_datasets:
            print(f"Dataset '{dataset_name}' not found. Please choose from the available datasets.")
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
                root='Data',
                merge_train_test=merge
            )

            print(f"Successfully processed dataset. Files saved to: {final_path}")
            print(f"Class names: {class_names}")

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")

if __name__ == "__main__":
    # Check if any command line arguments were provided
    if len(os.sys.argv) > 1:
        # Command line mode
        parser = argparse.ArgumentParser(description='Download and organize torchvision image datasets')
        parser.add_argument('--dataset', type=str, default='', help='Name of dataset to download')
        parser.add_argument('--root', type=str, default='Data', help='Root directory for downloaded datasets')
        parser.add_argument('--merge', action='store_true', help='Merge train and test sets (if available)')
        parser.add_argument('--all', action='store_true', help='Download all available datasets')
        args = parser.parse_args()

        if args.dataset or args.all:
            available_datasets = list_available_datasets()

            if args.all:
                datasets_to_process = available_datasets
            else:
                if args.dataset not in available_datasets:
                    print(f"Dataset '{args.dataset}' not found. Available datasets:")
                    for i, dataset in enumerate(available_datasets, 1):
                        print(f"{i}. {dataset}")
                    exit()
                datasets_to_process = [args.dataset]

            for dataset_name in datasets_to_process:
                print(f"\nProcessing dataset: {dataset_name}")
                try:
                    final_path, class_names = download_dataset(
                        dataset_name=dataset_name,
                        root=args.root,
                        merge_train_test=args.merge
                    )

                    print(f"Successfully processed dataset. Files saved to: {final_path}")
                    print(f"Class names: {class_names}")

                except Exception as e:
                    print(f"Error processing dataset {dataset_name}: {str(e)}")
        else:
            interactive_mode()
    else:
        # Interactive mode
        interactive_mode()
