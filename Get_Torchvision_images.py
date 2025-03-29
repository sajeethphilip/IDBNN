import os
import shutil
import tarfile
from torchvision import datasets
import torchvision
from typing import List, Dict, Tuple
import argparse
import json
from typing import Dict, Any
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Any

def get_image_properties(dataset_path: str) -> Tuple[int, Tuple[int, int], List[float], List[float]]:
    """
    Analyze images in the dataset to determine:
    - Number of channels (1 for grayscale, 3 for RGB)
    - Image size (width, height)
    - Mean pixel values per channel
    - Standard deviation per channel

    Args:
        dataset_path: Path to the dataset's train directory

    Returns:
        Tuple of (channels, (width, height), mean_values, std_values)
    """
    # Find first image in the dataset
    for root, _, files in os.walk(dataset_path):
        if files:
            first_image = os.path.join(root, files[0])
            break
    else:
        raise ValueError("No images found in dataset directory")

    # Open the image and convert to numpy array
    img = Image.open(first_image)
    img_array = np.array(img)

    # Determine image properties
    if len(img_array.shape) == 2:  # Grayscale
        channels = 1
        height, width = img_array.shape
    else:  # Color
        height, width, channels = img_array.shape

    # Calculate mean and std from a sample of images (max 100 for efficiency)
    means = []
    stds = []
    sample_count = 0
    max_samples = 100

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(root, file))
                img_array = np.array(img)

                if channels == 1:
                    if len(img_array.shape) == 3:  # Convert color to grayscale
                        img_array = np.mean(img_array, axis=2)
                    means.append(np.mean(img_array))
                    stds.append(np.std(img_array))
                else:
                    if len(img_array.shape) == 2:  # Convert grayscale to RGB
                        img_array = np.stack([img_array]*3, axis=-1)
                    means.append(np.mean(img_array, axis=(0,1)))
                    stds.append(np.std(img_array, axis=(0,1)))

                sample_count += 1
                if sample_count >= max_samples:
                    break
        if sample_count >= max_samples:
            break

    # Calculate overall mean and std
    if channels == 1:
        mean_val = [float(np.mean(means))]
        std_val = [float(np.mean(stds))]
    else:
        mean_val = [float(x) for x in np.mean(means, axis=0)]
        std_val = [float(x) for x in np.mean(stds, axis=0)]

    return channels, (width, height), mean_val, std_val
def create_config_file(dataset_name: str, dataset_path: str, class_names: List[str],
                      input_size: Tuple[int, int] = (32, 32), in_channels: int = 3) -> None:
    """
    Create a JSON configuration file for the dataset based on the template.

    Args:
        dataset_name: Name of the dataset
        dataset_path: Path where the dataset is stored
        class_names: List of class names in the dataset
        input_size: Default input size (width, height)
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
    """
    config_path = os.path.join(dataset_path, f"{dataset_name.lower()}.json")

    # Basic dataset information
    config: Dict[str, Any] = {
        "dataset": {
            "name": dataset_name.lower(),
            "type": "torchvision",
            "in_channels": in_channels,
            "num_classes": len(class_names),
            "input_size": list(input_size),
            "mean": [0.5] * in_channels,
            "std": [0.5] * in_channels,
            "resize_images": False,
            "train_dir": os.path.join(dataset_path, "train"),
            "test_dir": os.path.join(dataset_path, "test") if os.path.exists(os.path.join(dataset_path, "test")) else ""
        },
        # Rest of the configuration can be copied from template
        "model": {
            "encoder_type": "autoenc",
            "enable_adaptive": True,
            "feature_dims": 128,
            "learning_rate": 0.001,
            "optimizer": {
                "type": "Adam",
                "weight_decay": 0.0001,
                "momentum": 0.9,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-08
            },
           "scheduler": {
                "type": "ReduceLROnPlateau",
                "factor": 0.1,
                "patience": 10,
                "min_lr": 1e-06,
                "verbose": true
            },
            "autoencoder_config": {
                "reconstruction_weight": 1.0,
                "feature_weight": 0.1,
                "convergence_threshold": 0.001,
                "min_epochs": 10,
                "patience": 5,
                "enhancements": {
                    "enabled": true,
                    "use_kl_divergence": true,
                    "use_class_encoding": false,
                    "kl_divergence_weight": 0.5,
                    "classification_weight": 0.5,
                    "clustering_temperature": 1.0,
                    "min_cluster_confidence": 0.7
                }
            },
            "loss_functions": {
                "structural": {
                    "enabled": true,
                    "weight": 1.0,
                    "params": {
                        "edge_weight": 1.0,
                        "smoothness_weight": 0.5
                    }
                },
                "color_enhancement": {
                    "enabled": true,
                    "weight": 0.8,
                    "params": {
                        "channel_weight": 0.5,
                        "contrast_weight": 0.3
                    }
                },
                "morphology": {
                    "enabled": true,
                    "weight": 0.6,
                    "params": {
                        "shape_weight": 0.7,
                        "symmetry_weight": 0.3
                    }
                },
                "detail_preserving": {
                    "enabled": true,
                    "weight": 0.8,
                    "params": {
                        "detail_weight": 1.0,
                        "texture_weight": 0.8,
                        "frequency_weight": 0.6
                    }
                },
                "astronomical_structure": {
                    "enabled": true,
                    "weight": 1.0,
                    "components": {
                        "edge_preservation": true,
                        "peak_preservation": true,
                        "detail_preservation": true
                    }
                },
                "medical_structure": {
                    "enabled": true,
                    "weight": 1.0,
                    "components": {
                        "boundary_preservation": true,
                        "tissue_contrast": true,
                        "local_structure": true
                    }
                },
                "agricultural_pattern": {
                    "enabled": true,
                    "weight": 1.0,
                    "components": {
                        "texture_preservation": true,
                        "damage_pattern": true,
                        "color_consistency": true
                    }
                }
            },
            "enhancement_modules": {
                "astronomical": {
                    "enabled": true,
                    "components": {
                        "structure_preservation": true,
                        "detail_preservation": true,
                        "star_detection": true,
                        "galaxy_features": true,
                        "kl_divergence": true
                    },
                    "weights": {
                        "detail_weight": 1.0,
                        "structure_weight": 0.8,
                        "edge_weight": 0.7
                    }
                },
                "medical": {
                    "enabled": true,
                    "components": {
                        "tissue_boundary": true,
                        "lesion_detection": true,
                        "contrast_enhancement": true,
                        "subtle_feature_preservation": true
                    },
                    "weights": {
                        "boundary_weight": 1.0,
                        "lesion_weight": 0.8,
                        "contrast_weight": 0.6
                    }
                },
                "agricultural": {
                    "enabled": true,
                    "components": {
                        "texture_analysis": true,
                        "damage_detection": true,
                        "color_anomaly": true,
                        "pattern_enhancement": true,
                        "morphological_features": true
                    },
                    "weights": {
                        "texture_weight": 1.0,
                        "damage_weight": 0.8,
                        "pattern_weight": 0.7
                    }
                }
            }
        },
        "training": {
            "batch_size": 128,
            "epochs": 20,
            "num_workers": 4,
            "checkpoint_dir": "data/mnist/checkpoints",
            "validation_split": 0.2,
            "invert_DBNN": true,
            "reconstruction_weight": 0.5,
            "feedback_strength": 0.3,
            "inverse_learning_rate": 0.1,
            "early_stopping": {
                "patience": 5,
                "min_delta": 0.001
            }
        },
        "augmentation": {
            "enabled": true,
            "random_crop": {
                "enabled": true,
                "padding": 4
            },
            "random_rotation": {
                "enabled": true,
                "degrees": 10
            },
            "horizontal_flip": {
                "enabled": true,
                "probability": 0.5
            },
            "vertical_flip": {
                "enabled": false
            },
            "color_jitter": {
                "enabled": true,
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1
            },
            "normalize": {
                "enabled": true,
                "mean": [
                    0.5
                ],
                "std": [
                    0.5
                ]
            }
        },
        "execution_flags": {
            "mode": "train_and_predict",
            "use_gpu": false,
            "mixed_precision": true,
            "distributed_training": false,
            "debug_mode": false,
            "use_previous_model": true,
            "fresh_start": false
        },
        "output": {
            "features_file": "data/mnist/mnist.csv",
            "model_dir": "data/mnist/models",
            "visualization_dir": "data/mnist/visualizations"
        }
    }

    # Automatically disable color jitter for grayscale images
    if channels == 1:
        config["augmentation"]["color_jitter"]["enabled"] = False

    # Save the configuration file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Created configuration file: {config_path}")


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
