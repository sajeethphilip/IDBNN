def move_all_to_train(dataset_root: str, train_path: str) -> List[str]:
    """
    Move all image class subfolders to the train directory.
    Handles cases where:
    - There are train/test folders (move their contents to train)
    - Only raw class folders exist (move them to train)
    - There's a containing folder like 256_ObjectCategories (move its contents to train)
    """
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

                    # If class already exists in train, merge the contents
                    if os.path.exists(dest_path):
                        for item in os.listdir(class_path):
                            shutil.move(os.path.join(class_path, item), dest_path)
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
                        shutil.move(class_path, dest_path)
                        class_names.append(class_name)
                # Remove the now-empty containing directory
                shutil.rmtree(item_path)
            else:
                # It's a class folder itself, move it to train
                dest_path = os.path.join(train_path, item)
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
