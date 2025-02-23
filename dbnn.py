import os
import torch
import argparse
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List, Union
from typing import Dict, List, Union, Optional
from basic_utils import Colors


class DebugLogger:
    def __init__(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def log(self, msg, force=False):
        """Only print if enabled or forced"""
        if self.enabled or force:
            print(msg)

# Create single global instance
DEBUG = DebugLogger()

class DBNNInitializer:
    def __init__(self, dataset_name: str, config: Dict):
        self.dataset_name = dataset_name
        self.config = config
        self.device = self._setup_device_and_precision()
        self.model_type = config.get('modelType', 'Histogram')
        self._initialize_model_components()

    def _setup_device_and_precision(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            self.mixed_precision = True
            self.autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=True)
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.device = torch.device('cpu')
            self.mixed_precision = False
            self.autocast_ctx = torch.no_grad
            self.scaler = None
        return self.device

    def _initialize_model_components(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.likelihood_params = None
        self.feature_pairs = None
        self.best_W = None
        self.best_error = float('inf')
        self.current_W = None
        self.categorical_encoders = {}
        os.makedirs('Model', exist_ok=True)


class DBNNDataHandler:
    def __init__(self, dataset_name: str, config: Dict):
        self.dataset_name = dataset_name
        self.config = config
        self.data = self._load_dataset()
        self.target_column = self.config['target_column']

    def _load_dataset(self) -> pd.DataFrame:
        file_path = self.config.get('file_path')
        if file_path is None:
            raise ValueError(f"No file path specified in dataset configuration for: {self.dataset_name}")
        df = pd.read_csv(file_path)
        return df

    def _preprocess_data(self, X: pd.DataFrame, is_training: bool = True) -> torch.Tensor:
        X_processed = X.copy()
        if is_training:
            X_processed = self._remove_high_cardinality_columns(X_processed)
        X_processed = self._encode_categorical_features(X_processed, is_training)
        X_scaled = self.scaler.fit_transform(X_processed) if is_training else self.scaler.transform(X_processed)
        return torch.FloatTensor(X_scaled)

    def _remove_high_cardinality_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        threshold = self.config.get('cardinality_threshold', 0.8)
        columns_to_drop = [col for col in df.columns if df[col].nunique() / len(df) > threshold]
        return df.drop(columns=columns_to_drop)

    def _encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        categorical_columns = self._detect_categorical_columns(df)
        for col in categorical_columns:
            if is_training:
                self.categorical_encoders[col] = {val: idx for idx, val in enumerate(df[col].unique())}
            df[col] = df[col].map(self.categorical_encoders.get(col, {}))
        return df

    def _detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        return [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() < 50]


class DBNNTrainer:
    def __init__(self, initializer: DBNNInitializer, data_handler: DBNNDataHandler):
        self.initializer = initializer
        self.data_handler = data_handler
        self.error_rates = []

    def train(self, X_train, y_train, batch_size=32):
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        for epoch in range(self.initializer.config.get('epochs', 1000)):
            for i in range(0, n_samples, batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                self._train_batch(batch_X, batch_y)
            self.error_rates.append(self._compute_error(X_train, y_train))

    def _train_batch(self, batch_X, batch_y):
        if self.initializer.model_type == "Histogram":
            posteriors, bin_indices = self._compute_batch_posterior(batch_X)
        else:
            posteriors, _ = self._compute_batch_posterior_std(batch_X)
        predictions = torch.argmax(posteriors, dim=1)
        errors = (predictions != batch_y)
        if errors.any():
            self._update_weights(batch_X[errors], batch_y[errors], predictions[errors], bin_indices)

    def _compute_error(self, X, y):
        predictions = self.predict(X)
        return (predictions != y).float().mean().item()

    def _update_weights(self, failed_cases, true_labels, pred_labels, bin_indices):
        for case, true_label, pred_label in zip(failed_cases, true_labels, pred_labels):
            self._update_weights_for_case(case, true_label, pred_label, bin_indices)

    def _update_weights_for_case(self, case, true_label, pred_label, bin_indices):
        for group_idx in bin_indices:
            bin_i, bin_j = bin_indices[group_idx]
            self.initializer.weight_updater.update_histogram_weights(
                failed_case=case,
                true_class=true_label,
                pred_class=pred_label,
                bin_indices={group_idx: (bin_i, bin_j)},
                posteriors=None,
                learning_rate=self.initializer.config.get('learning_rate', 0.1)
            )


class DBNNPredictor:
    def __init__(self, initializer: DBNNInitializer, data_handler: DBNNDataHandler):
        self.initializer = initializer
        self.data_handler = data_handler

    def predict(self, X: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        predictions = []
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            if self.initializer.model_type == "Histogram":
                posteriors, _ = self._compute_batch_posterior(batch_X)
            else:
                posteriors, _ = self._compute_batch_posterior_std(batch_X)
            predictions.append(torch.argmax(posteriors, dim=1))
        return torch.cat(predictions)

    def _compute_batch_posterior(self, features: torch.Tensor):
        if self.initializer.likelihood_params is None:
            raise RuntimeError("Likelihood parameters not initialized")
        # Compute posteriors using likelihood_params
        pass

    def _compute_batch_posterior_std(self, features: torch.Tensor):
        # Compute posteriors for Gaussian model
        pass


class DBNNReconstructor:
    def __init__(self, initializer: DBNNInitializer, data_handler: DBNNDataHandler):
        self.initializer = initializer
        self.data_handler = data_handler

    def reconstruct_features(self, predictions: torch.Tensor) -> torch.Tensor:
        # Implement feature reconstruction logic
        pass

    def _compute_reconstruction_metrics(self, original_features: torch.Tensor, reconstructed_features: torch.Tensor):
        # Compute reconstruction metrics
        pass


class DBNNUtils:
    def __init__(self, initializer: DBNNInitializer):
        self.initializer = initializer

    def save_weights(self):
        if self.initializer.best_W is not None:
            weights_dict = {
                'weights': self.initializer.best_W.cpu().numpy().tolist(),
                'shape': list(self.initializer.best_W.shape)
            }
            with open(self._get_weights_filename(), 'w') as f:
                json.dump(weights_dict, f)

    def load_weights(self):
        weights_file = self._get_weights_filename()
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                weights_dict = json.load(f)
                self.initializer.best_W = torch.tensor(weights_dict['weights'], device=self.initializer.device)

    def _get_weights_filename(self):
        return os.path.join('Model', f'Best_{self.initializer.model_type}_{self.initializer.dataset_name}_weights.json')

class DatasetProcessor:
    def __init__(self):
        self.base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        self.compressed_extensions = ['.zip', '.gz', '.tar', '.7z', '.rar']
        self.debug = DebugLogger()
        self.colors = Colors()

    def process_dataset(self, file_path: str) -> None:
        """Process dataset with proper path handling.

        Args:
            file_path: Path to the dataset file
        """
        try:
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            # Create main data directory if it doesn't exist
            if not os.path.exists('data'):
                os.makedirs('data')

            # Setup dataset folder structure
            dataset_folder = os.path.join('data', base_name)
            os.makedirs(dataset_folder, exist_ok=True)

            print(f"\nProcessing dataset:")
            print(f"Base name: {base_name}")
            print(f"Dataset folder: {dataset_folder}")

            # Define target CSV path
            target_csv = os.path.join(dataset_folder, f"{base_name}.csv")

            # If file exists at original path and isn't in dataset folder, copy it
            if os.path.exists(file_path) and os.path.isfile(file_path) and file_path != target_csv:
                try:
                    import shutil
                    shutil.copy2(file_path, target_csv)
                    print(f"Copied dataset to: {target_csv}")
                except Exception as e:
                    print(f"Warning: Could not copy dataset: {str(e)}")

            # If file doesn't exist in target location, try downloading from UCI
            if not os.path.exists(target_csv):
                print(f"File not found locally: {target_csv}")
                print("Attempting to download from UCI repository...")
                downloaded_path = self._download_from_uci(base_name.upper())
                if downloaded_path:
                    print(f"Successfully downloaded dataset to {downloaded_path}")
                    # Ensure downloaded file is in the correct location
                    if downloaded_path != target_csv:
                        try:
                            import shutil
                            shutil.move(downloaded_path, target_csv)
                        except Exception as e:
                            print(f"Warning: Could not move downloaded file: {str(e)}")
                else:
                    print(f"Could not find or download dataset: {base_name}")
                    return None

            # Verify file exists before proceeding
            if not os.path.exists(target_csv):
                raise FileNotFoundError(f"Dataset file not found at {target_csv}")

            # Process based on dataset structure
            config = self._create_dataset_configs(dataset_folder, base_name)

            if self._has_test_train_split(dataset_folder, base_name):
                print("Found train/test split structure")
                return self._handle_split_dataset(dataset_folder, base_name)
            elif os.path.exists(target_csv):
                print("Found single CSV file structure")
                return self._handle_single_csv(dataset_folder, base_name, config)
            elif self._is_compressed(file_path):
                print("Found compressed file, extracting...")
                extracted_path = self._decompress(file_path, dataset_folder)
                return self.process_dataset(extracted_path)
            else:
                print(f"Could not determine dataset structure for {dataset_folder}")
                return None

        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
            traceback.print_exc()
            return None
    def _has_single_csv(self, folder_path: str, base_name: str) -> bool:
        """Check if dataset has single CSV file"""
        # Check both possible locations
        csv_paths = [
            os.path.join(folder_path, f"{base_name}.csv"),
            os.path.join(folder_path, base_name, f"{base_name}.csv")
        ]
        exists = any(os.path.exists(path) for path in csv_paths)
        if exists:
            found_path = next(path for path in csv_paths if os.path.exists(path))
            print(f"Found CSV file: {found_path}")
        return exists

    def _has_test_train_split(self, folder_path: str, base_name: str) -> bool:
        """Check for train/test split in dataset folder structure"""
        dataset_folder = os.path.join(folder_path, base_name)
        train_path = os.path.join(dataset_folder, 'train')
        test_path = os.path.join(dataset_folder, 'test')

        # Check if both train and test folders exist
        has_folders = os.path.exists(train_path) and os.path.exists(test_path)

        if has_folders:
            # Check for either dataset-named files or train.csv/test.csv
            train_files = [
                os.path.join(train_path, f"{base_name}.csv"),
                os.path.join(train_path, "train.csv")
            ]
            test_files = [
                os.path.join(test_path, f"{base_name}.csv"),
                os.path.join(test_path, "test.csv")
            ]

            has_train = any(os.path.exists(f) for f in train_files)
            has_test = any(os.path.exists(f) for f in test_files)

            if has_train and has_test:
                train_file = next(f for f in train_files if os.path.exists(f))
                test_file = next(f for f in test_files if os.path.exists(f))
                print(f"Found train file: {train_file}")
                print(f"Found test file: {test_file}")
                return True

        return False


    def find_dataset_pairs(self, data_dir: str = 'data') -> List[Tuple[str, str, str]]:
            """Find and validate dataset configuration pairs.

            Args:
                data_dir: Base directory to search for datasets

            Returns:
                List of tuples (dataset_name, config_path, csv_path)
            """
            if not os.path.exists(data_dir):
                print(f"\nNo '{data_dir}' directory found. Creating one...")
                os.makedirs(data_dir)
                return []

            dataset_pairs = []
            processed_datasets = set()
            adaptive_conf = self._load_global_adaptive_config()

            # Walk through all subdirectories
            for root, dirs, files in os.walk(data_dir):
                conf_files = [f for f in files if f.endswith('.conf') and f != 'adaptive_dbnn.conf']

                for conf_file in conf_files:
                    basename = os.path.splitext(conf_file)[0]
                    if basename in processed_datasets:
                        continue

                    conf_path = os.path.join(root, conf_file)

                    # Check for CSV in multiple possible locations
                    csv_paths = [
                        os.path.join(root, f"{basename}.csv"),                     # Same directory as conf
                        os.path.join(root, basename, f"{basename}.csv"),          # Subdirectory
                        os.path.join(root, basename, 'train', f"{basename}.csv"), # Train directory
                        os.path.join(root, basename, 'train', "train.csv"),       # Train directory with default name
                        os.path.join(root, 'train', f"{basename}.csv"),          # Direct train directory
                        os.path.join(root, 'train', "train.csv")                 # Direct train directory with default name
                    ]

                    # Find first existing CSV file
                    csv_path = next((path for path in csv_paths if os.path.exists(path)), None)

                    if csv_path:
                        if adaptive_conf:
                            self._update_config_with_adaptive(conf_path, adaptive_conf)

                        print(f"\nFound dataset: {basename}")
                        print(f"Config: {conf_path}")
                        print(f"Data: {csv_path}")

                        dataset_pairs.append((basename, conf_path, csv_path))
                        processed_datasets.add(basename)

            return dataset_pairs

    def _load_global_adaptive_config(self) -> Dict:
        adaptive_path = 'adaptive_dbnn.conf'
        if os.path.exists(adaptive_path):
            try:
                with open(adaptive_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.debug.log(f"Warning: Could not load adaptive configuration: {str(e)}")
        return {}

    def _update_config_with_adaptive(self, conf_path: str, adaptive_conf: Dict):
        """Update dataset configuration with global adaptive settings."""
        try:
            with open(conf_path, 'r') as f:
                dataset_conf = json.load(f)

            # Update execution flags
            if 'execution_flags' in adaptive_conf:
                dataset_conf['execution_flags'] = adaptive_conf['execution_flags']

            # Update training parameters
            if 'training_params' in adaptive_conf:
                if 'training_params' not in dataset_conf:
                    dataset_conf['training_params'] = {}
                dataset_conf['training_params'].update(adaptive_conf['training_params'])

            with open(conf_path, 'w') as f:
                json.dump(dataset_conf, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not update configuration: {str(e)}")
            traceback.print_exc()


    def _handle_split_dataset(self, folder_path: str, base_name: str):
        """Handle dataset with train/test split following specific folder structure rules"""
        # Setup paths
        dataset_folder = os.path.join(folder_path, base_name)
        train_path = os.path.join(dataset_folder, 'train')
        test_path = os.path.join(dataset_folder, 'test')
        main_csv_path = os.path.join(dataset_folder, f"{base_name}.csv")

        # Load configuration
        config = self._validate_config(folder_path, base_name)
        model = DBNN(base_name, config)

        if config.get('modelType', 'Histogram') == 'Histogram':
            if input("Merge train/test data? (y/n): ").lower() == 'y':
                # Check if merged file already exists
                if os.path.exists(main_csv_path):
                    print(f"Using existing merged dataset: {main_csv_path}")
                    merged_df = pd.read_csv(main_csv_path)
                else:
                    print("Merging train and test datasets...")
                    # Look for dataset-named files first, then fall back to train.csv/test.csv
                    train_file = os.path.join(train_path, f"{base_name}.csv")
                    if not os.path.exists(train_file):
                        train_file = os.path.join(train_path, "train.csv")

                    test_file = os.path.join(test_path, f"{base_name}.csv")
                    if not os.path.exists(test_file):
                        test_file = os.path.join(test_path, "test.csv")

                    try:
                        train_df = pd.read_csv(train_file)
                        test_df = pd.read_csv(test_file)
                        merged_df = pd.concat([train_df, test_df], ignore_index=True)

                        # Save merged file in dataset folder
                        os.makedirs(dataset_folder, exist_ok=True)
                        merged_df.to_csv(main_csv_path, index=False)
                        print(f"Saved merged dataset to: {main_csv_path}")
                    except Exception as e:
                        print(f"Error merging datasets: {str(e)}")
                        return None

                # Process merged dataset
                results = self.run_benchmark(base_name, model)
                return self._save_results(results, dataset_folder, base_name)

            else:
                # Use separate train and test files
                print("Using separate train and test datasets...")
                # Try dataset-named files first
                train_file = os.path.join(train_path, f"{base_name}.csv")
                if not os.path.exists(train_file):
                    train_file = os.path.join(train_path, "train.csv")
                    if not os.path.exists(train_file):
                        print(f"Error: No training file found in {train_path}")
                        return None

                test_file = os.path.join(test_path, f"{base_name}.csv")
                if not os.path.exists(test_file):
                    test_file = os.path.join(test_path, "test.csv")
                    if not os.path.exists(test_file):
                        print(f"Error: No test file found in {test_path}")
                        return None

                try:
                    train_df = pd.read_csv(train_file)
                    test_df = pd.read_csv(test_file)
                    print(f"Using training data from: {train_file}")
                    print(f"Using test data from: {test_file}")
                    results = self.run_benchmark(base_name, model, train_df, test_df)
                    return self._save_results(results, dataset_folder, base_name)
                except Exception as e:
                    print(f"Error processing split datasets: {str(e)}")
                    return None

    # In DatasetProcessor class
    def _handle_single_csv(self, folder_path: str, base_name: str, config: Dict):
        """Handle dataset with single CSV file and debug config processing"""
        #print("\nDEBUGEntering _handle_single_csv")
        # print(f"DEBUG:  Initial config: {json.dumps(config, indent=2) if config else 'None'}")

        # Handle CSV paths
        csv_paths = [
            os.path.join(folder_path, f"{base_name}.csv"),
            os.path.join(folder_path, base_name, f"{base_name}.csv")
        ]
        csv_path = next((path for path in csv_paths if os.path.exists(path)), None)

        if not csv_path:
            return None

        # Ensure we have a valid config
        if config is None:
           ## print("DEBUG: No config provided, validating...")
            config = self._validate_config(folder_path, base_name)

        ##print("\nDEBUGConfig before GlobalConfig conversion:")
        #print(json.dumps(config, indent=2))

        # Create GlobalConfig
        global_config = GlobalConfig.from_dict(config)

        # Create and return DBNN model
        model = DBNN(base_name, global_config)
        results = self.run_benchmark(base_name, model)
        return self._save_results(results, folder_path, base_name)

    def _try_uci_download(self, dataset_name: str, folder_path: str, config: Dict):
       if dataset_path := self._download_from_uci(dataset_name.upper()):
           return self.process_dataset(dataset_path)
       print(f"Could not find or download dataset: {dataset_name}")
       return None


    def process_by_path(self, file_path: str) -> None:
        """Main entry point for processing a dataset by path"""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        folder_path = os.path.join('data', base_name)

        # Sequential processing logic
        processors = [
            self._handle_split_dataset,
            self._handle_single_csv,
            self._handle_compressed,
            self._handle_uci_download
        ]

        for processor in processors:
            if result := processor(folder_path, base_name):
                return result

        print(f"Could not process dataset: {file_path}")

    def save_results(self, results: Dict, dataset_name: str) -> None:
        """Save comprehensive results to text file.

        Args:
            results: Dictionary containing all results
            dataset_name: Name of the dataset
        """
        results_dir = os.path.join('data', dataset_name)
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f'{dataset_name}_results.txt')

        with open(results_path, 'w') as f:
            # Header
            f.write(f"Results for Dataset: {dataset_name}\n\n")

            # Classification Report
            if 'classification_report' in results:
                f.write("Classification Report:\n")
                f.write(results['classification_report'])
                f.write("\n\n")

            # Confusion Matrix
            if 'confusion_matrix' in results:
                f.write("Confusion Matrix:\n")
                matrix = results['confusion_matrix']
                f.write("\n".join(["\t".join(map(str, row)) for row in matrix]))
                f.write("\n\n")

            # Error Rates
            if 'error_rates' in results:
                f.write("Error Rates:\n")
                error_rates = results['error_rates']
                if error_rates:
                    for i, rate in enumerate(error_rates):
                        f.write(f"Epoch {i+1}: {rate:.4f}\n")
                else:
                    f.write("N/A\n")
                f.write("\n")

            # Test Accuracy
            if 'test_accuracy' in results:
                f.write(f"Test Accuracy: {results['test_accuracy']:.4f}\n\n")

            # Reconstruction Metrics
            if 'reconstruction_metrics' in results:
                f.write("Reconstruction Metrics:\n")
                recon_metrics = results['reconstruction_metrics']
                if 'final_reconstruction_error' in recon_metrics:
                    error = recon_metrics['final_reconstruction_error']
                    if error is not None:
                        f.write(f"Final Reconstruction Error: {error:.6f}\n")
                f.write("\n")

            # Save paths
            if 'reconstruction_path' in results:
                f.write("Output Files:\n")
                f.write(f"Reconstruction data: {results['reconstruction_path']}\n")
                if 'metadata_path' in results:
                    f.write(f"Reconstruction metadata: {results['metadata_path']}\n")
                f.write("\n")

        print(f"Results saved to {results_path}")

    def _save_results(self, results_tuple, folder_path: str, dataset_name: str):
            """Save formatted results to a text file.

            Args:
                results_tuple: Tuple of (model, results) from run_benchmark
                folder_path: Path to save results
                dataset_name: Name of the dataset
            """
            results_path = os.path.join(folder_path, f"{dataset_name}_results.txt")

            # Unpack the tuple
            model, results = results_tuple

            # Format the results into a human-readable string
            result_text = f"Results for Dataset: {dataset_name}\n\n"

            # Add classification report
            result_text += "Classification Report:\n"
            if isinstance(results, dict):
                result_text += results.get('classification_report', 'N/A') + "\n\n"
            else:
                result_text += "N/A\n\n"

            # Add confusion matrix
            result_text += "Confusion Matrix:\n"
            if isinstance(results, dict) and results.get('confusion_matrix') is not None:
                confusion_matrix = results['confusion_matrix']
                result_text += "\n".join(["\t".join(map(str, row)) for row in confusion_matrix]) + "\n\n"
            else:
                result_text += "N/A\n\n"

            # Add error rates
            result_text += "Error Rates:\n"
            if isinstance(results, dict):
                error_rates = results.get('error_rates', [])
                if error_rates:
                    result_text += "\n".join([f"Epoch {i+1}: {rate:.4f}" for i, rate in enumerate(error_rates)]) + "\n\n"
                else:
                    result_text += "N/A\n\n"
            else:
                result_text += "N/A\n\n"

            # Add test accuracy
            if isinstance(results, dict):
                result_text += f"Test Accuracy: {results.get('test_accuracy', 'N/A')}\n"
            else:
                result_text += "Test Accuracy: N/A\n"

            # Write the formatted results to a text file
            with open(results_path, 'w') as f:
                f.write(result_text)

            print(f"\nResults saved to {results_path}")
            return results_tuple

    def run_benchmark(self, dataset_name: str, model=None, batch_size: int = 32):
       """Complete benchmarking implementation with full debug trace."""
       try:
           print(f"\nBenchmarking {self.colors.highlight_dataset(dataset_name)}")

           #print("\nDEBUGConfiguration Loading Phase")
           if hasattr(model.config, 'to_dict'):
               config_dict = model.config.to_dict()
           elif isinstance(model.config, dict):
               config_dict = model.config.copy()
           else:
               config_dict = {
                   'execution_flags': {'train': True},
                   'training_params': {'enable_adaptive': True}
               }

           should_train = config_dict.get('execution_flags', {}).get('train', True)
           enable_adaptive = config_dict.get('training_params', {}).get('enable_adaptive', True)
           invert_DBNN = config_dict.get('training_params', {}).get('invert_DBNN', False)

           print("\nDEBUG: Execution Flags:")
           print(f"- should_train: {should_train}")
           print(f"- enable_adaptive: {enable_adaptive}")
           print(f"- invert_DBNN: {invert_DBNN}")

           if not should_train:
               # Check for existence of model files
               weights_exist = os.path.exists(model._get_weights_filename())
               components_exist = os.path.exists(model._get_model_components_filename())
               predictions_exist = os.path.exists(f"{dataset_name}_predictions.csv")

               if weights_exist and components_exist:
                   print("Found existing model files, loading predictions...")
                   results = model.predict_and_save(
                       save_path=f"{dataset_name}_predictions.csv",
                       batch_size=batch_size
                   )
                   if results is not None:
                       return model, results

               missing_files = []
               if not weights_exist:
                   missing_files.append("model weights")
               if not components_exist:
                   missing_files.append("model components")
               if not predictions_exist:
                   missing_files.append("prediction file")

               print(f"\nMissing required files: {', '.join(missing_files)}")
               response = input("Training is disabled but required files are missing. Enable training? (y/n): ")
               if response.lower() != 'y':
                   print("Exiting without training")
                   return model, {"error": "Required files missing and training disabled"}
               else:
                   should_train = True
                   config_dict['execution_flags']['train'] = True

           if invert_DBNN:
               print("\nDEBUG: Inverse DBNN Settings:")
               for param in ['reconstruction_weight', 'feedback_strength', 'inverse_learning_rate']:
                   value = config_dict.get('training_params', {}).get(param, 0.1)
                   #print(f"- {param}: {value}")
              ## print("DEBUG: Initializing inverse model...")

               if not should_train:
                   inverse_model_path = os.path.join('Model', f'Best_inverse_{dataset_name}', 'inverse_model.pt')
                   if not os.path.exists(inverse_model_path):
                       print("Inverse model file missing")
                       if input("Train inverse model? (y/n): ").lower() != 'y':
                           print("Skipping inverse model")
                           config_dict['training_params']['invert_DBNN'] = False

           print("\nDEBUG: Starting Processing Phase")
           if should_train:
               if enable_adaptive:
                  ## print("DEBUG: Running adaptive training...")
                   history = model.adaptive_fit_predict(
                       max_rounds=model.max_epochs,
                       batch_size=batch_size
                   )
                  ## print("DEBUG: Adaptive training completed")

              ## print("DEBUG: Running prediction and save...")
               results = model.predict_and_save(
                   save_path=f"{dataset_name}_predictions.csv",
                   batch_size=batch_size
               )
              ## print("DEBUG: Prediction completed")

               if not isinstance(results, dict):
                   if hasattr(history, 'get'):
                       results = history
                   else:
                       results = {
                           'predictions': results if torch.is_tensor(results) else None,
                           'error_rates': getattr(model, 'error_rates', []),
                           'confusion_matrix': getattr(model, 'confusion_matrix', None),
                           'classification_report': getattr(model, 'classification_report', '')
                       }

               if invert_DBNN and hasattr(model, 'inverse_model'):
                   try:
                      ## print("DEBUG: Processing inverse model...")
                       X_test = model.data.drop(columns=[model.target_column])
                       test_probs = model._get_test_probabilities(X_test)
                       reconstruction_features = model.inverse_model.reconstruct_features(test_probs)
                       results = model.update_results_with_reconstruction(
                           results, X_test, reconstruction_features,
                           test_probs, model.y_tensor,
                           f"{dataset_name}_predictions.csv"
                       )
                   except Exception as e:
                       print(f"Error in inverse model processing: {str(e)}")
                       traceback.print_exc()

               return model, results

           return model, results

       except Exception as e:
           print("\nDEBUG: Error in benchmark")
           print("-" * 50)
           print(f"Error type: {type(e).__name__}")
           print(f"Error message: {str(e)}")
           traceback.print_exc()
           return None

    def plot_training_progress(self, error_rates: List[float], dataset_name: str):
        plt.figure(figsize=(10, 6))
        plt.plot(error_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Error Rate')
        plt.title(f'Training Progress - {dataset_name.capitalize()}')
        plt.grid(True)
        plt.savefig(f'{dataset_name}_training_progress.png')
        plt.close()

    def plot_confusion_matrix(self, confusion_mat: np.ndarray, class_names: np.ndarray, dataset_name: str):
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {dataset_name.capitalize()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{dataset_name}_confusion_matrix.png')
        plt.close()




    def _handle_compressed(self, folder_path: str, dataset_name: str):
        file_path = os.path.join(folder_path, f"{dataset_name}")
        if not self._is_compressed(file_path):
            return None

        extracted_path = self._decompress(file_path)
        return self.process_by_path(extracted_path)

    def _handle_uci_download(self, folder_path: str, dataset_name: str):
        if dataset_path := self._download_from_uci(dataset_name.upper()):
            return self.process_by_path(dataset_path)
        return None

    def _validate_inverse_config(self) -> bool:
        """
        Validate inverse DBNN configuration without modification.
        Only checks for presence and validity of required parameters.
        """
        if not hasattr(self.config, 'to_dict') and not isinstance(self.config, dict):
            print("Warning: Invalid configuration object")
            return False

        # Check if inverse DBNN is enabled (respect existing value)
        invert_DBNN = self._get_config_param('invert_DBNN', False)
        if not invert_DBNN:
            return False

        # Only validate presence and basic type checking of required parameters
        required_params = {
            'reconstruction_weight': float,
            'feedback_strength': float,
            'inverse_learning_rate': float
        }

        for param, expected_type in required_params.items():
            value = self._get_config_param(param, None)
            if value is None:
                print(f"Missing required inverse parameter: {param}")
                return False
            if not isinstance(value, expected_type):
                print(f"Invalid type for {param}: expected {expected_type.__name__}, got {type(value).__name__}")
                return False

        return True

    def _validate_config(self, folder_path: str, base_name: str) -> Dict:
        """Validate and load configuration"""
        config_path = os.path.join(folder_path, f"{base_name}.conf")

        try:
            # # print(f"\nDEBUG: Loading config from {config_path}")

            with open(config_path, 'r') as f:
                config = json.load(f)
                # # print(f"DEBUG:   Loaded raw config: {json.dumps(config, indent=2)}")

            # Ensure required sections exist
            if 'training_params' not in config:
                config['training_params'] = {}
            if 'execution_flags' not in config:
                config['execution_flags'] = {}

            return config

        except Exception as e:
            print(f"ERROR: Failed to load config: {str(e)}")
            return None

    def _create_or_load_dataset_config(self, folder_path: str, dataset_name: str) -> Dict:
       """Create or load dataset-specific configuration"""
       config_path = os.path.join(folder_path, f"{dataset_name}.conf")

       if os.path.exists(config_path):
           with open(config_path, 'r') as f:
               return json.load(f)

       # Create default dataset config
       csv_path = os.path.join(folder_path, f"{dataset_name}.csv")
       df = pd.read_csv(csv_path, nrows=0)

       default_config = {
           "file_path": csv_path,
           "column_names": df.columns.tolist(),
           "separator": ",",
           "has_header": True,
           "target_column": df.columns[-1],
           "likelihood_config": {
               "feature_group_size": 2,
               "max_combinations": 1000,
               "bin_sizes": [20]
           },
           "active_learning": {
               "tolerance": 1.0,
               "cardinality_threshold_percentile": 95,
               "strong_margin_threshold": 0.3,
               "marginal_margin_threshold": 0.1,
               "min_divergence": 0.1
           },
           "training_params": {
               "Save_training_epochs": True,
               "training_save_path": f"training_data/{dataset_name}"
           },
           "modelType": "Histogram"
       }

       with open(config_path, 'w') as f:
           json.dump(default_config, f, indent=4)

       return default_config

    def _create_dataset_configs(self, folder_path: str, dataset_name: str) -> Dict:
       """Create or load both dataset and adaptive configs"""
       dataset_config = self._create_or_load_dataset_config(folder_path, dataset_name)
       adaptive_config = self._create_or_load_adaptive_config(folder_path, dataset_name)
       return self._merge_configs(dataset_config, adaptive_config)

    def _create_or_load_adaptive_config(self, folder_path: str, dataset_name: str) -> Dict:
       """Create or load dataset-specific adaptive config"""
       adaptive_path = os.path.join(folder_path, 'adaptive_dbnn.conf')
       if os.path.exists(adaptive_path):
           with open(adaptive_path, 'r') as f:
               return json.load(f)

       default_adaptive = {
           "training_params": {
               "trials": 100,
               "cardinality_threshold": 0.9,
               "cardinality_tolerance": 4,
               "learning_rate": 0.1,
               "random_seed": 42,
               "epochs": 100,
               "test_fraction": 0.2,
               "enable_adaptive": True,
               "modelType": "Histogram",
               "compute_device": "auto",
               "use_interactive_kbd": False,
               "debug_enabled": True,
               "Save_training_epochs": True,
               "training_save_path": f"training_data/{dataset_name}"
           },
           "execution_flags": {
               "train": True,
               "train_only": False,
               "predict": True,
               "gen_samples": False,
               "fresh_start": False,
               "use_previous_model": True
           }
       }

       with open(adaptive_path, 'w') as f:
           json.dump(default_adaptive, f, indent=4)
       return default_adaptive

    def _merge_configs(self, dataset_config: Dict, adaptive_config: Dict) -> Dict:
       """Merge dataset and adaptive configs with adaptive taking precedence"""
       merged = dataset_config.copy()
       if 'training_params' in adaptive_config:
           merged['training_params'].update(adaptive_config['training_params'])
       if 'execution_flags' in adaptive_config:
           merged['execution_flags'] = adaptive_config['execution_flags']
       return merged

    def _create_default_config(self, folder_path: str, dataset_name: str) -> Dict:
        csv_path = os.path.join(folder_path, f"{dataset_name}.csv")
        df = pd.read_csv(csv_path, nrows=0)

        config = {
            "file_path": csv_path,
            "column_names": df.columns.tolist(),
            "separator": ",",
            "has_header": True,
            "target_column": df.columns[-1],
            "likelihood_config": {
                "feature_group_size": 2,
                "max_combinations": 1000,
                "bin_sizes": [20]
            },
            "active_learning": {
                "tolerance": 1.0,
                "cardinality_threshold_percentile": 95,
                "strong_margin_threshold": 0.3,
                "marginal_margin_threshold": 0.1,
                "min_divergence": 0.1
            },
            "training_params": {
                "trials": 100,
                "cardinality_threshold": 0.9,
                "cardinality_tolerance": 4,
                "learning_rate": 0.1,
                "random_seed": 42,
                "epochs": 1000,
                "test_fraction": 0.2,
                "enable_adaptive": True,
                "Save_training_epochs": True,
                "training_save_path": f"training_data/{dataset_name}",
                "modelType": "Histogram",
                "minimum_training_accuracy": 0.95,  # Added default value
                "enable_vectorized": False,  # Default to classic training
                "vectorization_warning_acknowledged": False  # Track if user has acknowledged
            },
            "execution_flags": {
                "train": True,
                "train_only": False,
                "predict": True,
                "fresh_start": False,
                "use_previous_model": True
            }
        }

        config_path = os.path.join(folder_path, f"{dataset_name}.conf")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        return config

    def _validate_and_update_config(self, config: Dict, folder_path: str) -> Dict:
        required_fields = ['file_path', 'column_names', 'target_column']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        return config

    def _merge_data(self, train_path: str, test_path: str) -> pd.DataFrame:
        train_df = pd.read_csv(os.path.join(train_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(test_path, "test.csv"))
        return pd.concat([train_df, test_df], ignore_index=True)

    def _process_merged_data(self, df: pd.DataFrame, config: Dict):
        model = DBNN(config)
        return model.fit(df)

    def _process_split_data(self, train_path: str, test_path: str, config: Dict):
        model = DBNN(config)
        train_df = pd.read_csv(os.path.join(train_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(test_path, "test.csv"))
        model.fit(train_df)
        return model.evaluate(test_df)

    def _process_single_file(self, file_path: str, config: Dict):
        model = DBNN(config)
        df = pd.read_csv(file_path)
        return model.fit(df)

    def _is_compressed(self, file_path: str) -> bool:
        return any(file_path.endswith(ext) for ext in self.compressed_extensions)

    def _decompress(self, file_path: str) -> str:
        extract_path = os.path.join('data', 'temp')
        os.makedirs(extract_path, exist_ok=True)

        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        elif file_path.endswith(('.tar', '.gz')):
            with tarfile.open(file_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_path)
        elif file_path.endswith('.7z'):
            with py7zr.SevenZipFile(file_path, 'r') as sz_ref:
                sz_ref.extractall(extract_path)

        return extract_path

    def _download_from_uci(self, dataset_name: str) -> Optional[str]:
        """Download dataset from UCI repository"""
        folder_path = os.path.join('data', dataset_name.lower())
        os.makedirs(folder_path, exist_ok=True)

        save_path = os.path.join(folder_path, f"{dataset_name.lower()}.csv")

        # Try different UCI repository URL patterns
        url_patterns = [
            f"{self.base_url}/{dataset_name}/{dataset_name}.data",
            f"{self.base_url}/{dataset_name.lower()}/{dataset_name.lower()}.data",
            f"{self.base_url}/{dataset_name}/{dataset_name}.csv",
            f"{self.base_url}/{dataset_name.lower()}/{dataset_name.lower()}.csv"
        ]

        for url in url_patterns:
            try:
                print(f"Trying URL: {url}")
                response = requests.get(url)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Successfully downloaded to {save_path}")
                    return save_path
            except Exception as e:
                self.debug.log(f"Failed to download from {url}: {str(e)}")
                continue

        return None

    @staticmethod
    def generate_test_datasets():
        """Generate synthetic test datasets"""
        datasets = {
            'xor.csv': [
                'x1,x2,target\n',
                *['0,0,0\n0,1,1\n1,0,1\n1,1,0\n' * 3]
            ],
            'xor3d.csv': [
                'x1,x2,x3,target\n',
                *['0,0,0,0\n0,0,1,1\n0,1,0,1\n0,1,1,1\n1,0,0,1\n1,0,1,1\n1,1,0,1\n1,1,1,0\n' * 3]
            ]
        }

        for filename, content in datasets.items():
            with open(filename, 'w') as f:
                f.writelines(content)

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from tqdm import tqdm

class InvertibleDBNN(torch.nn.Module):
    """Enhanced Invertible Deep Bayesian Neural Network implementation with proper gradient tracking"""

    def __init__(self,
                 forward_model: 'DBNN',
                 feature_dims: int,
                 reconstruction_weight: float = 0.5,
                 feedback_strength: float = 0.3,
                 debug: bool = False):
        """
        Initialize the invertible DBNN.

        Args:
            forward_model: The forward DBNN model
            feature_dims: Number of input feature dimensions
            reconstruction_weight: Weight for reconstruction loss (0-1)
            feedback_strength: Strength of reconstruction feedback (0-1)
            debug: Enable debug logging
        """
        super(InvertibleDBNN, self).__init__()
        self.forward_model = forward_model
        self.device = forward_model.device
        self.feature_dims = feature_dims
        self.reconstruction_weight = reconstruction_weight
        self.feedback_strength = feedback_strength

        # Enable logging if debug is True
        self.debug = debug
        if debug:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)

        # Initialize model components
        self.n_classes = len(self.forward_model.label_encoder.classes_)
        self.inverse_likelihood_params = None
        self.inverse_feature_pairs = None

        # Feature scaling parameters as buffers
        self.register_buffer('min_vals', None)
        self.register_buffer('max_vals', None)
        self.register_buffer('scale_factors', None)

        # Metrics tracking
        self.metrics = {
            'reconstruction_errors': [],
            'forward_errors': [],
            'total_losses': [],
            'accuracies': []
        }

        # Initialize all components
        self._initialize_inverse_components()

    def save_inverse_model(self, custom_path: str = None) -> bool:
        try:
            save_dir = custom_path or os.path.join('Model', f'Best_inverse_{self.forward_model.dataset_name}')
            os.makedirs(save_dir, exist_ok=True)

            # Save model state
            model_state = {
                'weight_linear': self.weight_linear.data,
                'weight_nonlinear': self.weight_nonlinear.data,
                'bias_linear': self.bias_linear.data,
                'bias_nonlinear': self.bias_nonlinear.data,
                'feature_attention': self.feature_attention.data,
                'layer_norm': self.layer_norm.state_dict(),
                'metrics': self.metrics,
                'feature_dims': self.feature_dims,
                'n_classes': self.n_classes,
                'reconstruction_weight': self.reconstruction_weight,
                'feedback_strength': self.feedback_strength
            }

            # Save scale parameters if they exist
            for param in ['min_vals', 'max_vals', 'scale_factors', 'inverse_feature_pairs']:
                if hasattr(self, param):
                    model_state[param] = getattr(self, param)

            model_path = os.path.join(save_dir, 'inverse_model.pt')
            torch.save(model_state, model_path)

            # Save config
            config = {
                'feature_dims': self.feature_dims,
                'reconstruction_weight': float(self.reconstruction_weight),
                'feedback_strength': float(self.feedback_strength),
                'n_classes': int(self.n_classes),
                'device': str(self.device)
            }

            config_path = os.path.join(save_dir, 'inverse_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            print(f"Saved inverse model to {save_dir}")
            return True

        except Exception as e:
            print(f"Error saving inverse model: {str(e)}")
            traceback.print_exc()
            return False

    def load_inverse_model(self, custom_path: str = None) -> bool:
       try:
           load_dir = custom_path or os.path.join('Model', f'Best_inverse_{self.forward_model.dataset_name}')
           model_path = os.path.join(load_dir, 'inverse_model.pt')
           config_path = os.path.join(load_dir, 'inverse_config.json')

           if not (os.path.exists(model_path) and os.path.exists(config_path)):
               print(f"No saved inverse model found at {load_dir}")
               return False

           model_state = torch.load(model_path, map_location=self.device, weights_only=True)

           with open(config_path, 'r') as f:
               config = json.load(f)

           if config['feature_dims'] != self.feature_dims or config['n_classes'] != self.n_classes:
               raise ValueError("Model architecture mismatch")

           # Load parameters
           self.weight_linear.data = model_state['weight_linear']
           self.weight_nonlinear.data = model_state['weight_nonlinear']
           self.bias_linear.data = model_state['bias_linear']
           self.bias_nonlinear.data = model_state['bias_nonlinear']
           self.feature_attention.data = model_state['feature_attention']
           self.layer_norm.load_state_dict(model_state['layer_norm'])

           # Safely update or register buffers
           for param in ['min_vals', 'max_vals', 'scale_factors', 'inverse_feature_pairs']:
               if param in model_state:
                   buffer_data = model_state[param]
                   if buffer_data is not None:
                       if hasattr(self, param) and getattr(self, param) is not None:
                           getattr(self, param).copy_(buffer_data)
                       else:
                           self.register_buffer(param, buffer_data)

           # Restore other attributes
           self.metrics = model_state.get('metrics', {})
           self.reconstruction_weight = model_state.get('reconstruction_weight', 0.5)
           self.feedback_strength = model_state.get('feedback_strength', 0.3)

           print(f"Loaded inverse model from {load_dir}")
           return True

       except Exception as e:
           print(f"Error loading inverse model: {str(e)}")
           traceback.print_exc()
           return False

    def _initialize_inverse_components(self):
        """Initialize inverse model parameters with proper buffer handling"""
        try:
            # Initialize feature pairs
            class_indices = torch.arange(self.n_classes, device=self.device)
            feature_indices = torch.arange(self.feature_dims, device=self.device)
            feature_pairs = torch.cartesian_prod(class_indices, feature_indices)

            # Safely register buffer
            if not hasattr(self, 'inverse_feature_pairs'):
                self.register_buffer('inverse_feature_pairs', feature_pairs)
            else:
                self.inverse_feature_pairs = feature_pairs

            # Number of pairs
            n_pairs = len(feature_pairs)

            # Initialize weights as nn.Parameters
            self.weight_linear = torch.nn.Parameter(
                torch.empty((n_pairs, self.feature_dims), device=self.device),
                requires_grad=True
            )
            self.weight_nonlinear = torch.nn.Parameter(
                torch.empty((n_pairs, self.feature_dims), device=self.device),
                requires_grad=True
            )

            # Initialize with proper scaling
            torch.nn.init.xavier_uniform_(self.weight_linear)
            torch.nn.init.kaiming_normal_(self.weight_nonlinear)

            # Initialize biases as nn.Parameters
            self.bias_linear = torch.nn.Parameter(
                torch.zeros(self.feature_dims, device=self.device),
                requires_grad=True
            )
            self.bias_nonlinear = torch.nn.Parameter(
                torch.zeros(self.feature_dims, device=self.device),
                requires_grad=True
            )

            # Initialize layer normalization
            self.layer_norm = torch.nn.LayerNorm(self.feature_dims).to(self.device)

            # Initialize feature attention
            self.feature_attention = torch.nn.Parameter(
                torch.ones(self.feature_dims, device=self.device),
                requires_grad=True
            )

            # Safely register scaling buffers
            for name in ['min_vals', 'max_vals', 'scale_factors']:
                if not hasattr(self, name):
                    self.register_buffer(name, None)

            if self.debug:
                self.logger.debug(f"Initialized inverse components:")
                self.logger.debug(f"- Feature pairs shape: {self.inverse_feature_pairs.shape}")
                self.logger.debug(f"- Linear weights shape: {self.weight_linear.shape}")
                self.logger.debug(f"- Nonlinear weights shape: {self.weight_nonlinear.shape}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize inverse components: {str(e)}")
    def _compute_feature_scaling(self, features: torch.Tensor):
        """Compute feature scaling parameters for consistent reconstruction"""
        with torch.no_grad():
            self.min_vals = features.min(dim=0)[0]
            self.max_vals = features.max(dim=0)[0]
            self.scale_factors = self.max_vals - self.min_vals
            self.scale_factors[self.scale_factors == 0] = 1.0

    def _scale_features(self, features: torch.Tensor) -> torch.Tensor:
        """Scale features to [0,1] range"""
        return (features - self.min_vals) / self.scale_factors

    def _unscale_features(self, scaled_features: torch.Tensor) -> torch.Tensor:
        """Convert scaled features back to original range"""
        return (scaled_features * self.scale_factors) + self.min_vals

    def _compute_inverse_posterior(self, class_probs: torch.Tensor) -> torch.Tensor:
        """Enhanced inverse posterior computation with improved stability"""
        batch_size = class_probs.shape[0]
        class_probs = class_probs.to(dtype=self.weight_linear.dtype)

        reconstructed_features = torch.zeros(
            (batch_size, self.feature_dims),
            device=self.device,
            dtype=self.weight_linear.dtype
        )

        # Apply attention mechanism
        attention_weights = torch.softmax(self.feature_attention, dim=0)

        # Compute linear and nonlinear transformations
        linear_features = torch.zeros_like(reconstructed_features)
        nonlinear_features = torch.zeros_like(reconstructed_features)

        for feat_idx in range(self.feature_dims):
            # Get relevant pairs for this feature
            relevant_pairs = torch.where(self.inverse_feature_pairs[:, 1] == feat_idx)[0]

            # Get class contributions
            class_contributions = class_probs[:, self.inverse_feature_pairs[relevant_pairs, 0]]

            # Linear transformation
            linear_weights = self.weight_linear[relevant_pairs, feat_idx]
            linear_features[:, feat_idx] = torch.mm(
                class_contributions,
                linear_weights.unsqueeze(1)
            ).squeeze()

            # Nonlinear transformation with tanh activation
            nonlinear_weights = self.weight_nonlinear[relevant_pairs, feat_idx]
            nonlinear_features[:, feat_idx] = torch.tanh(torch.mm(
                class_contributions,
                nonlinear_weights.unsqueeze(1)
            ).squeeze())

        # Combine transformations with attention
        reconstructed_features = (
            attention_weights * linear_features +
            (1 - attention_weights) * nonlinear_features
        )

        # Add biases
        reconstructed_features += self.bias_linear + self.bias_nonlinear

        # Apply layer normalization
        reconstructed_features = self.layer_norm(reconstructed_features)

        return reconstructed_features

    def _compute_reconstruction_loss(self,
                                   original_features: torch.Tensor,
                                   reconstructed_features: torch.Tensor,
                                   class_probs: torch.Tensor,
                                   reduction: str = 'mean') -> torch.Tensor:
        """Enhanced reconstruction loss with multiple components"""
        # Scale features
        orig_scaled = self._scale_features(original_features)
        recon_scaled = self._scale_features(reconstructed_features)

        # MSE loss with feature-wise weighting
        mse_loss = torch.mean((orig_scaled - recon_scaled) ** 2, dim=1)

        # Feature correlation loss
        orig_centered = orig_scaled - orig_scaled.mean(dim=0, keepdim=True)
        recon_centered = recon_scaled - recon_scaled.mean(dim=0, keepdim=True)

        corr_loss = -torch.sum(
            orig_centered * recon_centered, dim=1
        ) / (torch.norm(orig_centered, dim=1) * torch.norm(recon_centered, dim=1) + 1e-8)

        # Distribution matching loss using KL divergence
        orig_dist = torch.distributions.Normal(
            orig_scaled.mean(dim=0),
            orig_scaled.std(dim=0) + 1e-8
        )
        recon_dist = torch.distributions.Normal(
            recon_scaled.mean(dim=0),
            recon_scaled.std(dim=0) + 1e-8
        )
        dist_loss = torch.distributions.kl_divergence(orig_dist, recon_dist).mean()

        # Combine losses with learned weights
        combined_loss = (
            mse_loss +
            0.1 * corr_loss +
            0.01 * dist_loss
        )

        if reduction == 'mean':
            return combined_loss.mean()
        return combined_loss

    def train(self, X_train, y_train, X_test, y_test, batch_size=32):
        """Complete training with optimized test evaluation"""
        print("\nStarting training...")
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Initialize tracking metrics
        error_rates = []
        best_train_accuracy = 0.0
        best_test_accuracy = 0.0
        patience_counter = 0
        plateau_counter = 0
        min_improvement = 0.001
        patience = 5 if self.in_adaptive_fit else 100
        max_plateau = 5
        prev_accuracy = 0.0

        # Main training loop with progress tracking
        with tqdm(total=self.max_epochs, desc="Training epochs") as epoch_pbar:
            for epoch in range(self.max_epochs):
                # Train on all batches
                failed_cases = []
                n_errors = 0

                # Process training batches
                with tqdm(total=n_batches, desc=f"Training batches", leave=False) as batch_pbar:
                    for i in range(0, n_samples, batch_size):
                        batch_end = min(i + batch_size, n_samples)
                        batch_X = X_train[i:batch_end]
                        batch_y = y_train[i:batch_end]

                        # Forward pass and error collection
                        posteriors = self._compute_batch_posterior(batch_X)[0]
                        predictions = torch.argmax(posteriors, dim=1)
                        errors = (predictions != batch_y)
                        n_errors += errors.sum().item()

                        # Collect failed cases for weight updates
                        if errors.any():
                            fail_idx = torch.where(errors)[0]
                            for idx in fail_idx:
                                failed_cases.append((
                                    batch_X[idx],
                                    batch_y[idx].item(),
                                    posteriors[idx].cpu().numpy()
                                ))
                        batch_pbar.update(1)

                # Update weights after processing all batches
                if failed_cases:
                    self._update_priors_parallel(failed_cases, batch_size)

                # Calculate training metrics
                train_error_rate = n_errors / n_samples
                train_accuracy = 1 - train_error_rate
                error_rates.append(train_error_rate)

                # Evaluate on test set once per epoch
                if X_test is not None and y_test is not None:
                    test_predictions = self.predict(X_test, batch_size=batch_size)
                    test_accuracy = (test_predictions == y_test.cpu()).float().mean().item()

                    # Print confusion matrix only for best test performance
                    if test_accuracy > best_test_accuracy:
                        best_test_accuracy = test_accuracy
                        print("\nTest Set Performance:")
                        y_test_labels = self.label_encoder.inverse_transform(y_test.cpu().numpy())
                        test_pred_labels = self.label_encoder.inverse_transform(test_predictions.cpu().numpy())
                        self.print_colored_confusion_matrix(y_test_labels, test_pred_labels)

                # Update progress bar with metrics
                epoch_pbar.set_postfix({
                    'train_acc': f"{train_accuracy:.4f}",
                    'best_train': f"{best_train_accuracy:.4f}",
                    'test_acc': f"{test_accuracy:.4f}",
                    'best_test': f"{best_test_accuracy:.4f}"
                })
                epoch_pbar.update(1)

                # Check improvement and update tracking
                accuracy_improvement = train_accuracy - prev_accuracy
                if accuracy_improvement <= min_improvement:
                    plateau_counter += 1
                else:
                    plateau_counter = 0

                if train_accuracy > best_train_accuracy + min_improvement:
                    best_train_accuracy = train_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Save best model
                if train_error_rate <= self.best_error:
                    self.best_error = train_error_rate
                    self.best_W = self.current_W.clone()
                    self._save_best_weights()

                # Early stopping checks
                if train_accuracy == 1.0:
                    print("\nReached 100% training accuracy")
                    break

                if patience_counter >= patience:
                    print(f"\nNo improvement for {patience} epochs")
                    break

                if plateau_counter >= max_plateau:
                    print(f"\nAccuracy plateaued for {max_plateau} epochs")
                    break

                prev_accuracy = train_accuracy

            self._save_model_components()
            return self.current_W.cpu(), error_rates

    def reconstruct_features(self, class_probs: torch.Tensor) -> torch.Tensor:
        """Reconstruct input features from class probabilities with dtype handling"""
        with torch.no_grad():
            # Ensure consistent dtype
            class_probs = class_probs.to(dtype=torch.float32)
            reconstructed = self._compute_inverse_posterior(class_probs)

            if hasattr(self, 'min_vals') and self.min_vals is not None:
                reconstructed = self._unscale_features(reconstructed)
                # Ensure output matches input dtype
                return reconstructed.to(dtype=self.weight_linear.dtype)
            return reconstructed.to(dtype=self.weight_linear.dtype)

    def evaluate(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            features: Input features
            labels: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        with torch.no_grad():
            # Forward pass
            if self.forward_model.model_type == "Histogram":
                class_probs, _ = self.forward_model._compute_batch_posterior(features)
            else:
                class_probs, _ = self.forward_model._compute_batch_posterior_std(features)

            # Get predictions and convert to numpy
            predictions = torch.argmax(class_probs, dim=1)
            predictions_np = predictions.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Convert to original class labels
            true_labels = self.forward_model.label_encoder.inverse_transform(labels_np)
            pred_labels = self.forward_model.label_encoder.inverse_transform(predictions_np)

            # Compute classification report and confusion matrix
            from sklearn.metrics import classification_report, confusion_matrix
            class_report = classification_report(true_labels, pred_labels)
            conf_matrix = confusion_matrix(true_labels, pred_labels)

            # Calculate test accuracy
            test_accuracy = (predictions == labels).float().mean().item()

            # Get training error rates from metrics history
            error_rates = self.metrics.get('forward_errors', [])
            if not error_rates and hasattr(self.forward_model, 'error_rates'):
                error_rates = self.forward_model.error_rates

            # Get reconstruction metrics
            reconstructed_features = self.reconstruct_features(class_probs)
            reconstruction_loss = self._compute_reconstruction_loss(
                features, reconstructed_features, reduction='mean'
            ).item()

            # Prepare results dictionary matching expected format
            results = {
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'error_rates': error_rates,
                'test_accuracy': test_accuracy,
                'reconstruction_loss': reconstruction_loss
            }

            # Format results as string for display/saving
            formatted_output = f"Results for Dataset: {self.forward_model.dataset_name}\n\n"
            formatted_output += f"Classification Report:\n{class_report}\n\n"
            formatted_output += "Confusion Matrix:\n"
            formatted_output += "\n".join(["\t".join(map(str, row)) for row in conf_matrix])
            formatted_output += "\n\nError Rates:\n"

            if error_rates:
                formatted_output += "\n".join([f"Epoch {i+1}: {rate:.4f}" for i, rate in enumerate(error_rates)])
            else:
                formatted_output += "N/A"

            formatted_output += f"\n\nTest Accuracy: {test_accuracy:.4f}\n"

            # Store formatted output in results
            results['formatted_output'] = formatted_output

            return results
