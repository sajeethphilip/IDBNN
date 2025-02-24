import torch
import argparse
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List, Union
import os
import json
import pickle

# Import the new DBNN modules
from dbnn import (
    DBNN,
    DebugLogger,
    GlobalConfig,
    DBNNInitializer,
    DBNNDataHandler,
    DBNNTrainer,
    DBNNPredictor,
    DBNNReconstructor,
    DBNNUtils,
    DatasetProcessor,
    InvertibleDBNN
)

# Initialize debug logger
DEBUG = DebugLogger()



def main():
    parser = argparse.ArgumentParser(description='Process ML datasets with Adaptive DBNN')
    parser.add_argument("file_path", nargs='?', help="Path to dataset file or folder")
    args = parser.parse_args()

    # Initialize dataset processor
    processor = DatasetProcessor()

    if not args.file_path:
        # Handle interactive dataset selection
        try:
            dataset_pairs = processor.find_dataset_pairs()
            if dataset_pairs:
                for basename, conf_path, csv_path in dataset_pairs:
                    print(f"\nFound dataset: {basename}")
                    if input("\nProcess this dataset? (y/n): ").lower() == 'y':
                        processor.process_dataset(csv_path)
            else:
                print("\nNo datasets found in data folder")
        except KeyboardInterrupt:
            print("\nProcessing interrupted")
            return
    else:
        # Process specific dataset
        processor.process_dataset(args.file_path)

if __name__ == "__main__":
    main()
