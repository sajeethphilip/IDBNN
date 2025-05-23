#!/bin/bash

# Prompt for data name
read -p "Enter data name (e.g., galaxies): " data_name

# Prompt for data path
read -p "Enter data path (e.g., Data/galaxies): " data_path

# Prompt for mode (train/predict/all)
read -p "Enter mode (train/predict/all/fresh): " mode

# Prompt for Feature extractor model (Jnet/CNN)
read -p "Enter Feature Extractor model (jnet/cnn): " FEmodel

# Prompt for Feature DBNN model (Jnet/CNN)
read -p "Enter DBNN model (Histogram/Gaussian): " DBNNmodel


# Git pull
git pull

# Run operations based on mode
case $mode in
  "fresh")
    echo "Cleaning..."
    rm -rf "data/${data_name}/"
    rm -rf Model/*"${data_name}"_*
    rm -rf training_data/${data_name}/
    ;;&
  "train" | "all"| "fresh")

    echo "Running training..."
    python DynamicCNN.py train "$data_path" --name "$data_name" --model_type "$FEmodel"
    python adbnn.py --file_path "data/${data_name}/${data_name}.csv" --mode train --model_type "$DBNNmodel"
    ;;&  # Continue to next case (executes predict if mode is "all")
  "predict" | "all" | "fresh")
    echo "Running prediction..."
    python DynamicCNN.py predict "$data_path" --name "$data_name" --model_type "$FEmodel"
    python adbnn.py --file_path "data/${data_name}/${data_name}.csv" --mode predict --model_type "$DBNNmodel"
    ;;
  *)
    echo "Invalid mode. Choose 'train', 'predict', or 'all'"
    exit 1
    ;;
esac

echo "Pipeline completed!"
