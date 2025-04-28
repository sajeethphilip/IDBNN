#!/bin/bash

# Prompt for data name
read -p "Enter data name (e.g., galaxies): " data_name

read -p "Enter data path (e.g., Data/galaxies/): " input_path

# Prompt for mode (train/predict/all)
read -p "Enter mode (train/predict/all/fresh): " mode


# Git pull
git pull

# Run operations based on mode
case $mode in
  "fresh")
    echo "Cleaning..."
    rm -rf "data/${data_name}/Model/"
    rm -rf Model/*"${data_name}"_*
    ;;&
  "train" | "all"| "fresh")
    
    echo "Running training..."
    python cdbnn.py --mode train --data_name "$data_name" --input_path "$input_path"
    python adbnn.py --file_path "data/${data_name}/${data_name}.csv" --mode train
    ;;&  # Continue to next case (executes predict if mode is "all")
  "predict" | "all" | "fresh")
    echo "Running prediction..."
    python cdbnn.py --mode predict --data_name "$data_name" --input_path "$input_path"
    python adbnn.py --file_path "data/${data_name}/${data_name}.csv" --mode predict
    ;;
  *)
    echo "Invalid mode. Choose 'train', 'predict', or 'all'"
    exit 1
    ;;
esac

echo "Pipeline completed!"
