#!/bin/bash

# Prompt for data name
read -p "Enter data name (e.g., galaxies): " data_name

# Prompt for mode (train/predict/all)
read -p "Enter mode (train/predict/all): " mode



# Git pull
git pull

# Run operations based on mode
case $mode in
  "train" | "all")
      # Cleanup
    rm -rf "data/${data_name}/Model/"
    rm -rf Model/*"${data_name}"_*
    echo "Running training..."
    python DynamicCNN.py train "$data_name"
    python adbnn.py --file_path "data/${data_name}/${data_name}.csv" --mode train
    ;;&  # Continue to next case (executes predict if mode is "all")
  "predict" | "all")
    echo "Running prediction..."
    python DynamicCNN.py predict "$data_name"
    python adbnn.py --file_path "data/${data_name}/${data_name}.csv" --mode predict
    ;;
  *)
    echo "Invalid mode. Choose 'train', 'predict', or 'all'"
    exit 1
    ;;
esac

echo "Pipeline completed!"
