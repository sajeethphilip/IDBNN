```
>> git pull && python cdbnn.py && python adbnn.py

Enter mode (train/predict) [train]:

Enter dataset type (torchvision/custom) [custom]:
Enter dataset name/path [galaxies.zip]:
Enable inverse DBNN mode? (y/n) [y]:
Enter encoder type (cnn/autoenc) [cnn]:
Enter batch size [128]:
Enter number of epochs [20]:
Enter output directory [data]:
2025-03-17 01:43:32,564 - INFO - Extracting compressed file: galaxies.zip
Create train/test split? (y/n): n
2025-03-17 01:43:42,423 - INFO - Dataset processed: train_dir=data/galaxies/train, test_dir=None
2025-03-17 01:43:42,423 - INFO - Generating/verifying configurations...
2025-03-17 01:43:42,423 - INFO - Starting configuration generation for dataset: galaxies
2025-03-17 01:43:42,423 - INFO - Generating main configuration...
2025-03-17 01:43:42,424 - INFO - Using image properties from config file
2025-03-17 01:43:42,487 - INFO - Found existing main config, merging...
2025-03-17 01:43:42,488 - INFO - Main configuration saved: data/galaxies/galaxies.json
2025-03-17 01:43:42,488 - INFO - Generating dataset configuration...
2025-03-17 01:43:42,488 - INFO - Found existing dataset config, merging...
2025-03-17 01:43:42,489 - INFO - Dataset configuration saved: data/galaxies/galaxies.conf
2025-03-17 01:43:42,489 - INFO - Generating DBNN configuration...
2025-03-17 01:43:42,489 - INFO - Found existing DBNN config, merging...
2025-03-17 01:43:42,489 - INFO - DBNN configuration saved: data/galaxies/adaptive_dbnn.conf

Configuring General Enhancement Parameters:
Enable KL divergence clustering? (y/n) [y]:
Enter KL divergence weight (0-1) [0.1]:
Enable class encoding? (y/n) [y]:
Enter classification weight (0-1) [0.1]:
Enter clustering temperature (0.1-2.0) [1.0]:
Enter minimum cluster confidence (0-1) [0.7]:
Enable phase 2 training (clustering and fine-tuning)? (y/n) [y]:

Configuring Enhancement Features for General Mode:
You can enable any combination of features

Enable Astronomical features (star detection, galaxy structure preservation)? (y/n) [n]: y
Astronomical features added.
Enable Medical features (tissue boundary, lesion detection)? (y/n) [n]:
Enable Agricultural features (texture analysis, damage detection)? (y/n) [n]:

Current Enhancement Configuration:

Astronomical Features:
- Components: structure_preservation, detail_preservation, star_detection, galaxy_features, kl_divergence
- Weights: detail_weight: 1.0, structure_weight: 0.8, edge_weight: 0.7

Learning Rates:
- Phase 1: 0.001
- Phase 2: 0.0005
2025-03-17 01:44:57,245 - INFO - Initializing general enhanced model...
2025-03-17 01:44:57,246 - INFO - Creating model with enhancements: astronomical, medical, agricultural
2025-03-17 01:44:57,246 - INFO - Layer sizes: [32, 64, 128, 256, 256, 256]

Ready to start training. Proceed? (y/n): y


```
