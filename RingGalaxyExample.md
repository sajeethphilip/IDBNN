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

2025-03-17 02:09:21,540 - INFO - Starting model training...
2025-03-17 02:09:21,541 - INFO - Starting/Resuming Phase 1: Pure reconstruction training
2025-03-17 02:09:21,543 - INFO - Initialized new unified checkpoint
Phase 1 - Epoch 1: 100%|███████████████████████████████████████████████████████████████████████████████| 99/99 [00:24<00:00,  3.96it/s, loss=0.7696, best=inf]
2025-03-17 02:09:46,751 - INFO - Saved state phase1 to unified checkpoint
2025-03-17 02:09:46,752 - INFO - New best model for phase1 with loss: 0.7696
2025-03-17 02:09:46,752 - INFO - Phase 1 - Epoch 1: Loss = 0.7696, Best = 0.7696
Phase 1 - Epoch 2: 100%|████████████████████████████████████████████████████████████████████████████| 99/99 [00:25<00:00,  3.94it/s, loss=0.6646, best=0.7696]
2025-03-17 02:10:12,640 - INFO - Saved state phase1 to unified checkpoint
2025-03-17 02:10:12,640 - INFO - New best model for phase1 with loss: 0.6646
2025-03-17 02:10:12,640 - INFO - Phase 1 - Epoch 2: Loss = 0.6646, Best = 0.6646
Phase 1 - Epoch 3: 100%|████████████████████████████████████████████████████████████████████████████| 99/99 [00:24<00:00,  4.06it/s, loss=0.6643, best=0.6646]
2025-03-17 02:10:37,445 - INFO - Saved state phase1 to unified checkpoint
2025-03-17 02:10:37,446 - INFO - New best model for phase1 with loss: 0.6643
2025-03-17 02:10:37,446 - INFO - Phase 1 - Epoch 3: Loss = 0.6643, Best = 0.6643
Phase 1 - Epoch 4: 100%|████████████████████████████████████████████████████████████████████████████| 99/99 [00:23<00:00,  4.22it/s, loss=0.6637, best=0.6643]
2025-03-17 02:11:01,334 - INFO - Saved state phase1 to unified checkpoint
2025-03-17 02:11:01,335 - INFO - New best model for phase1 with loss: 0.6637
2025-03-17 02:11:01,335 - INFO - Phase 1 - Epoch 4: Loss = 0.6637, Best = 0.6637
....

2025-03-17 03:50:10,818 - INFO - Features saved to data/galaxies/galaxies.csv
2025-03-17 03:50:10,818 - INFO - Total features saved: 135
2025-03-17 03:50:10,819 - INFO - Processing completed successfully!
DBNN Dataset Processor
========================================
usage: adbnn.py [-h] [--file_path [FILE_PATH]] [--mode {train,train_predict,invertDBNN}]

Process ML datasets

optional arguments:
  -h, --help            show this help message and exit
  --file_path [FILE_PATH]
                        Path to dataset file or folder
  --mode {train,train_predict,invertDBNN}
                        Mode to run the network: train, train_predict, or invertDBNN.
usage: adbnn.py [-h] [--file_path [FILE_PATH]] [--mode {train,train_predict,invertDBNN}]

Process ML datasets

optional arguments:
  -h, --help            show this help message and exit
  --file_path [FILE_PATH]
                        Path to dataset file or folder
  --mode {train,train_predict,invertDBNN}
                        Mode to run the network: train, train_predict, or invertDBNN.
============================================================
Dataset: galaxies
Config file: data/galaxies/galaxies.conf
Data file: data/galaxies/galaxies.csv
============================================================
Dataset Information:
Dataset name: galaxies
Configuration file: data/galaxies/galaxies.conf (4.4 KB)
Data file: data/galaxies/galaxies.csv (18691.4 KB)
Model type: Histogram
Target column: target
Number of columns: 129
Excluded features: 0
Features:
  feature_0
  feature_1
  feature_2
  feature_3
  feature_4
  ... and 124 more

Process this dataset? (y/n): y
Processing dataset: galaxies
Loaded best weights from Model/Best_Histogram_galaxies_weights.json
Using device: cuda
Using original data order (no shuffling required)
Loaded best weights from Model/Best_Histogram_galaxies_weights.json
Using original data order (no shuffling required)
[DEBUG] Generating feature combinations after filtering out features with high cardinality set by the conf file:
- n_features: 128
- group_size: 2
- max_combinations: 10000
- bin_sizes: [21]
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loading cached feature combinations from data/galaxies/training_data/galaxies/feature_combinations.pkl
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loaded feature combinations: torch.Size([1830, 2])
Dataset shape: torch.Size([12632, 128])
Bin sizes: [64]
Using original data order (no shuffling required)
Adaptive training started at: 2025-03-17 03:50:34
target
 Initial data shape: X=(12632, 129), y=12632
Number of classes in data = [0 1]
<bound method NDFrame.head of        feature_0  feature_1  feature_2  feature_3  feature_4  feature_5  ...  feature_124  feature_125  feature_126  feature_127  target  original_index
0       0.038956  -0.116702  -0.036961   0.119280  -0.085445   0.049149  ...     0.280519    -0.031874    -0.003397     0.052938       0               0
1       0.638382  -0.032868  -0.038245   0.089761  -0.202669   0.525085  ...     0.558681    -0.018332    -0.024747     0.557497       0               1
2       0.492175  -0.170648  -0.038416   0.085148  -0.200476   0.378307  ...     0.414804    -0.012266    -0.063049     0.523771       0               2
3       0.730011  -0.149996  -0.039406   0.090917  -0.247599   0.605390  ...     0.709854    -0.015759    -0.021829     0.796731       0               3
4       0.487268  -0.081182  -0.038580   0.098442  -0.149054   0.397495  ...     0.571912    -0.031604    -0.004842     0.546710       0               4
...          ...        ...        ...        ...        ...        ...  ...          ...          ...          ...          ...     ...             ...
12627   0.313439  -0.308851  -0.030555   0.091439  -0.263172   0.248972  ...     0.584872    -0.044785     0.008434     0.318932       1           12627
12628   0.303588  -0.102883  -0.038293   0.107569  -0.013486   0.253513  ...     0.544572    -0.012600    -0.002367     0.439162       0           12628
12629   0.415428  -0.255094  -0.034059   0.084471  -0.032345   0.361214  ...     0.624342    -0.039793     0.023227     0.443271       1           12629
12630   0.192983  -0.214469  -0.041186   0.091015   0.337450   0.118773  ...     0.242265    -0.057006    -0.039371     0.326471       0           12630
12631   0.501341  -0.088689  -0.039442   0.091163  -0.034817   0.405114  ...     0.632384    -0.028419    -0.009838     0.541648       0           12631

[12632 rows x 130 columns]>
[DEBUG] Generating feature combinations after filtering out features with high cardinality set by the conf file:
- n_features: 128
- group_size: 2
- max_combinations: 10000
- bin_sizes: [21]
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loading cached feature combinations from data/galaxies/training_data/galaxies/feature_combinations.pkl
---------------------BEWARE!! Remove if you get Error on retraining------------------------
[DEBUG] Loaded feature combinations: torch.Size([1830, 2])
Dataset shape: torch.Size([12632, 128])
Bin sizes: [64]
Loading previous model state
Loaded best weights from Model/Best_Histogram_galaxies_weights.json
Loading previous training data...
Loaded 0 previous training samples
Initializing new training set with minimum samples
Round 1/20
Training set size: 4
Test set size: 12628
Training epochs:   0%|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 0/1000 [00:00<?, ?it/s Prediction batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.63it/s]
Training epochs:   0%|▌                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 1/1000 [00:00<13:46,  1.21it/s, train_err=0.5000 (best: 0.5000), train_acc=0.5000 (best: 0.5000) Prediction batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.67it/s]
Training epochs:   0%|█                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 2/1000 [00:02<26:12,  1.58s/it, train_err=0.2500 (best: 0.2500), train_acc=0.7500 (best: 0.7500) Prediction batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.67it/s]
Training epochs:   0%|█▌                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 3/1000 [00:04<26:43,  1.61s/it, train_err=0.2500 (best: 0.2500), train_acc=0.7500 (best: 0.7500) Prediction batches: 
...


```
