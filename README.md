
![image](https://github.com/user-attachments/assets/fe1e97fe-a4f7-4e10-b433-6e00f5940a3b)

adbnn Algorithm
```
For each adaptive round:
    # Inner Training Loop
    While not (converged OR max_epochs reached):
        - Train only on training data
        - Update weights based on failed examples in training data:
            weight_update = learning_rate * (1 - P1/P2)
            where P1 = posterior prob for true class
                  P2 = posterior prob for wrongly predicted class
        - Check convergence criteria:
            * All training examples correct OR
            * Training accuracy plateaus for patience iterations OR
            * Max epochs reached
    
    # Testing Phase
    - Test on all non-training data
    - For each class in failed test examples:
        a) Find example with max wrong posterior (P2):
           if P2 margin > strong_margin_threshold:
              Add if cardinality low and divergence > min_divergence
        b) Find example with min wrong posterior:
           if P2 margin < marginal_margin_threshold:
              Add if cardinality low and divergence > min_divergence
    
    # Save Split if Improved
    If test_accuracy > best_test_accuracy:
        - Save current training data (before adding new samples)
        - Save current test data (before removing new train examples)
```
<svg fill="none" viewBox="0 0 800 400" width="800" height="400" xmlns="http://www.w3.org/2000/svg">
  <foreignObject width="100%" height="100%">
    <div xmlns="http://www.w3.org/2000/svg">
    <!DOCTYPE html>
<html>
<head>
<style>
.image-pair { display: inline-block; margin: 10px; text-align: center; }
.image-pair img { width: 128px; height: 128px; margin: 5px; }
</style>
</head>
<body>
<h1>Training Reconstructions - Epoch 2</h1>
<div class="image-pair">
<p>batch_0_sample_0</p>
<img src="batch_0_sample_0_original.png" alt="Original">
<img src="batch_0_sample_0_reconstruction.png" alt="Reconstruction">
</div>
<div class="image-pair">
<p>batch_0_sample_1</p>
<img src="batch_0_sample_1_original.png" alt="Original">
<img src="batch_0_sample_1_reconstruction.png" alt="Reconstruction">
</div>
<div class="image-pair">
<p>batch_0_sample_2</p>
<img src="batch_0_sample_2_original.png" alt="Original">
<img src="batch_0_sample_2_reconstruction.png" alt="Reconstruction">
</div>

</body>
</html>
    </div>
  </foreignObject>
</svg>

``` json

// 1. Main Configuration (dataset_name.json)
{
    "dataset": {
        "name": "sample_dataset",
        "type": "custom",
        "in_channels": 3,
        "input_size": [224, 224],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "image_type": "general"  // Options: "general", "astronomical", "medical", "agricultural"
    },
    "model": {
        "encoder_type": "enhanced",
        "feature_dims": 128,
        "learning_rate": 0.001,
        "enhancement_modules": {
            "astronomical": {
                "enabled": false,
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
                "enabled": false,
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
                "enabled": false,
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
        },
        "loss_functions": {
            "base_autoencoder": {
                "enabled": true,
                "weight": 1.0
            },
            "astronomical_structure": {
                "enabled": false,
                "weight": 1.0,
                "components": {
                    "edge_preservation": true,
                    "peak_preservation": true,
                    "detail_preservation": true
                }
            },
            "medical_structure": {
                "enabled": false,
                "weight": 1.0,
                "components": {
                    "boundary_preservation": true,
                    "tissue_contrast": true,
                    "local_structure": true
                }
            },
            "agricultural_pattern": {
                "enabled": false,
                "weight": 1.0,
                "components": {
                    "texture_preservation": true,
                    "damage_pattern": true,
                    "color_consistency": true
                }
            }
        }
    },
    "training": {
        "batch_size": 32,
        "epochs": 20,
        "num_workers": 4,
        "enhancement_specific": {
            "feature_extraction_frequency": 5,
            "pattern_validation_steps": 100,
            "adaptive_weight_adjustment": true
        }
    }
}

// 2. Dataset Configuration (dataset_name.conf)
{
    "file_path": "data/cifar100/cifar100.csv",   // Path to save extracted features
    "column_names": ["feature_0", ..., "target"], // Column names in CSV
    "separator": ",",
    "has_header": true,
    "target_column": "target",
    "modelType": "Histogram",                     // Type of model for DBNN

    "feature_group_size": 2,                      // Size of feature groups for analysis
    "max_combinations": 1000,                     // Maximum feature combinations to consider
    "bin_sizes": [21],                           // Bin sizes for histogram analysis

    "active_learning": {
        "tolerance": 1.0,                         // Tolerance for active learning
        "cardinality_threshold_percentile": 95,   // Percentile for cardinality threshold
        "strong_margin_threshold": 0.3,           // Threshold for strong margin
        "marginal_margin_threshold": 0.1,         // Threshold for marginal cases
        "min_divergence": 0.1                     // Minimum divergence threshold
    }
}

// 3. DBNN Configuration (adaptive_dbnn.conf)
{
    "training_params": {
        "trials": 100,                            // Number of training trials
        "epochs": 1000,                           // Maximum epochs per trial
        "learning_rate": 0.1,                     // Learning rate for DBNN
        "test_fraction": 0.2,                     // Fraction of data for testing
        "random_seed": 42,
        "minimum_training_accuracy": 0.95,        // Minimum accuracy to accept training
        "cardinality_threshold": 0.9,             // Threshold for cardinality checks
        "cardinality_tolerance": 4,               // Tolerance for cardinality variations
        "n_bins_per_dim": 20,                     // Number of bins per dimension
        "enable_adaptive": true,                  // Enable adaptive binning
        "modelType": "Histogram",                 // Type of DBNN model
        "compute_device": "auto"                  // "auto", "cpu", or "cuda"
    },

    "execution_flags": {
        "train": true,                           // Enable training phase
        "train_only": false,                     // Run only training phase
        "predict": true,                         // Enable prediction phase
        "fresh_start": false,                    // Start fresh or use existing model
        "use_previous_model": true,              // Use previously trained model if available
        "gen_samples": false                     // Generate samples during training
    }
}


```
## NOTE:
## For inverse mode:
# Interactive mode:
python cdbnn.py

# Command line predict mode:
python cdbnn.py --mode predict --data car --data_type custom --invert-dbnn
```json
"active_learning": {
    "tolerance": 1.0,
    "cardinality_threshold_percentile": 95,
    "strong_margin_threshold": 0.3,    // Decrease to add more samples
    "marginal_margin_threshold": 0.1,  // Decrease to add more samples
    "min_divergence": 0.1
}
```
```
In the configuration, the margin thresholds for selecting samples are controlled by
these parameters in the "active_learning" section:

marginal_margin_threshold: This parameter controls the threshold for marginal
failures (cases where the model is less confident but still incorrect).
Lowe values (e.g., 0.1) will be more permissive and add more samples.
strong_margin_threshold: This parameter controls the threshold for substantial
failures (cases where the model is very confident but wrong).
Lower values  (e.g., 0.3) will include more samples.
```


```
The reconstruction process involves two key mappings:

The forward mapping (f): Features → Class Probabilities
The inverse mapping (g): Class Probabilities → Reconstructed Features

In the forward direction, we compute class probabilities using histogram-based likelihood estimation:
P(class|features) ∝ exp(∑ log(w_ij * h_ij(features)))
where:

w_ij are the learned weights
h_ij are the histogram bin probabilities
The sum is over feature pairs (i,j)

For reconstruction, we implement an inverse mapping that tries to recover the original features from these class probabilities. This inverse mapping uses both linear and nonlinear components:
g(p) = α * g_linear(p) + (1-α) * g_nonlinear(p)
where:

p is the vector of class probabilities
α is the attention weight (learned during training)
g_linear is a linear transformation: W_l * p + b_l
g_nonlinear is a nonlinear transformation: tanh(W_nl * p + b_nl)

The reconstruction is trained to minimize two objectives:

Reconstruction Loss: ||x - g(f(x))||²
This measures how well we can recover the original features
Forward Consistency Loss: ||f(g(p)) - p||²
This ensures reconstructed features produce similar class probabilities

The total loss is:
L = β * Reconstruction_Loss + (1-β) * Forward_Consistency_Loss
where β is the reconstruction weight (typically 0.5).
The reconstruction quality is then measured in three ways:

Feature-wise error: How close are the reconstructed values to the originals?
Classification consistency: Do the reconstructed features produce the same classifications?
Distribution matching: Do the reconstructed features maintain the same statistical properties?

This formalism creates a bidirectional mapping between feature space and probability space, allowing us to:

Understand what feature values led to specific classifications
Validate the model's learned representations
Generate new examples with desired classification properties

The reconstruction accuracy serves as a measure of how well the model has captured the underlying structure of the data, beyond just classification performance.


Certainly! Below is the formatted version of your Feature Extraction System User Manual in Markdown format, suitable for a GitHub README file:

```markdown
# Feature Extraction System User Manual

## Unit 1: Algorithm Description and Technical Details

### 1.1 System Overview

The system implements a hybrid feature extraction approach combining convolutional neural networks (CNNs) and autoencoders with specialized loss functions for enhanced feature detection. It's designed to process image datasets and extract meaningful features while preserving important visual characteristics like structures, colors, and morphological properties.

### 1.2 Core Algorithms

#### 1.2.1 CNN Feature Extractor

The CNN-based feature extractor employs a hierarchical architecture:

```
Input → Conv1(32) → BN → ReLU → MaxPool 
     → Conv2(64) → BN → ReLU → MaxPool 
     → Conv3(128) → BN → ReLU → AdaptiveAvgPool 
     → Linear(feature_dims) → BatchNorm
```

**Mathematical formulation for each layer:**
- **Convolution:** `F(x) = W * x + b`
- **Batch Normalization:** `y = γ(x-μ)/σ + β`
- **ReLU:** `f(x) = max(0,x)`
- **MaxPool:** `y[i] = max(x[i:i+k])`

#### 1.2.2 Autoencoder Architecture

The autoencoder implements a symmetric architecture with dynamic sizing:

**Encoder:**
```
Input → Conv(32) → BN → LeakyReLU 
     → Conv(64) → BN → LeakyReLU 
     → Conv(128) → BN → LeakyReLU 
     → Linear(feature_dims)
```

**Decoder:**
```
Linear(feature_dims) → ConvTranspose 
                    → BN → LeakyReLU 
                    → ConvTranspose 
                    → Output
```

#### 1.2.3 Loss Functions

The system implements multiple specialized loss functions:

1. **Structural Loss:**
   ```
   L_struct = MSE(x,x̂) + λ₁‖∇x - ∇x̂‖₂ + λ₂TV(x̂)
   ```
   where `∇` is the Sobel gradient operator and `TV` is total variation.

2. **Color Enhancement Loss:**
   ```
   L_color = MSE(x,x̂) + λ₁‖Corr(x) - Corr(x̂)‖₂ + λ₂‖σ(x) - σ(x̂)‖₂
   ```
   where `Corr` computes channel correlations and `σ` is channel-wise standard deviation.

3. **Morphology Loss:**
   ```
   L_morph = MSE(x,x̂) + λ₁‖M(x) - M(x̂)‖₂ + λ₂(Sym_h(x̂) + Sym_v(x̂))
   ```
   where `M` computes moment statistics and `Sym_{h,v}` measure horizontal and vertical symmetry.

### 1.3 Data Flow

1. **Input Processing:**
   - Image loading and preprocessing
   - Resolution standardization
   - Channel normalization
   - Data augmentation (if enabled)

2. **Feature Extraction:**
   - Forward pass through selected architecture
   - Loss computation and backpropagation
   - Feature vector generation
   - Dimensionality reduction

3. **Output Generation:**
   - Feature vector serialization
   - CSV file generation
   - Configuration file creation
   - Optional visualization generation

### 1.4 Memory Management

The system implements several memory optimization techniques:
- Chunked data processing
- Garbage collection triggers
- GPU memory clearing
- Batch processing with configurable sizes

---

## Unit 2: Configuration Parameters

### 2.1 Model Configuration

#### Basic Parameters
- **`encoder_type`:** CNN or Autoencoder selection
  - **Effects:** Determines feature extraction approach
  - **Values:** `"cnn"` or `"autoenc"`

- **`feature_dims`:** Output feature dimensionality
  - **Effects:** Controls compression level
  - **Range:** `32-1024` (recommended)

#### Loss Function Parameters

1. **Structural Loss:**
   ```json
   "structural": {
       "enabled": true,
       "weight": 0.7,
       "params": {
           "edge_weight": 1.0,
           "smoothness_weight": 0.5
       }
   }
   ```
```
The configuration sample for cdbnn
```
```json
{
  "dataset": {
    "_comment": "Dataset configuration section",
    "name": "dataset_name",
    "type": "custom",                    // Options: "torchvision" or "custom"
    "in_channels": 3,                    // Number of input channels (3 for RGB, 1 for grayscale)
    "input_size": [224, 224],           // Input image dimensions [height, width]
    "num_classes": 10,                   // Number of classes in the dataset
    "mean": [0.485, 0.456, 0.406],      // Normalization mean values
    "std": [0.229, 0.224, 0.225],       // Normalization standard deviation values
    "train_dir": "path/to/train",       // Training data directory
    "test_dir": "path/to/test"          // Test data directory
  },

  "model": {
    "_comment": "Model architecture and training configuration",
    "encoder_type": "autoenc",          // Options: "cnn" or "autoenc"
    "feature_dims": 128,                // Dimension of extracted features
    "learning_rate": 0.001,             // Base learning rate

    "architecture": {
      "_comment": "Enhanced architecture components configuration",
      "use_global_convolution": true,   // Enable Global Convolution Network modules
      "use_boundary_refinement": true,  // Enable Boundary Refinement modules
      "gcn_kernel_size": 7,            // Kernel size for global convolution (odd numbers only)
      "feature_enhancement": {
        "enabled": true,               // Enable feature enhancement blocks
        "initial_channels": 32,        // Initial number of channels
        "growth_rate": 32             // Channel growth rate in dense blocks
      }
    },

    "loss_functions": {
      "_comment": "Loss function configuration section",
      "perceptual": {
        "enabled": true,
        "type": "PerceptualLoss",
        "weight": 1.0,
        "params": {
          "l1_weight": 1.0,           // Weight for L1 loss component
          "ms_ssim_weight": 1.0,      // Weight for MS-SSIM loss component
          "edge_weight": 0.5          // Weight for edge-awareness loss component
        }
      },
      "structural": {
        "enabled": true,
        "type": "StructuralLoss",
        "weight": 0.7,
        "params": {
          "edge_weight": 1.0,         // Weight for edge detection loss
          "smoothness_weight": 0.5,   // Weight for smoothness preservation
          "boundary_weight": 0.3      // Weight for boundary enhancement
        }
      },
      "color_enhancement": {
        "enabled": true,
        "type": "ColorEnhancementLoss",
        "weight": 0.5,
        "params": {
          "channel_weight": 0.5,      // Weight for channel correlation
          "contrast_weight": 0.3      // Weight for contrast preservation
        }
      },
      "morphology": {
        "enabled": true,
        "type": "MorphologyLoss",
        "weight": 0.3,
        "params": {
          "shape_weight": 0.7,        // Weight for shape preservation
          "symmetry_weight": 0.3      // Weight for symmetry preservation
        }
      }
    },

    "optimizer": {
      "_comment": "Optimizer configuration",
      "type": "Adam",                 // Options: "Adam", "SGD"
      "weight_decay": 1e-4,           // L2 regularization factor
      "momentum": 0.9,                // Momentum for SGD
      "beta1": 0.9,                   // Adam beta1 parameter
      "beta2": 0.999,                 // Adam beta2 parameter
      "epsilon": 1e-8                 // Adam epsilon parameter
    },

    "scheduler": {
      "_comment": "Learning rate scheduler configuration",
      "type": "ReduceLROnPlateau",    // Options: "StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"
      "factor": 0.1,                  // Factor to reduce learning rate
      "patience": 10,                 // Epochs to wait before reducing LR
      "min_lr": 1e-6,                // Minimum learning rate
      "verbose": true                 // Print learning rate updates
    }
  },

  "training": {
    "_comment": "Training process configuration",
    "batch_size": 32,                // Batch size for training
    "epochs": 20,                    // Number of training epochs
    "num_workers": 4,                // Number of data loading workers
    "checkpoint_dir": "checkpoints", // Directory to save checkpoints
    "validation_split": 0.2,         // Fraction of data used for validation

    "early_stopping": {
      "patience": 5,                 // Epochs to wait before early stopping
      "min_delta": 0.001            // Minimum improvement required
    },

    "loss_weights": {
      "_comment": "Weights for combining different losses",
      "perceptual": 1.0,
      "structural": 0.7,
      "reconstruction": 0.5
    }
  },

  "augmentation": {
    "_comment": "Data augmentation configuration",
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
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225]
    }
  },

  "execution_flags": {
    "_comment": "Execution control flags",
    "mode": "train_and_predict",     // Options: "train_only", "predict_only", "train_and_predict"
    "use_gpu": true,                 // Use GPU if available
    "mixed_precision": true,         // Use mixed precision training
    "distributed_training": false,   // Enable distributed training
    "debug_mode": false,            // Enable debug logging
    "use_previous_model": true,     // Load previous checkpoint if available
    "fresh_start": false           // Ignore existing checkpoints
  },

  "logging": {
    "_comment": "Logging configuration",
    "log_dir": "logs",             // Directory for log files
    "tensorboard": {
      "enabled": true,
      "log_dir": "runs"           // Directory for tensorboard logs
    },
    "save_frequency": 5,           // Save checkpoint every N epochs
    "metrics": [                   // Metrics to track
      "loss",
      "accuracy",
      "reconstruction_error"
    ]
  },

  "output": {
    "_comment": "Output configuration",
    "features_file": "features.csv",  // Path to save extracted features
    "model_dir": "models",           // Directory to save trained models
    "visualization_dir": "viz"       // Directory for visualizations
  }
}
```
   - **`edge_weight`:** Controls edge detection sensitivity
   - **`smoothness_weight`:** Balances region continuity

2. **Color Enhancement:**
   ```json
   "color_enhancement": {
       "enabled": true,
       "weight": 0.5,
       "params": {
           "channel_weight": 0.5,
           "contrast_weight": 0.3
       }
   }
   ```
   - **`channel_weight`:** Controls channel correlation importance
   - **`contrast_weight`:** Adjusts color contrast preservation

3. **Morphology:**
   ```json
   "morphology": {
       "enabled": true,
       "weight": 0.3,
       "params": {
           "shape_weight": 0.7,
           "symmetry_weight": 0.3
       }
   }
   ```
   - **`shape_weight`:** Controls shape preservation strength
   - **`symmetry_weight`:** Adjusts symmetry importance

### 2.2 Training Parameters

#### Basic Training
```json
"training_params": {
    "batch_size": 32,
    "epochs": 1000,
    "learning_rate": 0.1,
    "test_fraction": 0.2
}
```
- **Effects on training:**
  - Larger `batch_size`: Faster training, more memory usage
  - Higher `learning_rate`: Faster convergence but potential instability
  - More `epochs`: Better accuracy but longer training time

#### Advanced Parameters
```json
"training_params": {
    "minimum_training_accuracy": 0.95,
    "cardinality_threshold": 0.9,
    "cardinality_tolerance": 4,
    "enable_adaptive": true
}
```
- **Effects on model behavior:**
  - `minimum_training_accuracy`: Controls early stopping
  - `cardinality_threshold`: Affects feature discretization
  - `enable_adaptive`: Enables dynamic learning rate adjustment

### 2.3 Execution Parameters

```json
"execution_flags": {
    "train": true,
    "train_only": false,
    "predict": true,
    "fresh_start": false,
    "use_previous_model": true
}
```
- **Controls workflow:**
  - `train_only`: Training without prediction
  - `fresh_start`: Ignores previous checkpoints
  - `use_previous_model`: Enables transfer learning

### 2.4 Performance Optimization

1. **Memory Management:**
   ```json
   "training": {
       "batch_size": 32,
       "num_workers": 4
   }
   ```
   - Adjust based on available system resources
   - Larger values improve speed but increase memory usage

2. **GPU Utilization:**
   ```json
   "execution_flags": {
       "use_gpu": true,
       "mixed_precision": true
   }
   ```
   - Enable for faster training on compatible systems
   - `mixed_precision` reduces memory usage with minimal accuracy impact

### 2.5 Output Configuration

```json
"output": {
    "features_file": "path/to/output.csv",
    "model_dir": "path/to/models",
    "visualization_dir": "path/to/viz"
}
```
- Controls output organization
- Enables selective saving of models and visualizations
```




```
