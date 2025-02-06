``` json

{
    // Basic dataset configuration
    "file_path": "data/dataset_name/dataset_name.csv",  // Path to the dataset file
    "column_names": [                    // List of column names in order
        "feature1",
        "feature2",
        "#feature3",                     // Prefix with # to exclude a feature
        "target"
    ],
    "separator": ",",                    // CSV separator character
    "has_header": true,                  // Whether CSV has header row
    "target_column": "target",           // Name of the target column

    // Model type and core settings
    "modelType": "Histogram",            // Options: "Histogram" or "Gaussian"

    // Feature processing configuration
    "likelihood_config": {
        "feature_group_size": 2,         // Size of feature groups (usually 2)
        "max_combinations": 1000,        // Maximum number of feature combinations
        "bin_sizes": [20]               // Bin sizes for histogram model. Single value applies to all dimensions
    },

    // Active learning parameters
    "active_learning": {
        "tolerance": 1.0,                // Learning tolerance for active learning
        "cardinality_threshold_percentile": 95,  // Percentile for cardinality threshold
        "strong_margin_threshold": 0.3,   // Threshold for strong classification margins
        "marginal_margin_threshold": 0.1, // Threshold for marginal classification margins
        "min_divergence": 0.1            // Minimum divergence for sample selection
    },

    // Training parameters
    "training_params": {
        // Core training parameters
        "trials": 100,                   // Number of training trials
        "epochs": 1000,                  // Maximum number of epochs
        "learning_rate": 0.1,            // Base learning rate
        "test_fraction": 0.2,            // Fraction of data for testing
        "random_seed": 42,               // Random seed for reproducibility
        "minimum_training_accuracy": 0.95, // Minimum accuracy before stopping training

        // Cardinality handling
        "cardinality_threshold": 0.9,    // Threshold for feature cardinality
        "cardinality_tolerance": 4,      // Decimal places for rounding features

        // Model specific parameters
        "n_bins_per_dim": 20,           // Number of bins per dimension for histogram
        "enable_adaptive": true,         // Enable adaptive learning

        // Inverse DBNN parameters (all default to false/0 if not specified)

        "epochs": 1000,
        "invert_DBNN": false,           // Enable inverse model functionality
        "reconstruction_weight": 0.5,    // Weight for reconstruction loss (0-1)
        "feedback_strength": 0.3,        // Strength of feedback between models (0-1)
        "inverse_learning_rate": 0.1,    // Learning rate for inverse model

        // Training data management
        "Save_training_epochs": false,   // Save data for each training epoch
        "training_save_path": "training_data"  // Path for saving training data
    },

    // Execution flags
    "execution_flags": {
        "train": true,                   // Enable model training
        "train_only": false,            // Only train, no prediction
        "predict": true,                // Enable prediction
        "fresh_start": false,           // Start training from scratch
        "use_previous_model": true      // Use previously saved model if available
    }
}

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
