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



```
