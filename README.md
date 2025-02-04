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
