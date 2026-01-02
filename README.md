# Wind Field Reconstruction

A comprehensive deep learning framework for wind field reconstruction from sparse sensor measurements. This project implements and compares multiple deep learning architectures (U-Net, ViTAE, CWGAN) with traditional Kriging interpolation methods for reconstructing wind velocity fields from sparse sensor observations.

## Overview

This project addresses the challenge of reconstructing complete wind velocity fields from sparse sensor measurements. The framework processes raw CSV data containing wind velocity measurements, generates training datasets with both normal and perturbed sensor configurations, and trains multiple models to reconstruct full 15×15 grid velocity fields.

## Project Structure

```
.
├── dataset/                    # Raw CSV data and generated training data
│   ├── *.csv                  # Raw wind field data files
│   └── *.npy                  # Generated training data (X, y, perturbed)
├── model_standard/            # Trained model weights
│   ├── 0_U-Net-*.keras
│   ├── 0_ViTAE_base-*.keras
│   ├── 0_CWGAN-Generator-*.keras
│   └── 1_*.keras              # Method 1 models
├── data_generate.ipynb        # Data preprocessing and generation pipeline
└── windreconstruction.ipynb   # Model training and evaluation
```

## Features

- **Multi-Model Comparison**: Implements U-Net, ViTAE, and CWGAN architectures alongside Kriging interpolation
- **Robustness Testing**: Evaluates models on both normal and perturbed sensor configurations
- **Flexible Sensor Configurations**: Supports 5, 10, 15, 20, 25, and 30 sensor configurations
- **Optimal Sensor Placement**: Supports both optimal and uniform sensor position strategies
- **Comprehensive Evaluation**: Multiple metrics including MSE, SSIM, PSNR, and vector field metrics

## Data Format

### Input Data (X)
- **Shape**: `(N, 15, 15, 3)`
  - Channel 0: Interpolated U-component velocity field
  - Channel 1: Interpolated V-component velocity field
  - Channel 2: Sensor position mask (1 at sensor locations, 0 elsewhere)

### Output Data (y)
- **Shape**: `(N, 15, 15, 2)`
  - Channel 0: Ground truth U-component velocity field
  - Channel 1: Ground truth V-component velocity field

## Installation

### Requirements

```bash
pip install numpy pandas scipy matplotlib tensorflow scikit-learn scikit-image pykrige tqdm
```

### Key Dependencies

- `tensorflow` >= 2.x
- `numpy`
- `pandas`
- `scipy`
- `pykrige` (for Kriging interpolation)
- `scikit-image`

## Usage

### 1. Data Generation

The `data_generate.ipynb` notebook processes raw CSV files and generates training datasets:

**Key Functions:**
- `generate_data_csv()`: Converts CSV data to training format with sensor interpolation
- `generate_perturbed_test_data()`: Creates perturbed sensor position datasets for robustness testing
- `load_optimal_sensors()`: Loads pre-computed optimal sensor positions
- `get_uniform_sensor_positions_center()`: Generates uniform sensor distributions


### 2. Model Training

The `windreconstruction.ipynb` notebook contains the training pipeline:

**Implemented Models:**

1. **U-Net**: Convolutional encoder-decoder architecture with skip connections
   - Input: `(15, 15, 3)`
   - Output: `(15, 15, 2)`
   - Loss: Weighted vector loss

2. **ViTAE**: Vision Transformer with Autoencoder architecture
   - Base model configuration
   - Loss: Mean squared error

3. **CWGAN**: Conditional Wasserstein GAN with gradient penalty
   - Generator: U-Net style architecture with noise input
   - Discriminator: Conditional discriminator
   - Loss: Wasserstein distance with gradient penalty

4. **Kriging**: Ordinary Kriging interpolation
   - Traditional geostatistical method
   - Optimized correlation length via variogram fitting


### 3. Evaluation

Models are evaluated on both normal and perturbed test sets using multiple metrics:

- **SSIM** (Structural Similarity Index): Measures structural similarity between predicted and ground truth fields using luminance, contrast, and structure comparisons
- **NMSE** (Normalized Mean Squared Error): Calculated as `mean((exp - pred)²) / (mean(exp) × mean(pred))`, providing scale-invariant error measurement
- **MG** (Geometric Mean Bias): Calculated as `exp(mean(ln(exp) - ln(pred)))`, where 1 indicates no bias. Values >1 indicate overestimation, <1 indicate underestimation
- **FAC2** (Fraction of predictions within a factor of 2): Percentage of data points satisfying `0.5 ≤ pred/exp ≤ 2` or both values are within a small threshold W

Results are saved as `.npy` files in the metrics and predictions directories.

## Data Splits

The project supports two data splitting methods:

- **Method 0**: Train on 0° data, test on 22° and 45° data
- **Method 1**: Train on first CSV files from each angle, test on remaining files

## Sensor Configurations

The framework supports multiple sensor counts:
- 5, 10, 15, 20, 25, 30 sensors

**Training configurations:**
- **Optimal sensor positions**: Pre-computed optimal placements
- **Uniform distribution**: Centered uniform grid sampling

**Test configurations:**
- **Normal test set**: Uses the same sensor positions as training
- **Perturbed test set**: Sensor positions randomly shifted by ±1 grid cell to evaluate model robustness to sensor placement errors

## Model Files

Trained models are saved in two directories:

**Standard models** (`model_standard/`):
- Models trained with standard sensor configurations
- Naming convention: `{method}_{ModelName}_{sensor_num}.keras`
- Examples:
  - `0_U-Net-ImageReconstruction_10.keras`
  - `0_ViTAE_base-ImageReconstruction_15.keras`
  - `0_CWGAN-Generator_20.keras`

**QR-optimized models** (`model_opt/`):
- Models trained with QR decomposition optimized sensor positions
- Same naming convention as standard models
- These models use sensor positions optimized via QR decomposition for improved reconstruction performance

## Citation

If you use this code in your research, please cite:

```bibtex
@software{wind_reconstruction,
  title={Wind Field Reconstruction from Sparse Sensor Measurements},
  author={Yihang Zhou, Chao Lin, Hideki Kikumoto, Ryozo Ooka, Sibo Cheng},
  year={2025},
  url={https://github.com/Yng314/wind-reconstruction}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
