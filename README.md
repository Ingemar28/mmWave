# mmWave

## Overview

This repository provides code and utilities for preprocessing, analyzing, and classifying point cloud data captured by mmWave radars. The main goal is to process raw radar data to detect the presence and actions (such as "sit", "stand", "hybrid") of individuals at different tables using advanced deep learning models. The codebase was developed as part of research described in an academic paper.

## Features

- **Preprocessing:** Removes clutter, applies noise removal (DBSCAN and static point filtering), computes density and centroids, and structures the data for downstream ML tasks.
- **Feature Extraction:** Calculates point cloud density, centroids, and filters data within table boundaries.
- **Data Handling:** Aggregates, pads, and saves processed data for consistent input to deep learning models.
- **Modeling:** Implements PointNet and LSTM-based neural networks to classify actions and detect presence using raw point clouds and derived features.
- **Evaluation:** Provides scripts for model training, cross-validation, and performance analysis (accuracy, confusion matrix).
- **Visualization:** Includes utilities for plotting loss curves and density distributions.

## Directory Structure

- `preprocess/`: Scripts and utilities for processing raw mmWave radar data.
- `training/`: Deep learning models for action classification and presence detection.
- `util/`: Utility functions for data manipulation, clustering, and plotting.

## Main Pipeline

1. **Preprocessing (`preprocess/preprocess.py`):**
    - Loads raw JSON data from radar captures.
    - Removes noise using DBSCAN and static point filtering.
    - Extracts features: density, centroids, and filtered points within table areas.
    - Saves processed features as pickled DataFrames (`density_dfs.pkl`, `centroid_dfs.pkl`, `point_dfs.pkl`).

2. **Feature Handling:**
    - Pads or truncates point sets to a fixed number of points (for deep learning input consistency).
    - Aggregates features per table and per time block.

3. **Model Training and Evaluation (`training/`):**
    - `three_class_raw_points.py`: Classifies actions ("sit", "stand", "hybrid") using PointNet + BiLSTM models on raw point clouds.
    - `three_class_v_den_cen.py`: Hybrid models using raw points and derived features (density, centroid).
    - `presence.py`: Detects presence using LSTM models on density and centroid features.
    - Cross-validation and performance metrics are computed for each model.

4. **Utilities (`util/util.py`):**
    - Functions for clustering (DBSCAN), density/centroid calculation, static point removal, and plotting.

## Example Usage

### Preprocessing

```bash
python preprocess/preprocess.py
```

### Training a Model

```bash
python training/three_class_raw_points.py
```

### Presence Detection

```bash
python training/presence.py
```

## Dependencies

- Python 3.x
- numpy, pandas, matplotlib, scikit-learn
- tensorflow, keras
- pickle

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
```

## Data

- Place raw radar data as JSON files in the `data/` directory.
- Processed data will be saved as `.pkl` files in the same directory.

## Citation

If you use this codebase in your research or publication, please cite our paper:

```
[Add your paper citation here]
```

## Contact

For questions or collaboration, please contact [Ingemar28](https://github.com/Ingemar28).

---
This README was generated based on the repository's codebase for use in an academic publication.
