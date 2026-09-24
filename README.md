# PG-PointConv: Physics-Guided Point Cloud Learning for Aging-friendly Renovations

An end-to-end network designated as Physics-Guided PointConv (PG-PointConv) that integrates physical information for identifying hidden three-dimensional physical obstacles in aging housing[cite: 1].

## Overview
The indoor spaces of aging housing are highly unstructured. Existing methods struggle to accurately identify hidden 3D physical obstacles, lacking quantitative calculations for renovation priorities[cite: 1]. 

PG-PointConv operates without explicit ICP preprocessing and learns local-to-global representations from acquired point sets[cite: 1]. It dynamically reshapes local convolutional weights using a feature difference attention mechanism and embeds high-risk physical properties via structural tensor priors[cite: 1]. A physics-guided risk loss function smoothly maps discrete obstacle classification into a continuous risk measure equation[cite: 1].

## Project Structure
- `models/`: Core network architecture (PG-PointConv, Structural Tensor, Feature Difference Attention).
- `risk/`: Post-processing logic (DBSCAN region generation, geometric measurement, physical risk target computation).
- `data/`: Max-pooling voxelization and dataset loaders.
- `train.py`: End-to-end training loop with composite penalty loss.
- `eval.py`: Inference script for converting point clouds into a 0-10 continuous risk score and 3-level intervention priority sequence.

## Installation
Dependencies: PyTorch 2.0+, CUDA 11.8+, Python 3.10+.
```bash
git clone [https://github.com/YourUsername/PG-PointConv.git](https://github.com/YourUsername/PG-PointConv.git)
cd PG-PointConv
pip install -r requirements.txt
