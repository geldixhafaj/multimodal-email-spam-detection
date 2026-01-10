# Multimodal Email Spam Detection with Explainable AI

This repository contains the experimental and practical implementation for a research study on **multimodal email spam detection**, combining transformer-based text representations with structural email features and explainable AI techniques.

The work supports the practical chapter (Chapter 3) of an MSc dissertation in Cyber Security and is structured to ensure **reproducibility, transparency, and licensing compliance**.

---

## Research Objective

The primary objective of this study is to evaluate whether augmenting transformer-based text classifiers with non-semantic structural features improves spam detection performance, while maintaining interpretability.

Specifically, the study compares:

- A **baseline text-only model** using DistilBERT
- A **multimodal model** combining transformer embeddings with engineered structural features
- Explainability outputs using attention visualization and post-hoc methods (SHAP / LIME)

---

## Methodology Overview

The experimental pipeline consists of the following stages:

1. Dataset acquisition and exploratory data analysis (EDA)
2. Fine-tuning a DistilBERT baseline classifier
3. Extraction of transformer embeddings
4. Structural feature engineering (e.g., length, URLs, digits, uppercase ratio)
5. Early-fusion multimodal classification using an MLP
6. Model evaluation and robustness analysis
7. Explainable AI analysis (global and local explanations)

All experiments were executed in **Google Colab** using Python, with GPU acceleration enabled where available.

---

## Repository Structure

```text
multimodal-email-spam-detection/
├── assets/
│   ├── figures/
│   └── tables/
│
├── configs/
│   ├── baseline_distilbert.json
│   ├── feature_extraction.json
│   └── multimodal_mlp.json
│
├── notebooks/
│   ├── 01_download_data.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_baseline_distilbert_train.ipynb
│   ├── 04_embeddings_extract.ipynb
│   ├── 05_structural_features.ipynb
│   ├── 06_multimodal_mlp_train.ipynb
│   ├── 07_explainability.ipynb
│   └── 08_results_tables_figures.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── feature_utils.py
│   ├── metrics_utils.py
│   ├── seed_utils.py
│   └── viz_utils.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

```
## Dataset Access

Due to licensing restrictions, **datasets are not included** in this repository.

The datasets used in this study are publicly available on Kaggle and must be downloaded manually by users wishing to reproduce the experiments.

Datasets used:

- Spam Emails
  https://www.kaggle.com/datasets/abdallahwagih/spam-emails

Users must comply with the original dataset licenses when accessing and using the data.

After downloading, datasets should be placed locally in the `data/` directory.

---

## Reproducibility Instructions

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
Download datasets from Kaggle and place them in the data/ directory

Execute notebooks sequentially in the order defined by their numeric prefix

All paths are relative and compatible with Google Colab.

## Evaluation Metrics

The models are evaluated using the following metrics:

1. Accuracy
2. Precision
3. Recall
4. F1-score (primary metric)
5. ROC-AUC

## Confusion matrices

Robustness is assessed via seed variation and threshold sensitivity analysis.

## Explainability

Model interpretability is addressed using:

1. Attention visualization for transformer-based models
2. SHAP / LIME for post-hoc explanation of multimodal predictions
3. Global feature importance analysis
4. Local case studies for true positives and misclassifications

Explainability figures are exported to the assets/figures/ directory and referenced in the dissertation.

## Computational Environment

1. Python 3.x
2. Google Colab
3. GPU acceleration (where available)
4. Training time and inference latency may vary depending on runtime configuration.

## License

This repository is licensed under the MIT License.

The license applies only to the code in this repository and does not cover third-party datasets.

## Academic Use and Citation

This repository is intended for academic and research purposes.

If you use or adapt this work, please cite the corresponding dissertation or related publication.

Author
Geldi Xhafaj
MSc Cyber Security
Northumbria University
