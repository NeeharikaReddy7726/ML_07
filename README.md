# FIB-Fusion: Federated Information-Balanced Multimodal Learning

**Robust Multimodal Learning under Missing Modalities and Non-IID Client Drift**

## Overview

FIB-Fusion is a research project exploring robust multimodal learning in decentralized environments using Federated Learning (FL).

Traditional multimodal learning systems often assume that every participant has access to all modalities. However, real-world edge devices may have incomplete data due to hardware limitations, privacy restrictions, or missing sensor streams.

Additionally, multimodal neural networks can develop modality optimization imbalance, where dominant modalities such as text suppress weaker modalities such as audio and vision. Non-IID data distributions across clients further introduce client drift during federated optimization.

This project investigates a unified framework combining information-balanced learning, modality-aware aggregation, and FedProx regularization to address these challenges.

## Research Objectives

- Handle heterogeneous modality availability across federated clients.
- Mitigate modality optimization imbalance between text, audio, and visual representations.
- Reduce client drift caused by non-IID data distributions.
- Improve classification robustness under missing modality conditions.
- Investigate the effectiveness of information-balanced objectives in decentralized learning environments.

## Proposed Methodology

FIB-Fusion combines three principal components:

### 1. Information-Balanced Multimodal Learning (IBML)

Uses the Balance-Information Objective (BIO) and Task-Complexity Modulation (TCM) to prevent dominant modalities from suppressing weaker modalities during training.

### 2. Modality-Aware Aggregation

Performs modality-specific parameter aggregation at the server. Only clients possessing a particular modality contribute updates to its corresponding model parameters.

This prevents missing modality updates from contaminating the global model.

### 3. FedProx Regularization

Introduces a proximal penalty during local optimization to constrain client updates and mitigate divergence caused by non-IID data distributions.

## Model Architecture

The framework uses independent modality-specific encoders for:

- Text
- Audio
- Vision

Each available modality is projected into a shared latent representation. Decoupled linear classification heads generate modality-specific logits, which are combined to produce the final prediction.

Missing modalities are excluded from the forward pass and parameter aggregation.

## Datasets

The extended implementation explores four real-world multimodal datasets.

| Dataset | Task | Modalities |
|---|---|---|
| CMU-MOSI | Sentiment Analysis | Text, Audio, Vision |
| CMU-MOSEI | Sentiment Analysis | Text, Audio, Vision |

The research paper presents the primary experimental investigation on CMU-MOSI and CMU-MOSEI. The extended notebook additionally incorporates UR-FUNNY and MUStARD.

## Federated Learning Setup

- 10 simulated federated clients.
- Heterogeneous modality availability.
- Seven modality configurations: TAV, TA, TV, AV, T, A, V.
- Dirichlet-based non-IID label partitioning.
- Multiple random seeds for experimental evaluation.
- Local model optimization and server-side aggregation.

## Experimental Evaluation

The implementation includes comparisons against multiple baselines and ablation variants:

- Centralized Learning
- Centralized IBML
- Local-Only Training
- FedAvg
- FedProx
- Modality-Aware Aggregation
- Modality-Aware Aggregation + BIO
- Modality-Aware Aggregation + TCM
- Full IBML
- FedProx + IBML (FIB-Fusion)

Evaluation metrics include:

- Accuracy
- Macro-F1 Score
- Weighted-F1 Score
- Balanced Accuracy
- Missing-Modality F1
- Modality Discrepancy Ratio
- Communication Cost
- Convergence Behaviour

## Repository Contents

```text
FIB-Fusion/
│
├── FIB_Fusion_IBML_4Datasets.ipynb
│
└── README.md
```

## Implementation

The primary implementation is provided as a Google Colab-compatible Jupyter Notebook.

The notebook covers:

1. Dataset downloading and preprocessing.
2. Temporal mean pooling and feature normalization.
3. Non-IID federated client partitioning.
4. Modality-specific model construction.
5. BIO and TCM implementation.
6. FedProx local optimization.
7. Modality-aware server aggregation.
8. Baseline comparisons and ablation studies.
9. Multi-seed evaluation and result visualization.

### Running the Notebook

1. Open the notebook in Google Colab.
2. Configure the dataset and experiment settings.
3. Run the setup and data preparation cells.
4. Execute the unit checks and initial smoke test.
5. Run the complete experiments.
6. Examine the generated metrics, plots, and cross-dataset comparisons.

**Note:** The complete experiments require appropriate computational resources, particularly for CMU-MOSEI.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab
- Federated Learning
- Deep Learning

## Research Paper

The accompanying research paper presents the methodology, mathematical formulation, experimental setup, comparative analysis, and findings of FIB-Fusion.

## Project Team

- Meghna Pawle
- Neeharika Tera
- Nihali Khaire

**Faculty Guide:** Dr.Shankadeep Chaterjee
Department of Artificial Intelligence

## Acknowledgements

This work builds upon existing research in Federated Learning, Information-Balanced Multimodal Learning, and multimodal sentiment analysis.

The project uses established multimodal benchmarks, including CMU-MOSI, CMU-MOSEI.

---

**FIB-Fusion — Towards Robust, Balanced, and Decentralized Multimodal AI.**
