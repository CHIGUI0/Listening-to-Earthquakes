# Listening to Earthquakes: Predicting Laboratory Earthquake Time‑to‑Failure

## Project Overview

Real‑time earthquake early‑warning hinges on estimating *when* the next rupture will occur.
Laboratory experiments by LANL make this possible at small scale: as a sheared fault gouge repetitively fails, a piezo‑electric sensor records acoustic emissions.  Predicting TTF from these noisily‑measured signals has become a classic regression benchmark.

Our contribution is a **lightweight, reproducible framework** that:

* **Segments** the 26 GB raw waveform into manageable windows (five window/stride pairs from 150 k → 1.5 k samples).
* **Extracts 24 univariate statistics** per window and prunes them to the top‑10 most important features for each model using permutation importance.
* **Implements** five modelling families

  * Tree‑based: Random Forest
  * Convolutional: Temporal Convolutional Network (TCN)
  * Feed‑forward: Multi‑Layer Perceptron (MLP)
  * Recurrent: Long Short‑Term Memory (LSTM)
  * Attention‑based: Encoder‑only Transformer
* **Beats strong Kaggle baselines**—achieving a public MAE of **1.50** (private 2.59) with a 3‑layer LSTM on 1.5 k/0.75 k windows.

For full methodological details see [`CS_412_Project_Final_Report.pdf`](./CS_412_Project_Final_Report.pdf).

---

## Repository Structure

```
Listening-to-Earthquakes/
├── dataprocess/              # Sliding‑window & feature‑extraction notebooks/scripts
├── model_training/           # Training scripts & W&B logs for RF / TCN / MLP / LSTM / Transformer
├── upload_kaggle.sh          # Utility for batch submission to Kaggle leaderboard
├── README.md                 # (you are here)
└── CS_412_Project_Final_Report.pdf  # Full paper with experiments & discussion
```

---

## Quick Start

```bash
# 1. Clone & install
$ git clone https://github.com/CHIGUI0/Listening-to-Earthquakes.git
$ cd Listening-to-Earthquakes

# 2. Download raw data (≈3 GB) from Kaggle
import kagglehub

path = kagglehub.dataset_download("penguingui/listening-to-earthquakes")

# 3. Run the default LSTM experiment
$ bash model_training/train_LSTM.sh
```
